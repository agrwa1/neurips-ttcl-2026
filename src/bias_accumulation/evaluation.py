"""Inference and destination-count metrics for bias accumulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class EvaluationResult:
    predictions: pd.DataFrame
    logits: np.ndarray
    output_to_original_class: np.ndarray


def run_inference(
    backbone,
    classifier,
    dataset,
    partition,
    output_to_original_class: list[int],
    runtime_config: dict,
    evaluation_config: dict,
) -> EvaluationResult:
    import torch
    from torch.utils.data import DataLoader

    device = torch.device(runtime_config.get("device", "cuda"))
    backbone.to(device).eval()
    classifier.module.to(device).eval()
    worker_count = int(runtime_config.get("num_workers", 0))
    worker_options = {}
    if worker_count:
        worker_options["prefetch_factor"] = int(
            runtime_config.get("prefetch_factor", 1)
        )
    loader = DataLoader(
        dataset,
        batch_size=int(evaluation_config.get("batch_size", 256)),
        shuffle=False,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        timeout=int(runtime_config.get("dataloader_timeout_seconds", 120))
        if worker_count
        else 0,
        **worker_options,
    )
    output_to_original = np.asarray(output_to_original_class, dtype=np.int64)
    original_to_output = {
        int(value): index for index, value in enumerate(output_to_original)
    }
    class_to_task = partition.task_for_class()
    task_output_indices = {
        task_id: np.asarray(
            [
                index
                for index, class_id in enumerate(output_to_original)
                if class_to_task[int(class_id)] == task_id
            ],
            dtype=np.int64,
        )
        for task_id in range(partition.num_tasks)
    }

    row_chunks: list[pd.DataFrame] = []
    logit_chunks: list[np.ndarray] = []
    iterator = iter(loader)
    try:
        for sample_ids, images, original_targets in iterator:
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), _autocast(device, runtime_config):
                features = backbone(images)
                logits = classifier(features)
            logits_float = logits.float()
            probabilities = torch.softmax(logits_float, dim=1)
            confidence, predicted_output = probabilities.max(dim=1)
            top2 = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
            margin = top2[:, 0] - top2[:, 1] if top2.shape[1] == 2 else top2[:, 0]
            entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)

            logits_np = logits_float.cpu().numpy()
            probabilities_np = probabilities.cpu().numpy()
            predicted_output_np = predicted_output.cpu().numpy()
            predicted_original = output_to_original[predicted_output_np]
            true_original = np.asarray(original_targets, dtype=np.int64)
            true_output = np.asarray(
                [original_to_output[int(value)] for value in true_original]
            )
            true_probability = probabilities_np[
                np.arange(len(true_output)), true_output
            ]
            squared_probability_sum = np.square(probabilities_np).sum(axis=1)
            brier = squared_probability_sum - 2.0 * true_probability + 1.0

            chunk: dict[str, Any] = {
                "sample_id": list(sample_ids),
                "true_original_class": true_original,
                "true_output_index": true_output,
                "true_semantic_task": [
                    class_to_task[int(value)] for value in true_original
                ],
                "predicted_original_class": predicted_original,
                "predicted_output_index": predicted_output_np,
                "predicted_semantic_task": [
                    class_to_task[int(value)] for value in predicted_original
                ],
                "correct": predicted_original == true_original,
                "confidence": confidence.cpu().numpy(),
                "top_two_margin": margin.cpu().numpy(),
                "entropy": entropy.cpu().numpy(),
                "true_probability": true_probability,
                "negative_log_likelihood": -np.log(
                    np.clip(true_probability, 1e-12, 1.0)
                ),
                "brier_score": brier,
            }
            for task_id, indices in task_output_indices.items():
                chunk[f"task_{task_id:02d}_probability_mass"] = probabilities_np[
                    :, indices
                ].sum(axis=1)
                chunk[f"task_{task_id:02d}_max_logit"] = logits_np[:, indices].max(
                    axis=1
                )
            row_chunks.append(pd.DataFrame(chunk))
            logit_chunks.append(logits_np.astype(np.float32, copy=False))
    finally:
        from .training import _shutdown_dataloader

        _shutdown_dataloader(loader, iterator)

    return EvaluationResult(
        predictions=pd.concat(row_chunks, ignore_index=True),
        logits=np.concatenate(logit_chunks, axis=0),
        output_to_original_class=output_to_original,
    )


def canonical_capture_table(
    predictions: pd.DataFrame,
    task_sequence: tuple[int, ...],
    canonical_position: int,
    num_tasks: int,
) -> pd.DataFrame:
    errors = predictions.loc[~predictions["correct"].astype(bool)]
    denominator = len(errors)
    rows = []
    later_tasks = set(task_sequence[canonical_position + 1 :])
    earlier_tasks = set(task_sequence[:canonical_position])
    for task_id in range(num_tasks):
        count = int((errors["predicted_semantic_task"] == task_id).sum())
        rows.append(
            {
                "canonical_position": canonical_position,
                "predicted_semantic_task": task_id,
                "error_count": count,
                "total_misclassified": denominator,
                "proportion_of_misclassified": count / denominator
                if denominator
                else np.nan,
                "learned_before_canonical": task_id in earlier_tasks,
                "learned_after_canonical": task_id in later_tasks,
                "plot_primary_curve": task_id in later_tasks and task_id != 0,
            }
        )
    return pd.DataFrame(rows)


def _autocast(device, runtime_config: dict):
    import torch

    enabled = bool(runtime_config.get("amp", True)) and device.type == "cuda"
    dtype = (
        torch.bfloat16
        if runtime_config.get("amp_dtype", "bfloat16") == "bfloat16"
        else torch.float16
    )
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)

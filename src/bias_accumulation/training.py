"""ARC-style sequential linear probing with a strictly frozen backbone."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..data import ClassSubset
from ..models import ExpandingLinearClassifier


@dataclass
class TrainingResult:
    classifier: ExpandingLinearClassifier
    output_to_original_class: list[int]
    history: pd.DataFrame
    classifier_statistics: pd.DataFrame
    optimizer_state: dict[str, Any] | None
    wall_time_seconds: float


def train_sequence(
    backbone,
    train_dataset,
    partition,
    task_sequence: tuple[int, ...],
    training_config: dict,
    runtime_config: dict,
) -> TrainingResult:
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    device = _resolve_device(runtime_config.get("device", "cuda"))
    backbone.to(device).eval()
    classifier = ExpandingLinearClassifier(backbone.spec.feature_dim, device)
    output_to_original: list[int] = []
    history_rows: list[dict[str, Any]] = []
    statistics_rows: list[dict[str, Any]] = []
    last_optimizer_state = None
    start_time = time.monotonic()

    for stage_index, semantic_task in enumerate(task_sequence):
        task_classes = list(partition.tasks[semantic_task])
        known_classes = len(output_to_original)
        output_to_original.extend(task_classes)
        classifier.expand(len(output_to_original))

        task_dataset = ClassSubset(train_dataset, task_classes)
        worker_count = int(runtime_config.get("num_workers", 0))
        loader_generator = torch.Generator()
        loader_generator.manual_seed(partition.seed * 10_000 + stage_index)
        worker_options = {}
        if worker_count:
            worker_options["prefetch_factor"] = int(
                runtime_config.get("prefetch_factor", 1)
            )
        loader = DataLoader(
            task_dataset,
            batch_size=int(training_config["batch_size"]),
            shuffle=True,
            num_workers=worker_count,
            pin_memory=device.type == "cuda",
            persistent_workers=worker_count > 0,
            timeout=int(runtime_config.get("dataloader_timeout_seconds", 120))
            if worker_count
            else 0,
            worker_init_fn=_seed_worker,
            generator=loader_generator,
            **worker_options,
        )

        first = stage_index == 0
        epochs = int(training_config["epochs_first" if first else "epochs_later"])
        learning_rate = float(
            training_config["learning_rate_first" if first else "learning_rate_later"]
        )
        weight_decay = float(
            training_config["weight_decay_first" if first else "weight_decay_later"]
        )
        milestones = list(
            training_config["milestones_first" if first else "milestones_later"]
        )
        lr_decay = float(
            training_config["lr_decay_first" if first else "lr_decay_later"]
        )

        # ARC recreates SGD after every expansion and gives it the entire classifier
        # parameter tensor. Old rows therefore retain zero data-gradient but continue
        # to receive optimizer weight decay.
        optimizer = torch.optim.SGD(
            classifier.module.parameters(),
            lr=learning_rate,
            momentum=float(training_config.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_decay
        )
        original_to_output = {
            class_id: index for index, class_id in enumerate(output_to_original)
        }

        try:
            for epoch in range(epochs):
                classifier.module.train()
                backbone.eval()
                total_loss = 0.0
                total_examples = 0
                globally_correct = 0
                locally_correct = 0

                for _, images, original_targets in loader:
                    images = images.to(device, non_blocking=True)
                    original_targets = original_targets.tolist()
                    global_targets = torch.tensor(
                        [
                            original_to_output[int(target)]
                            for target in original_targets
                        ],
                        dtype=torch.long,
                        device=device,
                    )
                    local_targets = global_targets - known_classes

                    with _autocast(device, runtime_config):
                        features = backbone(images)
                        logits = classifier(features)
                        loss = F.cross_entropy(logits[:, known_classes:], local_targets)

                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

                    batch_size = len(original_targets)
                    total_loss += float(loss.detach()) * batch_size
                    total_examples += batch_size
                    globally_correct += int(
                        (logits.argmax(dim=1) == global_targets).sum().item()
                    )
                    locally_correct += int(
                        (logits[:, known_classes:].argmax(dim=1) == local_targets)
                        .sum()
                        .item()
                    )

                scheduler.step()
                history_rows.append(
                    {
                        "seed": partition.seed,
                        "stage_index": stage_index,
                        "semantic_task": semantic_task,
                        "epoch": epoch + 1,
                        "num_examples": total_examples,
                        "loss": total_loss / max(total_examples, 1),
                        "global_train_accuracy": globally_correct
                        / max(total_examples, 1),
                        "local_train_accuracy": locally_correct
                        / max(total_examples, 1),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "weight_decay": weight_decay,
                    }
                )
        finally:
            _shutdown_dataloader(loader)
        statistics_rows.extend(
            classifier_statistics(
                classifier.module,
                partition,
                output_to_original,
                stage_index,
                semantic_task,
            )
        )
        last_optimizer_state = optimizer.state_dict()

    return TrainingResult(
        classifier=classifier,
        output_to_original_class=output_to_original,
        history=pd.DataFrame(history_rows),
        classifier_statistics=pd.DataFrame(statistics_rows),
        optimizer_state=last_optimizer_state,
        wall_time_seconds=time.monotonic() - start_time,
    )


def classifier_statistics(
    classifier_module,
    partition,
    output_to_original: list[int],
    stage_index: int,
    trained_semantic_task: int,
) -> list[dict[str, Any]]:
    weight = classifier_module.weight.detach().float().cpu().numpy()
    bias = classifier_module.bias.detach().float().cpu().numpy()
    class_to_task = partition.task_for_class()
    rows: list[dict[str, Any]] = []
    for output_index, original_class in enumerate(output_to_original):
        rows.append(
            {
                "stage_index": stage_index,
                "trained_semantic_task": trained_semantic_task,
                "output_index": output_index,
                "original_class": original_class,
                "semantic_task": class_to_task[original_class],
                "weight_l2_norm": float(np.linalg.norm(weight[output_index])),
                "bias": float(bias[output_index]),
                "weight_mean": float(weight[output_index].mean()),
                "weight_std": float(weight[output_index].std()),
            }
        )
    return rows


def _resolve_device(requested: str):
    import torch

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable. Install an RTX-5090-compatible PyTorch "
            "build (CUDA 12.8 or newer) and confirm the WSL NVIDIA driver is visible."
        )
    if device.type == "cuda":
        major, minor = torch.cuda.get_device_capability(device)
        if major < 12:
            # Other GPUs remain valid for development; this is logged rather than rejected.
            pass
    return device


def _autocast(device, runtime_config: dict):
    import torch

    enabled = bool(runtime_config.get("amp", True)) and device.type == "cuda"
    dtype_name = runtime_config.get("amp_dtype", "bfloat16")
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def _seed_worker(worker_id: int) -> None:
    import random

    import torch

    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _shutdown_dataloader(loader, iterator=None) -> None:
    """Release persistent worker processes and their pipes between tasks."""

    if iterator is None:
        iterator = getattr(loader, "_iterator", None)
    if iterator is None:
        return
    shutdown = getattr(iterator, "_shutdown_workers", None)
    try:
        if shutdown is not None:
            shutdown()
    finally:
        loader._iterator = None

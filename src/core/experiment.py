"""Core continual-routing comparison experiment."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..artifacts import (
    mark_done,
    save_npz,
    save_torch,
    write_dataframe,
    write_json,
)
from ..config import resolved_config, task_counts_for_dataset
from ..data import (
    DATASET_NUM_CLASSES,
    ClassSubset,
    load_dataset_pair,
    load_feature_cache,
)
from ..models import ExpandingLinearClassifier, build_frozen_backbone, build_small_maf
from ..reproducibility import environment_metadata, seed_everything
from ..tasks import create_partition
from .aggregation import aggregate_core_results
from .artifacts import CoreRunPaths
from .evaluation import metric_row, routing_confusion
from .projection import fixed_random_projection
from .routing import (
    predict_within_task,
    route_by_flow,
    route_task_global,
    route_task_local,
)
from .training import (
    class_stratified_validation_split,
    classifier_logits,
    flow_log_probabilities,
    train_flow,
    train_linear_head,
)

LOGGER = logging.getLogger(__name__)


def run_core(
    config: dict[str, Any],
    repository_root: str | Path,
    datasets: Iterable[str] | None = None,
    backbones: Iterable[str] | None = None,
    task_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    max_flow_epochs: int | None = None,
    max_stages: int | None = None,
    projection_dim: int | None = None,
    aggregate: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run the four-method continual comparison, resuming completed partitions."""
    import torch

    root = Path(repository_root).resolve()
    resolved = resolved_config(config, root)
    output_root = Path(resolved["project"]["output_root"])
    experiment_root = output_root / "core"
    core_config = resolved["core"]
    flow_config = core_config["flow"]
    dataset_values = list(datasets or resolved["datasets"].keys())
    backbone_values = list(backbones or resolved["backbones"])
    task_override = (
        None if task_counts is None else [int(value) for value in task_counts]
    )
    task_counts_by_dataset = {
        name: task_counts_for_dataset(resolved, name, task_override)
        for name in dataset_values
    }
    seed_values = [int(value) for value in (seeds or resolved["seeds"])]
    projection_dim = int(
        core_config["projection_dims"][0] if projection_dim is None else projection_dim
    )
    if projection_dim < 2:
        raise ValueError("projection_dim must be at least two.")
    if projection_dim not in {int(value) for value in core_config["projection_dims"]}:
        raise ValueError(f"Unsupported projection dimension: {projection_dim}")
    projection_seed = int(core_config.get("projection_seed", 2029))
    if max_stages is not None and int(max_stages) < 1:
        raise ValueError("max_stages must be at least one.")
    device = torch.device(resolved["runtime"].get("device", "cuda"))
    write_json(experiment_root / "resolved_config.json", resolved)
    write_json(experiment_root / "environment.json", environment_metadata(root))

    manifest = {
        "projection": "fixed_gaussian_random",
        "projection_dim": projection_dim,
        "projection_seed": projection_seed,
        "projection_data_fit": False,
        "backbone_frozen": True,
        "classifier_routing": {
            "task_global_softmax_linear_probe": (
                "one softmax over all observed class outputs; compare each task's maximum"
            ),
            "task_local_softmax_linear_probe": (
                "one softmax per task's class outputs; compare each task's maximum"
            ),
        },
        "arc_head_training": resolved["training"],
        "flow_training": flow_config,
        "datasets": dataset_values,
        "backbones": backbone_values,
        "task_counts_by_dataset": task_counts_by_dataset,
        "seeds": seed_values,
    }
    write_json(
        experiment_root / "manifests" / f"gaussian_{projection_dim:04d}d.json", manifest
    )
    summary = {"planned": 0, "completed": 0, "skipped": 0, "failed": 0}
    dataset_cache: dict[str, tuple[object, object]] = {}

    for backbone_name in backbone_values:
        backbone = None
        if projection_dim == 64:
            backbone = build_frozen_backbone(backbone_name).to(device).eval()
            for parameter in backbone.module.parameters():
                parameter.requires_grad_(False)
        for dataset_name in dataset_values:
            if projection_dim == 64 and dataset_name not in dataset_cache:
                dataset_cache[dataset_name] = load_dataset_pair(
                    dataset_name, resolved["datasets"][dataset_name]
                )
            train_dataset = (
                dataset_cache[dataset_name][0] if projection_dim == 64 else None
            )
            cached = load_feature_cache(output_root, dataset_name, backbone_name)
            projection = fixed_random_projection(
                cached["train_features"].shape[1],
                projection_dim,
                projection_seed,
                dataset_name,
                backbone_name,
            )
            projected_train = (
                np.asarray(cached["train_features"], dtype=np.float32) @ projection
            )
            projected_test = (
                np.asarray(cached["test_features"], dtype=np.float32) @ projection
            )
            projection_directory = (
                experiment_root / "projections" / dataset_name / backbone_name
            )
            save_npz(
                projection_directory / f"gaussian_{projection_dim:03d}d.npz",
                matrix=projection,
            )

            for num_tasks in task_counts_by_dataset[dataset_name]:
                for seed in seed_values:
                    summary["planned"] += 1
                    baseline_paths = CoreRunPaths.create(
                        experiment_root,
                        dataset_name,
                        backbone_name,
                        num_tasks,
                        seed,
                        64,
                        max_stages=max_stages,
                    )
                    run_paths = CoreRunPaths.create(
                        experiment_root,
                        dataset_name,
                        backbone_name,
                        num_tasks,
                        seed,
                        projection_dim,
                        max_stages=max_stages,
                    )
                    baseline_directory = baseline_paths.directory
                    run_directory = run_paths.directory
                    stages_to_run = min(num_tasks, max_stages or num_tasks)
                    if run_paths.is_complete(stages_to_run) and not overwrite:
                        summary["skipped"] += 1
                        continue
                    (run_directory / "DONE.json").unlink(missing_ok=True)
                    reuse_heads_directory = None
                    if projection_dim != 64:
                        if not baseline_paths.is_complete(stages_to_run):
                            raise FileNotFoundError(
                                "A completed 64D core run is required before a "
                                f"{projection_dim}D flow run: {baseline_directory}"
                            )
                        reuse_heads_directory = baseline_directory
                    try:
                        _run_one_partition(
                            resolved=resolved,
                            flow_config=flow_config,
                            dataset_name=dataset_name,
                            backbone_name=backbone_name,
                            num_tasks=num_tasks,
                            seed=seed,
                            backbone=backbone,
                            train_dataset=train_dataset,
                            cached=cached,
                            projected_train=projected_train,
                            projected_test=projected_test,
                            projection=projection,
                            device=device,
                            run_directory=run_directory,
                            reuse_heads_directory=reuse_heads_directory,
                            max_flow_epochs=max_flow_epochs,
                            max_stages=max_stages,
                            resume=not overwrite,
                        )
                        (run_directory / "FAILED.json").unlink(missing_ok=True)
                        summary["completed"] += 1
                    except Exception as error:
                        summary["failed"] += 1
                        write_json(
                            run_directory / "FAILED.json",
                            {
                                "dataset": dataset_name,
                                "backbone": backbone_name,
                                "num_tasks": num_tasks,
                                "seed": seed,
                                "projection_dim": projection_dim,
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "traceback": traceback.format_exc(),
                            },
                        )
                        LOGGER.exception(
                            "Core run failed: %s/%s tasks=%d seed=%d",
                            dataset_name,
                            backbone_name,
                            num_tasks,
                            seed,
                        )
                        raise
        if backbone is not None:
            backbone.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if aggregate:
        artifacts = aggregate_core_results(experiment_root)
        summary["artifacts"] = {key: str(value) for key, value in artifacts.items()}
    else:
        summary["artifacts"] = {}
    return summary


def _run_one_partition(
    *,
    resolved: dict[str, Any],
    flow_config: dict[str, Any],
    dataset_name: str,
    backbone_name: str,
    num_tasks: int,
    seed: int,
    backbone,
    train_dataset,
    cached: dict[str, np.ndarray],
    projected_train: np.ndarray,
    projected_test: np.ndarray,
    projection: np.ndarray,
    device,
    run_directory: Path,
    reuse_heads_directory: Path | None,
    max_flow_epochs: int | None,
    max_stages: int | None,
    resume: bool,
) -> None:
    import torch

    if reuse_heads_directory is None and (backbone is None or train_dataset is None):
        raise ValueError(
            "Training 64D classifier heads requires the backbone and image dataset."
        )
    seed_everything(seed, bool(resolved["runtime"].get("deterministic", True)))
    partition = create_partition(
        dataset_name, DATASET_NUM_CLASSES[dataset_name], num_tasks, seed
    )
    class_to_task = np.empty(DATASET_NUM_CLASSES[dataset_name], dtype=np.int64)
    for task_id, classes in enumerate(partition.tasks):
        class_to_task[np.asarray(classes, dtype=np.int64)] = task_id
    train_tasks = class_to_task[cached["train_classes"]]
    test_tasks = class_to_task[cached["test_classes"]]
    fit_indices, validation_indices = class_stratified_validation_split(
        cached["train_classes"],
        float(flow_config.get("validation_fraction", 0.1)),
        int(flow_config.get("validation_seed", 2041)),
    )
    sequential = ExpandingLinearClassifier(cached["train_features"].shape[1], device)
    sequential_classes: list[int] = []
    flow_states: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    per_task_rows: list[dict[str, Any]] = []
    history_rows: list[dict[str, Any]] = []
    architecture = {
        "input_dim": projected_train.shape[1],
        "num_layers": int(flow_config.get("num_layers", 4)),
        "hidden_multiplier": int(flow_config.get("hidden_multiplier", 2)),
        "hidden_layers": int(flow_config.get("hidden_layers", 1)),
        "log_scale_clip": float(flow_config.get("log_scale_clip", 3.0)),
    }
    run_directory.mkdir(parents=True, exist_ok=True)
    write_json(
        run_directory / "run_config.json",
        {
            "dataset": dataset_name,
            "backbone": backbone_name,
            "num_tasks": num_tasks,
            "seed": seed,
            "partition": partition.as_dict(),
            "projection": "fixed_gaussian_random",
            "projection_scheme": "nested_gaussian_64d_prefix_v1",
            "projection_dim": int(projected_train.shape[1]),
            "projection_shape": list(projection.shape),
            "reused_heads_from": (
                str(reuse_heads_directory.resolve())
                if reuse_heads_directory is not None
                else None
            ),
            "flow_architecture": architecture,
            "arc_head_training": resolved["training"],
            "flow_training": flow_config,
        },
    )

    stages_to_run = (
        min(num_tasks, int(max_stages)) if max_stages is not None else num_tasks
    )
    if reuse_heads_directory is not None:
        missing_heads = [
            reuse_heads_directory / "heads" / f"stage_{stage:02d}.pt"
            for stage in range(1, stages_to_run + 1)
            if not (reuse_heads_directory / "heads" / f"stage_{stage:02d}.pt").is_file()
        ]
        if missing_heads:
            raise FileNotFoundError(
                f"Completed 64D run is missing reusable stage heads: {missing_heads[0]}"
            )
        LOGGER.info(
            "Reusing dimension-independent heads from %s", reuse_heads_directory
        )
    method_names = [
        "task_global_softmax_linear_probe",
        "task_local_softmax_linear_probe",
        "flow_routed_linear_probe",
        "offline_joint_linear_head",
    ]
    start_stage = 0
    partial_metric_path = run_directory / "stage_metrics.partial.csv"
    partial_per_task_path = run_directory / "per_task_metrics.partial.csv"
    partial_history_path = run_directory / "training_history.partial.csv"
    if resume and all(
        path.is_file()
        for path in (partial_metric_path, partial_per_task_path, partial_history_path)
    ):
        for stage_index in range(stages_to_run):
            if not all(
                path.is_file()
                for path in (
                    run_directory / "flows" / f"task_{stage_index:02d}.pt",
                    run_directory / "heads" / f"stage_{stage_index + 1:02d}.pt",
                    run_directory
                    / "stages"
                    / f"stage_{stage_index + 1:02d}_predictions.npz",
                )
            ):
                break
            start_stage = stage_index + 1
        if start_stage:
            metric_frame = pd.read_csv(partial_metric_path)
            recorded_stages = (
                int(metric_frame["stage_index"].max()) + 1
                if not metric_frame.empty
                else 0
            )
            start_stage = min(start_stage, recorded_stages)
            metric_rows = metric_frame[
                metric_frame["stage_index"] < start_stage
            ].to_dict("records")
            per_task_frame = pd.read_csv(partial_per_task_path)
            per_task_rows = per_task_frame[
                per_task_frame["stage_index"] < start_stage
            ].to_dict("records")
            history_frame = pd.read_csv(partial_history_path)
            history_rows = history_frame[
                history_frame["stage_index"] < start_stage
            ].to_dict("records")
            for task_id in range(start_stage):
                payload = torch.load(
                    run_directory / "flows" / f"task_{task_id:02d}.pt",
                    map_location="cpu",
                    weights_only=False,
                )
                flow_states.append(payload["state_dict"])
                sequential_classes.extend(partition.tasks[task_id])
            sequential.expand(len(sequential_classes))
            previous_head = torch.load(
                run_directory / "heads" / f"stage_{start_stage:02d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            sequential.module.load_state_dict(previous_head["sequential_state_dict"])
            LOGGER.info("Resuming %s at stage %d", run_directory, start_stage + 1)

    for stage_index in range(start_stage, stages_to_run):
        task_classes = list(partition.tasks[stage_index])
        old_count = len(sequential_classes)
        sequential_classes.extend(task_classes)
        sequential.expand(len(sequential_classes))
        reused_head = None
        if reuse_heads_directory is not None:
            reused_head = torch.load(
                reuse_heads_directory / "heads" / f"stage_{stage_index + 1:02d}.pt",
                map_location="cpu",
                weights_only=False,
            )
            if list(reused_head["output_classes"]) != sequential_classes:
                raise ValueError(
                    "Reusable head class order does not match the requested task partition."
                )
            sequential.module.load_state_dict(reused_head["sequential_state_dict"])
        else:
            sequential_history = train_linear_head(
                backbone=backbone,
                head=sequential.module,
                dataset=ClassSubset(train_dataset, task_classes),
                output_classes=sequential_classes,
                loss_start=old_count,
                training=resolved["training"],
                runtime=resolved["runtime"],
                stage_index=stage_index,
                joint=False,
                device=device,
                seed=seed,
            )
            history_rows.extend(sequential_history)
        # Flow and joint baselines use isolated RNG streams; restoring this state keeps the
        # sequential head identical to a run in which those comparison methods were absent.
        sequential_cpu_rng = torch.random.get_rng_state()
        sequential_cuda_rng = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )

        task_fit = fit_indices[train_tasks[fit_indices] == stage_index]
        task_validation = validation_indices[
            train_tasks[validation_indices] == stage_index
        ]
        flow_seed = (
            int(flow_config.get("random_seed", 2027))
            + seed * 1000
            + num_tasks * 100
            + stage_index
        )
        torch.manual_seed(flow_seed)
        torch.cuda.manual_seed_all(flow_seed)
        flow = build_small_maf(**architecture)
        flow_result = train_flow(
            flow,
            projected_train[task_fit],
            projected_train[task_validation],
            flow_config,
            device,
            seed=flow_seed,
            max_epochs=max_flow_epochs,
        )
        flow_state = {
            key: value.detach().cpu().clone()
            for key, value in flow.state_dict().items()
        }
        flow_states.append(flow_state)
        history_rows.extend(
            {
                "stage_index": stage_index,
                "method": "online_flow",
                **row,
            }
            for row in flow_result["history"]
        )
        save_torch(
            run_directory / "flows" / f"task_{stage_index:02d}.pt",
            {
                "state_dict": flow_state,
                "architecture": architecture,
                "task_id": stage_index,
                "classes": task_classes,
                "best_epoch": flow_result["best_epoch"],
                "best_validation_nll": flow_result["best_validation_nll"],
                "restored_train_nll": flow_result["restored_train_nll"],
                "restored_validation_nll": flow_result["restored_validation_nll"],
                "generalization_gap": flow_result["generalization_gap"],
            },
        )
        flow.to("cpu")

        seen_mask = test_tasks <= stage_index
        seen_features = np.asarray(cached["test_features"][seen_mask], dtype=np.float32)
        true_classes = cached["test_classes"][seen_mask].astype(np.int64)
        true_tasks = test_tasks[seen_mask].astype(np.int64)
        sequential_logits = classifier_logits(sequential.module, seen_features, device)
        sequential_task, global_softmax_scores = route_task_global(
            sequential_logits, sequential_classes, class_to_task
        )
        sequential_prediction = predict_within_task(
            sequential_logits, sequential_classes, sequential_task, class_to_task
        )
        local_softmax_task, local_softmax_scores = route_task_local(
            sequential_logits, sequential_classes, class_to_task
        )
        local_softmax_prediction = predict_within_task(
            sequential_logits, sequential_classes, local_softmax_task, class_to_task
        )
        oracle_prediction = predict_within_task(
            sequential_logits, sequential_classes, true_tasks, class_to_task
        )

        flow_scores = np.empty((len(true_classes), stage_index + 1), dtype=np.float32)
        projected_seen = projected_test[seen_mask]
        for task_id, state in enumerate(flow_states):
            candidate = build_small_maf(**architecture)
            candidate.load_state_dict(state)
            flow_scores[:, task_id] = flow_log_probabilities(
                candidate,
                projected_seen,
                int(flow_config.get("eval_batch_size", 2048)),
                device,
            )
            candidate.to("cpu")
        flow_task = route_by_flow(flow_scores)
        flow_prediction = predict_within_task(
            sequential_logits, sequential_classes, flow_task, class_to_task
        )

        joint_classes = list(sequential_classes)
        joint = ExpandingLinearClassifier(cached["train_features"].shape[1], device)
        joint.expand(len(joint_classes))
        if reused_head is not None:
            joint.module.load_state_dict(reused_head["joint_state_dict"])
        else:
            joint_seed = seed * 100_000 + stage_index + 50_000
            torch.manual_seed(joint_seed)
            torch.cuda.manual_seed_all(joint_seed)
            joint_history = train_linear_head(
                backbone=backbone,
                head=joint.module,
                dataset=ClassSubset(train_dataset, joint_classes),
                output_classes=joint_classes,
                loss_start=0,
                training=resolved["training"],
                runtime=resolved["runtime"],
                stage_index=stage_index,
                joint=True,
                device=device,
                seed=seed,
            )
            history_rows.extend(joint_history)
        joint_logits = classifier_logits(joint.module, seen_features, device)
        joint_prediction = np.asarray(joint_classes)[joint_logits.argmax(axis=1)]
        joint_task = class_to_task[joint_prediction]
        joint_oracle = predict_within_task(
            joint_logits, joint_classes, true_tasks, class_to_task
        )

        methods = {
            "task_global_softmax_linear_probe": (
                sequential_prediction,
                sequential_task,
                oracle_prediction,
            ),
            "task_local_softmax_linear_probe": (
                local_softmax_prediction,
                local_softmax_task,
                oracle_prediction,
            ),
            "flow_routed_linear_probe": (flow_prediction, flow_task, oracle_prediction),
            "offline_joint_linear_head": (joint_prediction, joint_task, joint_oracle),
        }
        for method, (predicted_class, predicted_task, oracle_class) in methods.items():
            metric_rows.append(
                metric_row(
                    dataset_name,
                    backbone_name,
                    num_tasks,
                    seed,
                    stage_index,
                    method,
                    true_classes,
                    true_tasks,
                    predicted_class,
                    predicted_task,
                    oracle_class,
                    projection_dim=projected_train.shape[1],
                )
            )
            for true_task in range(stage_index + 1):
                mask = true_tasks == true_task
                per_task_rows.append(
                    metric_row(
                        dataset_name,
                        backbone_name,
                        num_tasks,
                        seed,
                        stage_index,
                        method,
                        true_classes[mask],
                        true_tasks[mask],
                        predicted_class[mask],
                        predicted_task[mask],
                        oracle_class[mask],
                        scope_true_task=true_task,
                        projection_dim=projected_train.shape[1],
                    )
                )
        save_npz(
            run_directory / "stages" / f"stage_{stage_index + 1:02d}_predictions.npz",
            true_classes=true_classes,
            true_tasks=true_tasks,
            task_global_softmax_classes=sequential_prediction,
            task_global_softmax_tasks=sequential_task,
            sequential_classes=sequential_prediction,
            sequential_tasks=sequential_task,
            task_global_softmax_scores=global_softmax_scores,
            sequential_routing_confusion=routing_confusion(
                true_tasks, sequential_task, stage_index + 1
            ),
            task_local_softmax_classes=local_softmax_prediction,
            task_local_softmax_tasks=local_softmax_task,
            task_local_softmax_scores=local_softmax_scores,
            task_local_softmax_routing_confusion=routing_confusion(
                true_tasks, local_softmax_task, stage_index + 1
            ),
            flow_classes=flow_prediction,
            flow_tasks=flow_task,
            flow_routing_confusion=routing_confusion(
                true_tasks, flow_task, stage_index + 1
            ),
            flow_log_probs=flow_scores,
            joint_classes=joint_prediction,
            joint_tasks=joint_task,
            joint_routing_confusion=routing_confusion(
                true_tasks, joint_task, stage_index + 1
            ),
        )
        save_torch(
            run_directory / "heads" / f"stage_{stage_index + 1:02d}.pt",
            {
                "sequential_state_dict": sequential.state_dict(),
                "joint_state_dict": joint.state_dict(),
                "output_classes": joint_classes,
            },
        )
        LOGGER.info(
            "%s/%s tasks=%d seed=%d dim=%d stage=%d complete",
            dataset_name,
            backbone_name,
            num_tasks,
            seed,
            projected_train.shape[1],
            stage_index + 1,
        )
        write_dataframe(
            run_directory / "stage_metrics.partial.csv", pd.DataFrame(metric_rows)
        )
        write_dataframe(
            run_directory / "per_task_metrics.partial.csv", pd.DataFrame(per_task_rows)
        )
        write_dataframe(
            run_directory / "training_history.partial.csv", pd.DataFrame(history_rows)
        )
        torch.random.set_rng_state(sequential_cpu_rng)
        if sequential_cuda_rng is not None:
            torch.cuda.set_rng_state_all(sequential_cuda_rng)

    write_dataframe(run_directory / "stage_metrics.csv", pd.DataFrame(metric_rows))
    write_dataframe(run_directory / "per_task_metrics.csv", pd.DataFrame(per_task_rows))
    write_dataframe(run_directory / "training_history.csv", pd.DataFrame(history_rows))
    mark_done(
        run_directory,
        {
            "stages": stages_to_run,
            "methods": method_names,
            "projection_dim": int(projected_train.shape[1]),
        },
    )

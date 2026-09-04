"""Validated loading and aggregation of the numeric paper-figure results."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATASETS = {"cifar100": "CIFAR-100", "imagenet_r": "ImageNet-R"}
BACKBONES = {"vit_b16": "ViT-B/16", "resnet50": "ResNet-50"}
METHODS = {
    "task_global_softmax": ("Task-global softmax", "#0072B2", "o", "-"),
    "task_local_softmax": ("Task-local softmax", "#CC79A7", "d", ":"),
    "flow_64": ("Flow 64D", "#D55E00", "s", "-"),
    "flow_128": ("Flow 128D", "#E69F00", "D", "-"),
    "flow_256": ("Flow 256D", "#009E73", "v", "-"),
    "flow_1024": ("Flow 1024D", "#56B4E9", "^", "-"),
    "offline_joint": ("Offline joint", "#4D4D4D", "X", "--"),
}
IDENTITY = ["dataset", "backbone", "num_tasks", "seed", "tasks_seen"]
VALUES = ["task_routing_accuracy", "class_accuracy"]
DIMENSIONS = (64, 128, 256, 1024)
SEEDS = (1993, 1994, 1995)
TASK_COUNTS = (10, 20)


@dataclass(frozen=True)
class CapacityResults:
    raw: pd.DataFrame
    summary: pd.DataFrame
    coverage: pd.DataFrame
    sources: list[dict[str, str]]


def load_capacity_results(experiment_root: Path) -> CapacityResults:
    """Load the baseline and only dataset-complete flow-dimension sweeps."""

    root = Path(experiment_root)
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, str]] = []
    coverage_rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        dimension_complete = {
            dimension: all(
                _run_directory(root, dataset, backbone, tasks, seed, dimension)
                .joinpath("DONE.json")
                .is_file()
                for backbone in BACKBONES
                for tasks in TASK_COUNTS
                for seed in SEEDS
            )
            for dimension in DIMENSIONS
        }
        if not dimension_complete[64]:
            raise ValueError(f"Missing completed 64D baseline partitions for {dataset}.")
        for backbone in BACKBONES:
            for tasks in TASK_COUNTS:
                for seed in SEEDS:
                    for dimension in DIMENSIONS:
                        directory = _run_directory(
                            root, dataset, backbone, tasks, seed, dimension
                        )
                        complete = (directory / "DONE.json").is_file()
                        coverage_rows.append(
                            {
                                "dataset": dataset,
                                "backbone": backbone,
                                "num_tasks": tasks,
                                "seed": seed,
                                "projection_dim": dimension,
                                "complete": complete,
                                "included": dimension_complete[dimension],
                            }
                        )
                        if not dimension_complete[dimension]:
                            continue
                        path = directory / "stage_metrics.csv"
                        contents = path.read_bytes()
                        frame = pd.read_csv(path)
                        _validate_run(
                            frame, dataset, backbone, tasks, seed, dimension, path
                        )
                        frames.append(_normalize_methods(frame, dimension))
                        sources.append(
                            {
                                "path": str(path.resolve()),
                                "sha256": hashlib.sha256(contents).hexdigest(),
                            }
                        )
    coverage = pd.DataFrame(coverage_rows)
    raw = pd.concat(frames, ignore_index=True)
    groups = ["dataset", "backbone", "num_tasks", "tasks_seen", "display_method"]
    summary = raw.groupby(groups, sort=False)[VALUES].agg(["mean", "std", "count"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return CapacityResults(raw, summary.reset_index(), coverage, sources)


def _run_directory(
    root: Path,
    dataset: str,
    backbone: str,
    tasks: int,
    seed: int,
    dimension: int,
) -> Path:
    base = root / "runs" / dataset / backbone / f"tasks_{tasks:02d}" / f"seed_{seed}"
    return base / f"dimension_{dimension:04d}"


def _validate_run(
    frame: pd.DataFrame,
    dataset: str,
    backbone: str,
    num_tasks: int,
    seed: int,
    dimension: int,
    path: Path,
) -> None:
    required = set(IDENTITY + VALUES + ["method", "examples"])
    if not required.issubset(frame.columns):
        missing = sorted(required - set(frame.columns))
        raise ValueError(f"Missing columns in {path}: {missing}")
    for column, value in (
        ("dataset", dataset),
        ("backbone", backbone),
        ("num_tasks", num_tasks),
        ("seed", seed),
    ):
        if frame.empty or not frame[column].eq(value).all():
            raise ValueError(f"Run identity mismatch for {column} in {path}")
    expected_methods = {
        "task_global_softmax_linear_probe",
        "task_local_softmax_linear_probe",
        "flow_routed_linear_probe",
        "offline_joint_linear_head",
    }
    if dimension == 64 and set(frame.method) != expected_methods:
        raise ValueError(f"Unexpected methods in {path}: {sorted(set(frame.method))}")
    if dimension != 64 and "flow_routed_linear_probe" not in set(frame.method):
        raise ValueError(f"Missing flow results in dimension sweep {path}")
    if frame.duplicated(["tasks_seen", "method"]).any():
        raise ValueError(f"Duplicate stage/method rows in {path}")
    methods = expected_methods if dimension == 64 else {"flow_routed_linear_probe"}
    expected_stages = set(range(1, num_tasks + 1))
    for method in methods:
        actual_stages = set(frame.loc[frame.method == method, "tasks_seen"])
        if actual_stages != expected_stages:
            raise ValueError(f"Incomplete {method} curve in {path}")
    scores = frame[VALUES].to_numpy(dtype=float)
    if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
        raise ValueError(f"Invalid accuracies in {path}")


def _normalize_methods(frame: pd.DataFrame, dimension: int) -> pd.DataFrame:
    result = frame.copy()
    result["display_method"] = result.method.map(
        {
            "task_global_softmax_linear_probe": "task_global_softmax",
            "task_local_softmax_linear_probe": "task_local_softmax",
            "flow_routed_linear_probe": f"flow_{dimension}",
            "offline_joint_linear_head": "offline_joint",
        }
    )
    result["projection_dim"] = dimension
    if dimension != 64:
        result = result[result.display_method == f"flow_{dimension}"]
    return result

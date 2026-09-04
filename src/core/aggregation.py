"""Cross-run tabular aggregation for the core experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..artifacts import is_done, write_dataframe


def aggregate_core_results(experiment_root: str | Path) -> dict[str, Path]:
    """Aggregate every completed non-smoke core run without producing figures."""

    root = Path(experiment_root)
    stage_paths = [
        path
        for path in root.glob("runs/**/stage_metrics.csv")
        if "_smoke_" not in path.as_posix() and is_done(path.parent)
    ]
    if not stage_paths:
        return {}
    data = pd.concat([pd.read_csv(path) for path in stage_paths], ignore_index=True)
    complete_path = root / "stage_metrics_complete.csv"
    write_dataframe(complete_path, data)

    groups = [
        "dataset",
        "backbone",
        "num_tasks",
        "projection_dim",
        "tasks_seen",
        "method",
    ]
    values = [
        "class_accuracy",
        "task_routing_accuracy",
        "routing_wrong_rate",
        "routing_correct_class_wrong_rate",
        "oracle_task_class_accuracy",
    ]
    aggregate = data.groupby(groups)[values].agg(["mean", "std", "count"])
    aggregate.columns = [f"{column}_{stat}" for column, stat in aggregate.columns]
    aggregate = aggregate.reset_index()
    aggregate_path = root / "stage_metrics_mean_std.csv"
    write_dataframe(aggregate_path, aggregate)
    artifacts = {"complete": complete_path, "aggregate": aggregate_path}

    per_task_paths = [
        path.with_name("per_task_metrics.csv")
        for path in stage_paths
        if path.with_name("per_task_metrics.csv").is_file()
    ]
    if per_task_paths:
        per_task = pd.concat(
            [pd.read_csv(path) for path in per_task_paths], ignore_index=True
        )
        per_task_complete_path = root / "per_task_metrics_complete.csv"
        write_dataframe(per_task_complete_path, per_task)
        per_task_aggregate = per_task.groupby(groups + ["true_task"])[values].agg(
            ["mean", "std", "count"]
        )
        per_task_aggregate.columns = [
            f"{column}_{stat}" for column, stat in per_task_aggregate.columns
        ]
        per_task_aggregate = per_task_aggregate.reset_index()
        per_task_aggregate_path = root / "per_task_metrics_mean_std.csv"
        write_dataframe(per_task_aggregate_path, per_task_aggregate)
        artifacts.update(
            {
                "per_task_complete": per_task_complete_path,
                "per_task_aggregate": per_task_aggregate_path,
            }
        )
    return artifacts

"""Metrics for the core continual-routing experiment."""

from __future__ import annotations

from typing import Any

import numpy as np


def routing_confusion(
    true_tasks: np.ndarray,
    predicted_tasks: np.ndarray,
    size: int,
) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.int64)
    np.add.at(matrix, (true_tasks, predicted_tasks), 1)
    return matrix


def metric_row(
    dataset: str,
    backbone: str,
    num_tasks: int,
    seed: int,
    stage_index: int,
    method: str,
    true_class: np.ndarray,
    true_tasks: np.ndarray,
    predicted_class: np.ndarray,
    predicted_task: np.ndarray,
    oracle_class: np.ndarray,
    scope_true_task: int | None = None,
    projection_dim: int = 64,
) -> dict[str, Any]:
    routing_correct = predicted_task == true_tasks
    class_correct = predicted_class == true_class
    return {
        "dataset": dataset,
        "backbone": backbone,
        "num_tasks": num_tasks,
        "seed": seed,
        "stage_index": stage_index,
        "tasks_seen": stage_index + 1,
        "method": method,
        "projection_dim": int(projection_dim),
        "true_task": scope_true_task,
        "examples": len(true_class),
        "class_accuracy": float(class_correct.mean()),
        "task_routing_accuracy": float(routing_correct.mean()),
        "routing_wrong_rate": float((~routing_correct).mean()),
        "routing_correct_class_wrong_rate": float(
            (routing_correct & ~class_correct).mean()
        ),
        "oracle_task_class_accuracy": float((oracle_class == true_class).mean()),
    }

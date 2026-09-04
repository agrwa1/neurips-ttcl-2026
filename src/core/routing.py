"""Pure routing rules for the core method comparison."""

from __future__ import annotations

import numpy as np


def predict_within_task(
    logits: np.ndarray,
    output_classes: list[int],
    selected_tasks: np.ndarray,
    class_to_task: np.ndarray,
) -> np.ndarray:
    """Choose the highest-logit class within each selected task."""

    output_classes_array = np.asarray(output_classes, dtype=np.int64)
    output_tasks = class_to_task[output_classes_array]
    prediction = np.empty(len(logits), dtype=np.int64)
    for task_id in np.unique(selected_tasks):
        rows = np.flatnonzero(selected_tasks == task_id)
        columns = np.flatnonzero(output_tasks == task_id)
        local_prediction = np.argmax(logits[np.ix_(rows, columns)], axis=1)
        prediction[rows] = output_classes_array[columns[local_prediction]]
    return prediction


def route_task_local(
    logits: np.ndarray,
    output_classes: list[int],
    class_to_task: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Route by maxima from independent within-task softmax distributions."""

    logits = np.asarray(logits, dtype=np.float64)
    output_classes_array = np.asarray(output_classes, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != len(output_classes_array):
        raise ValueError("logits must have one column for every output class.")
    output_tasks = class_to_task[output_classes_array]
    task_ids = np.unique(output_tasks)
    scores = np.empty((len(logits), len(task_ids)), dtype=np.float64)
    for column, task_id in enumerate(task_ids):
        task_logits = logits[:, output_tasks == task_id]
        shifted = task_logits - task_logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
        scores[:, column] = probabilities.max(axis=1)
    return task_ids[scores.argmax(axis=1)], scores


def route_task_global(
    logits: np.ndarray,
    output_classes: list[int],
    class_to_task: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Route by per-task maxima from one softmax over all observed classes."""

    logits = np.asarray(logits, dtype=np.float64)
    output_classes_array = np.asarray(output_classes, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != len(output_classes_array):
        raise ValueError("logits must have one column for every output class.")
    output_tasks = class_to_task[output_classes_array]
    task_ids = np.unique(output_tasks)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)
    scores = np.empty((len(logits), len(task_ids)), dtype=np.float64)
    for column, task_id in enumerate(task_ids):
        scores[:, column] = probabilities[:, output_tasks == task_id].max(axis=1)
    return task_ids[scores.argmax(axis=1)], scores


def route_by_flow(flow_log_probabilities: np.ndarray) -> np.ndarray:
    """Route each example to the task with maximum flow log density."""

    scores = np.asarray(flow_log_probabilities)
    if scores.ndim != 2 or scores.shape[1] == 0:
        raise ValueError("Flow log probabilities must be a non-empty matrix.")
    return scores.argmax(axis=1)

"""Seeded class partitions and canonical-task insertion schedules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class TaskPartition:
    dataset: str
    seed: int
    num_classes: int
    num_tasks: int
    class_order: tuple[int, ...]
    tasks: tuple[tuple[int, ...], ...]

    def as_dict(self) -> dict:
        result = asdict(self)
        result["class_order"] = list(self.class_order)
        result["tasks"] = [list(task) for task in self.tasks]
        return result

    def task_for_class(self) -> dict[int, int]:
        return {
            class_id: task_id
            for task_id, classes in enumerate(self.tasks)
            for class_id in classes
        }


def create_partition(
    dataset: str, num_classes: int, num_tasks: int, seed: int
) -> TaskPartition:
    if num_classes % num_tasks != 0:
        raise ValueError(
            f"{num_classes} classes cannot be evenly divided into {num_tasks} tasks."
        )
    # Preserve the reference NumPy RandomState partition exactly.
    order = np.random.RandomState(seed).permutation(num_classes).tolist()
    width = num_classes // num_tasks
    tasks = tuple(
        tuple(order[start : start + width]) for start in range(0, num_classes, width)
    )
    return TaskPartition(
        dataset=dataset,
        seed=seed,
        num_classes=num_classes,
        num_tasks=num_tasks,
        class_order=tuple(order),
        tasks=tasks,
    )


def canonical_insertion_sequence(
    num_tasks: int, canonical_position: int
) -> tuple[int, ...]:
    """Move semantic task 0 through the otherwise fixed order 1..T-1."""

    if not 0 <= canonical_position < num_tasks:
        raise ValueError(f"canonical_position must be in [0, {num_tasks - 1}].")
    sequence = list(range(1, num_tasks))
    sequence.insert(canonical_position, 0)
    return tuple(sequence)


def all_canonical_insertion_sequences(num_tasks: int) -> list[tuple[int, ...]]:
    return [
        canonical_insertion_sequence(num_tasks, position)
        for position in range(num_tasks)
    ]


def class_output_order(
    partition: TaskPartition, task_sequence: Iterable[int]
) -> tuple[int, ...]:
    return tuple(
        class_id for task_id in task_sequence for class_id in partition.tasks[task_id]
    )


def validate_sequence(partition: TaskPartition, task_sequence: Iterable[int]) -> None:
    sequence = tuple(task_sequence)
    if sorted(sequence) != list(range(partition.num_tasks)):
        raise ValueError(
            "A task sequence must contain each semantic task exactly once."
        )

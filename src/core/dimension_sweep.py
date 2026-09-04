"""Orchestration for the configured flow-projection dimension comparison."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any


def run_dimension_sweep(
    config: dict[str, Any],
    repository_root: str | Path,
    *,
    datasets: Iterable[str] | None = None,
    backbones: Iterable[str] | None = None,
    task_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    dimensions: Iterable[int] | None = None,
    max_flow_epochs: int | None = None,
    max_stages: int | None = None,
    overwrite: bool = False,
    aggregate: bool = True,
) -> dict[str, Any]:
    """Run 64D first, then reuse its heads for every larger dimension."""

    from ..config import resolved_config
    from .aggregation import aggregate_core_results
    from .experiment import run_core

    configured = [int(value) for value in config["core"]["projection_dims"]]
    requested = (
        configured if dimensions is None else [int(value) for value in dimensions]
    )
    unsupported = sorted(set(requested) - set(configured))
    if unsupported:
        raise ValueError(f"Unsupported projection dimensions: {unsupported}")
    ordered = [value for value in configured if value in requested]
    if any(value != 64 for value in ordered) and 64 not in ordered:
        ordered.insert(0, 64)

    summaries: dict[str, Any] = {}
    for dimension in ordered:
        summaries[str(dimension)] = run_core(
            config,
            repository_root,
            datasets=datasets,
            backbones=backbones,
            task_counts=task_counts,
            seeds=seeds,
            projection_dim=dimension,
            max_flow_epochs=max_flow_epochs,
            max_stages=max_stages,
            overwrite=overwrite,
            aggregate=False,
        )
    artifacts = {}
    if aggregate:
        output_root = Path(
            resolved_config(config, repository_root)["project"]["output_root"]
        )
        artifacts = aggregate_core_results(output_root / "core")
    return {
        "dimensions": ordered,
        "runs": summaries,
        "artifacts": {key: str(value) for key, value in artifacts.items()},
    }

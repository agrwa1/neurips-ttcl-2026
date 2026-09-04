"""Canonical paths for core run artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..artifacts import is_done


@dataclass(frozen=True)
class CoreRunPaths:
    """Resolved paths for one partition and projection dimension."""

    directory: Path

    @classmethod
    def create(
        cls,
        experiment_root: str | Path,
        dataset: str,
        backbone: str,
        num_tasks: int,
        seed: int,
        projection_dim: int,
        max_stages: int | None = None,
    ) -> CoreRunPaths:
        name = f"dimension_{int(projection_dim):04d}"
        if max_stages is not None and int(max_stages) < int(num_tasks):
            name += f"_smoke_{int(max_stages):02d}stages"
        directory = (
            Path(experiment_root)
            / "runs"
            / dataset
            / backbone
            / f"tasks_{int(num_tasks):02d}"
            / f"seed_{int(seed)}"
            / name
        )
        return cls(directory)

    def head(self, stage: int) -> Path:
        return self.directory / "heads" / f"stage_{int(stage):02d}.pt"

    def flow(self, task: int) -> Path:
        return self.directory / "flows" / f"task_{int(task):02d}.pt"

    def predictions(self, stage: int) -> Path:
        return self.directory / "stages" / f"stage_{int(stage):02d}_predictions.npz"

    def is_complete(self, stages: int) -> bool:
        """Check the completion marker and every required durable artifact."""

        required = [
            self.directory / "run_config.json",
            self.directory / "stage_metrics.csv",
            self.directory / "per_task_metrics.csv",
            self.directory / "training_history.csv",
        ]
        for stage in range(1, int(stages) + 1):
            required.extend(
                [
                    self.head(stage),
                    self.flow(stage - 1),
                    self.predictions(stage),
                ]
            )
        return is_done(self.directory) and all(path.is_file() for path in required)

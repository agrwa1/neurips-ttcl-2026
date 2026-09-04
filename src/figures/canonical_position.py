"""Plot task-global routing destinations while canonical task 0 changes position."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATASET_LABELS = {"cifar100": "CIFAR-100", "imagenet_r": "ImageNet-R"}
BACKBONE_LABELS = {"vit_b16": "ViT-B/16", "resnet50": "ResNet-50"}


@dataclass(frozen=True)
class CanonicalPositionFigureSpec:
    runs_root: Path
    output_dir: Path
    dataset: str = "cifar100"
    backbone: str = "vit_b16"
    tasks: int = 10
    seeds: tuple[int, ...] = (1993, 1994, 1995)


def paper_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def completed_run_directory(
    runs_root: Path,
    dataset: str,
    backbone: str,
    tasks: int,
    seed: int,
    position: int,
) -> Path:
    base = runs_root / dataset / backbone / f"tasks_{tasks:02d}" / f"seed_{seed}"
    candidates = sorted(base.glob(f"canonical_position_{position:02d}_*"))
    completed = [path for path in candidates if (path / "DONE.json").is_file()]
    if len(completed) != 1:
        raise RuntimeError(
            f"Expected one completed run for seed={seed}, position={position}; "
            f"found {len(completed)} under {base}"
        )
    return completed[0]


def load_seed_matrix(
    args: CanonicalPositionFigureSpec, seed: int
) -> tuple[np.ndarray, int]:
    destinations = list(range(1, args.tasks))
    counts = np.zeros((len(destinations), args.tasks), dtype=np.int64)
    expected_classes: tuple[int, ...] | None = None
    expected_samples: tuple[str, ...] | None = None
    canonical_examples: int | None = None

    for position in range(args.tasks):
        directory = completed_run_directory(
            args.runs_root,
            args.dataset,
            args.backbone,
            args.tasks,
            seed,
            position,
        )
        required = (
            directory / "run_manifest.json",
            directory / "task_sequence.csv",
            directory / "canonical_predictions.csv.gz",
            directory / "capture_by_task.csv",
        )
        missing = [path for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete completed run: missing {missing[0]}")

        manifest = json.loads(required[0].read_text(encoding="utf-8"))
        canonical_classes = tuple(int(value) for value in manifest["partition"]["tasks"][0])
        if expected_classes is None:
            expected_classes = canonical_classes
        elif canonical_classes != expected_classes:
            raise ValueError(f"Canonical task classes change across positions for seed {seed}")

        sequence = (
            pd.read_csv(required[1])
            .sort_values("sequence_position")["semantic_task"]
            .astype(int)
            .tolist()
        )
        if sequence[position] != 0 or [task for task in sequence if task != 0] != destinations:
            raise ValueError(f"Invalid canonical insertion sequence in {required[1]}")

        predictions = pd.read_csv(
            required[2],
            usecols=["sample_id", "true_semantic_task"],
        )
        if predictions.empty or not predictions.true_semantic_task.eq(0).all():
            raise ValueError(f"Expected only canonical-task predictions in {required[2]}")
        sample_ids = tuple(predictions.sample_id.astype(str))
        if expected_samples is None:
            expected_samples = sample_ids
            canonical_examples = len(sample_ids)
        elif sample_ids != expected_samples:
            raise ValueError(
                f"Canonical evaluation examples change across positions for seed {seed}"
            )

        capture = pd.read_csv(required[3]).set_index("predicted_semantic_task")
        if set(capture.index.astype(int)) != set(range(args.tasks)):
            raise ValueError(
                f"Capture table does not preserve every semantic task in {required[3]}"
            )
        for row, destination in enumerate(destinations):
            counts[row, position] = int(capture.loc[destination, "error_count"])

    if canonical_examples is None:
        raise RuntimeError(f"No canonical examples loaded for seed {seed}")
    return counts, canonical_examples


def generate_canonical_position_figure(
    runs_root: Path,
    output_dir: Path,
    *,
    dataset: str = "cifar100",
    backbone: str = "vit_b16",
    tasks: int = 10,
    seeds: tuple[int, ...] = (1993, 1994, 1995),
) -> Path:
    """Render the canonical-position intervention heatmap and return its path."""

    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if backbone not in BACKBONE_LABELS:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if not seeds:
        raise ValueError("At least one seed is required.")
    args = CanonicalPositionFigureSpec(
        Path(runs_root), Path(output_dir), dataset, backbone, tasks, tuple(seeds)
    )
    if args.tasks < 2:
        raise ValueError("At least two tasks are required for cross-task routing.")
    matrices_and_sizes = [load_seed_matrix(args, seed) for seed in args.seeds]
    percentages = [100.0 * counts / size for counts, size in matrices_and_sizes]
    maximum = max(float(values.max()) for values in percentages)
    color_max = max(maximum, 1.0)
    panel_labels = "abcdefghijklmnopqrstuvwxyz"

    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(
            1,
            len(args.seeds),
            figsize=(4.8 * len(args.seeds), 4.8),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes)
        image = None
        for index, (axis, seed, values, (counts, _)) in enumerate(
            zip(axes, args.seeds, percentages, matrices_and_sizes)
        ):
            image = axis.imshow(
                values,
                cmap="cividis",
                vmin=0.0,
                vmax=color_max,
                aspect="auto",
                interpolation="nearest",
            )
            for row in range(values.shape[0]):
                for column in range(values.shape[1]):
                    red, green, blue, _ = image.cmap(image.norm(values[row, column]))
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    color = "white" if luminance < 0.48 else "#171717"
                    axis.text(
                        column,
                        row,
                        str(counts[row, column]),
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=7.2,
                    )
            axis.set_title(
                f"({panel_labels[index]}) Seed {seed}",
                loc="left",
                fontweight="bold",
            )
            axis.set_xlabel("Acquisition position of canonical task 0")
            axis.set_xticks(np.arange(args.tasks), np.arange(1, args.tasks + 1))
            axis.tick_params(length=0)
            axis.set_xticks(np.arange(-0.5, args.tasks, 1), minor=True)
            axis.set_yticks(np.arange(-0.5, args.tasks - 1, 1), minor=True)
            axis.grid(which="minor", color="white", linewidth=0.65, alpha=0.7)
            axis.tick_params(which="minor", bottom=False, left=False)

        destinations = np.arange(1, args.tasks)
        axes[0].set_ylabel("Destination semantic task")
        axes[0].set_yticks(np.arange(len(destinations)), [f"Task {task}" for task in destinations])
        if image is None:
            raise RuntimeError("No heatmap was rendered.")
        colorbar = figure.colorbar(image, ax=axes, fraction=0.022, pad=0.018)
        colorbar.set_label("Canonical-task test examples routed to destination (%)")
        figure.suptitle(
            "Task-global routing destinations under canonical-task position intervention\n"
            f"{DATASET_LABELS[args.dataset]} · {BACKBONE_LABELS[args.backbone]} · "
            f"{args.tasks} tasks",
            fontsize=13,
            fontweight="bold",
        )
        figure.text(
            0.5,
            -0.025,
            "Color shows percentage of all canonical-task test examples; annotations are raw "
            "counts. Task 0 is excluded because it represents within-task class errors. "
            "Task identities are seed-specific.",
            ha="center",
            fontsize=8.2,
            color="#444444",
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        destination = args.output_dir / (
            f"canonical_position_destination_heatmap_{args.dataset}_{args.backbone}_"
            f"{args.tasks}tasks.png"
        )
        figure.savefig(destination, dpi=320, bbox_inches="tight")
        plt.close(figure)
    return destination.resolve()

"""Create paper-ready task-global routing-error figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

DATASET_LABELS = {"cifar100": "CIFAR-100", "imagenet_r": "ImageNet-R"}
BACKBONE_LABELS = {"vit_b16": "ViT-B/16", "resnet50": "ResNet-50"}


@dataclass(frozen=True)
class BiasFigureSpec:
    runs_root: Path
    output_dir: Path
    dataset: str = "cifar100"
    backbone: str = "vit_b16"
    tasks: int = 10
    canonical_position: int = 0
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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def locate_capture_file(
    root: Path,
    dataset: str,
    backbone: str,
    tasks: int,
    seed: int,
    canonical_position: int,
) -> Path:
    base = root / dataset / backbone / f"tasks_{tasks:02d}" / f"seed_{seed}"
    if not base.is_dir() and backbone in {"vit_b16", "resnet50"}:
        # The original experiment tree used an ``_arc`` backbone suffix.
        legacy_base = (
            root / dataset / f"{backbone}_arc" / f"tasks_{tasks:02d}" / f"seed_{seed}"
        )
        if legacy_base.is_dir():
            base = legacy_base
    pattern = f"canonical_position_{canonical_position:02d}_*/capture_by_task.csv"
    candidates = sorted(base.glob(pattern))
    completed = [path for path in candidates if (path.parent / "DONE.json").is_file()]
    candidates = completed or candidates
    if not candidates:
        raise FileNotFoundError(f"No capture table matches {base / pattern}")
    if len(candidates) > 1:
        raise RuntimeError(f"Multiple capture tables match seed {seed}: {candidates}")
    return candidates[0]


def load_seed_counts(args: BiasFigureSpec, seed: int) -> pd.DataFrame:
    path = locate_capture_file(
        args.runs_root,
        args.dataset,
        args.backbone,
        args.tasks,
        seed,
        args.canonical_position,
    )
    frame = pd.read_csv(path)
    required = {"predicted_semantic_task", "error_count"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame.columns))}")
    # The canonical block is a within-task classification error, not cross-task capture.
    frame = frame[frame.predicted_semantic_task != 0].copy()
    frame["seed"] = seed
    frame["source_path"] = str(path.resolve())
    return frame.sort_values("predicted_semantic_task")


def export(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")


def generate_bias_accumulation_figure(
    runs_root: Path,
    output_dir: Path,
    *,
    dataset: str = "cifar100",
    backbone: str = "vit_b16",
    tasks: int = 10,
    canonical_position: int = 0,
    seeds: tuple[int, ...] = (1993, 1994, 1995),
) -> list[Path]:
    """Render one routing-error figure per seed and return their paths."""

    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    if backbone not in BACKBONE_LABELS:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if not seeds:
        raise ValueError("At least one seed is required.")
    args = BiasFigureSpec(
        Path(runs_root),
        Path(output_dir),
        dataset,
        backbone,
        tasks,
        canonical_position,
        tuple(seeds),
    )
    frames = [load_seed_counts(args, seed) for seed in args.seeds]
    shared_max = max(float(frame.error_count.max()) for frame in frames)
    x_limit = max(1.0, shared_max * 1.17)
    pair_output = args.output_dir / "routing_errors" / f"{args.dataset}_{args.backbone}"
    destinations: list[Path] = []

    with plt.rc_context(paper_style()):
        for frame, seed in zip(frames, args.seeds):
            height = 2.55 if args.tasks == 10 else 4.5
            figure, axis = plt.subplots(figsize=(3.25, height), constrained_layout=True)
            task_ids = frame.predicted_semantic_task.astype(int).to_numpy()
            counts = frame.error_count.astype(float).to_numpy()
            bars = axis.barh(
                task_ids,
                counts,
                height=0.68,
                color="#4C72B0",
                edgecolor="#334E75",
                linewidth=0.45,
            )
            axis.bar_label(
                bars,
                labels=[f"{value:.0f}" for value in counts],
                padding=3,
                fontsize=8,
            )
            axis.set_xlabel("Misclassified first-task examples")
            axis.set_xlim(0, x_limit)
            axis.set_yticks([])
            axis.grid(axis="x", color="#D8D8D8", linewidth=0.6)
            axis.set_axisbelow(True)
            axis.annotate(
                "",
                xy=(-0.09, 0.86),
                xytext=(-0.09, 0.14),
                xycoords="axes fraction",
                arrowprops={"arrowstyle": "->", "color": "black", "linewidth": 1.0},
                annotation_clip=False,
            )
            axis.text(
                -0.02,
                0.94,
                "Latest",
                transform=axis.transAxes,
                ha="right",
                va="center",
            )
            axis.text(
                -0.02,
                0.06,
                "Start",
                transform=axis.transAxes,
                ha="right",
                va="center",
            )
            destination = pair_output / f"routing_errors_{args.tasks}tasks_seed_{seed}"
            export(figure, destination)
            plt.close(figure)
            destinations.append(destination.with_suffix(".pdf").resolve())

    return destinations

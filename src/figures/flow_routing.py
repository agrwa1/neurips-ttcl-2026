"""Create paper-ready flow-routing diagnostics and dimension-only comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm

from ..data import DATASET_NUM_CLASSES
from ..tasks import create_partition
from .results import (
    BACKBONES,
    DATASETS,
    METHODS,
    CapacityResults,
    load_capacity_results,
)

SETTINGS = (
    ("cifar100", 10),
    ("cifar100", 20),
    ("imagenet_r", 10),
    ("imagenet_r", 20),
)
FLOW_METHODS = ("flow_64", "flow_128", "flow_256", "flow_1024")


@dataclass(frozen=True)
class FlowFigureSpec:
    experiment_root: Path
    tsne_root: Path
    output_dir: Path
    tsne_dataset: str = "cifar100"
    tsne_tasks: int = 20
    tsne_seed: int = 1993
    decomposition_tasks: int = 20
    decomposition_dim: int = 1024


def paper_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def export(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".png"), dpi=320, bbox_inches="tight")


def export_pdf(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")


def render_tsne(args: FlowFigureSpec, destination: Path) -> None:
    dataset = args.tsne_dataset
    task_count = args.tsne_tasks
    seed = args.tsne_seed
    partition = create_partition(dataset, DATASET_NUM_CLASSES[dataset], task_count, seed)
    class_to_task = partition.task_for_class()
    backbones = ("vit_b16", "resnet50")
    panel_labels = "abcd"
    task_map = plt.get_cmap("tab20", task_count)
    task_norm = BoundaryNorm(np.arange(-0.5, task_count + 0.5), task_count)

    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(2, 2, figsize=(11.3, 8.0), constrained_layout=True)
        task_artist = None
        for row, backbone in enumerate(backbones):
            source = args.tsne_root / f"{dataset}_{backbone}_tsne.npz"
            if not source.is_file():
                raise FileNotFoundError(f"Missing cached t-SNE coordinates: {source}")
            with np.load(source) as archive:
                embedding = archive["embedding"]
                classes = archive["original_classes"].astype(np.int64)
            tasks = np.asarray([class_to_task[int(value)] for value in classes], dtype=np.int64)
            task_axis, class_axis = axes[row]
            task_artist = task_axis.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=tasks,
                cmap=task_map,
                norm=task_norm,
                s=5,
                alpha=0.56,
                linewidths=0,
                rasterized=True,
            )
            class_axis.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=classes,
                cmap="turbo",
                s=5,
                alpha=0.56,
                linewidths=0,
                rasterized=True,
            )
            for column, axis in enumerate((task_axis, class_axis)):
                index = row * 2 + column
                coloring = "semantic task" if column == 0 else "exact class"
                axis.set_title(
                    f"({panel_labels[index]}) {BACKBONES[backbone]} · colored by {coloring}",
                    loc="left",
                    fontweight="bold",
                )
                axis.set_xlabel("t-SNE 1")
                axis.set_ylabel("t-SNE 2")
                axis.grid(color="#E0E0E0", linewidth=0.45)
                axis.set_axisbelow(True)
        colorbar = figure.colorbar(
            task_artist,
            ax=axes[:, 0],
            ticks=np.arange(task_count),
            fraction=0.035,
            pad=0.02,
        )
        colorbar.set_label("Task ID")
        figure.suptitle(
            f"Frozen-backbone representation geometry · {DATASETS[dataset]} · "
            f"{task_count} tasks · seed {seed}",
            fontsize=14,
            fontweight="bold",
        )
        figure.text(
            0.5,
            0.006,
            "Task and class labels are used only for coloring; PCA and t-SNE are label-free.",
            ha="center",
            fontsize=8.5,
            color="#555555",
        )
        export(figure, destination)
        plt.close(figure)


def load_flow_decomposition(
    experiment_root: Path,
    dataset: str,
    backbone: str,
    tasks: int,
    dimension: int,
) -> pd.DataFrame:
    frames = []
    for seed in (1993, 1994, 1995):
        base = experiment_root / "runs" / dataset / backbone / f"tasks_{tasks:02d}" / f"seed_{seed}"
        directory = base / f"dimension_{dimension:04d}"
        path = directory / "stage_metrics.csv"
        if not (directory / "DONE.json").is_file() or not path.is_file():
            raise FileNotFoundError(f"Incomplete flow run: {directory}")
        frame = pd.read_csv(path)
        frame = frame[frame.method == "flow_routed_linear_probe"].copy()
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    metrics = [
        "class_accuracy",
        "task_routing_accuracy",
        "routing_correct_class_wrong_rate",
        "routing_wrong_rate",
        "oracle_task_class_accuracy",
    ]
    return combined.groupby("tasks_seen", as_index=False)[metrics].agg("mean")


def render_flow_decomposition(args: FlowFigureSpec, destination: Path) -> None:
    settings = (
        ("cifar100", "vit_b16"),
        ("cifar100", "resnet50"),
        ("imagenet_r", "vit_b16"),
        ("imagenet_r", "resnet50"),
    )
    panel_labels = "abcd"
    with plt.rc_context(paper_style()):
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.6), sharey=True)
        for index, (axis, (dataset, backbone)) in enumerate(zip(axes.flat, settings)):
            curve = load_flow_decomposition(
                args.experiment_root,
                dataset,
                backbone,
                args.decomposition_tasks,
                args.decomposition_dim,
            )
            x = curve.tasks_seen.to_numpy(dtype=int)
            correct = 100 * curve.class_accuracy.to_numpy(dtype=float)
            within = 100 * curve.routing_correct_class_wrong_rate.to_numpy(dtype=float)
            wrong_task = 100 * curve.routing_wrong_rate.to_numpy(dtype=float)
            oracle = 100 * curve.oracle_task_class_accuracy.to_numpy(dtype=float)
            axis.stackplot(
                x,
                correct,
                within,
                wrong_task,
                labels=["Correct class", "Correct task, wrong class", "Wrong task"],
                colors=["#4E9F6D", "#E6B94A", "#D95F5F"],
                alpha=0.92,
                linewidth=0,
            )
            axis.plot(
                x,
                oracle,
                color="#222222",
                linestyle="--",
                linewidth=1.5,
                label="Task-oracle class accuracy",
            )
            axis.set_title(
                f"({panel_labels[index]}) {DATASETS[dataset]} · {BACKBONES[backbone]}",
                loc="left",
                fontweight="bold",
            )
            axis.set_xlabel("Tasks observed")
            axis.set_ylabel("Share of evaluated examples (%)")
            axis.set_xlim(0.7, args.decomposition_tasks + 0.3)
            axis.set_ylim(0, 101)
            axis.set_xticks(np.arange(1, args.decomposition_tasks + 1, 2))
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.6)
            axis.set_axisbelow(True)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        figure.suptitle(
            f"Flow-routed prediction outcomes · {args.decomposition_tasks} tasks · "
            f"{args.decomposition_dim}D projection",
            fontsize=14,
            fontweight="bold",
        )
        figure.tight_layout(rect=(0, 0.08, 1, 0.95))
        export(figure, destination)
        plt.close(figure)


def render_dimension_curves(
    results: CapacityResults, backbone: str, destination: Path
) -> list[Path]:
    y_min = 20
    destinations: list[Path] = []
    with plt.rc_context(paper_style()):
        for dataset, tasks in SETTINGS:
            figure, axis = plt.subplots(figsize=(2.75, 1.95))
            subset = results.summary[
                (results.summary.dataset == dataset)
                & (results.summary.backbone == backbone)
                & (results.summary.num_tasks == tasks)
                & (results.summary.display_method.isin(FLOW_METHODS))
            ]
            for method in FLOW_METHODS:
                curve = subset[subset.display_method == method].sort_values("tasks_seen")
                if curve.empty:
                    continue
                label, color, marker, linestyle = METHODS[method]
                x = curve.tasks_seen.to_numpy(dtype=int)
                mean = 100 * curve.class_accuracy_mean.to_numpy(dtype=float)
                sd = 100 * curve.class_accuracy_std.fillna(0).to_numpy(dtype=float)
                axis.plot(
                    x,
                    mean,
                    label=label,
                    color=color,
                    marker=marker,
                    markevery=max(1, tasks // 10),
                    linestyle=linestyle,
                    linewidth=0.95,
                    markersize=2.45,
                    markeredgewidth=0.4,
                )
                axis.fill_between(
                    x,
                    np.maximum(0, mean - sd),
                    np.minimum(100, mean + sd),
                    color=color,
                    alpha=0.08,
                    linewidth=0,
                )
            axis.set_xlabel("Tasks observed")
            axis.set_ylabel("Exact-class accuracy (%)")
            axis.set_xlim(0.7, tasks + 0.3)
            axis.set_ylim(y_min, 101)
            axis.set_xticks(np.arange(1, tasks + 1, 1 if tasks == 10 else 2))
            axis.tick_params(labelsize=6.7, width=0.65, length=2.8)
            axis.xaxis.label.set_size(7.4)
            axis.yaxis.label.set_size(7.4)
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.35)
            axis.set_axisbelow(True)
            handles, labels = axis.get_legend_handles_labels()
            figure.legend(
                handles,
                labels,
                loc="lower center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 0.01),
                fontsize=6.15,
                handlelength=1.8,
                handletextpad=0.4,
                columnspacing=0.85,
                labelspacing=0.22,
                borderaxespad=0.1,
            )
            figure.tight_layout(rect=(0, 0.155, 1, 1), pad=0.3)
            panel_destination = destination / f"{dataset}_{tasks}tasks"
            export_pdf(figure, panel_destination)
            plt.close(figure)
            destinations.append(panel_destination.with_suffix(".pdf"))
    return destinations


def render_dimension_table(
    results: CapacityResults, backbone: str, destination: Path
) -> None:
    rows: list[list[str]] = []
    bold_cells: set[tuple[int, int]] = set()
    for row_index, (dataset, tasks) in enumerate(SETTINGS, start=1):
        subset = results.summary[
            (results.summary.dataset == dataset)
            & (results.summary.backbone == backbone)
            & (results.summary.num_tasks == tasks)
            & (results.summary.tasks_seen == tasks)
            & (results.summary.display_method.isin(FLOW_METHODS))
        ].set_index("display_method")
        row = [DATASETS[dataset], str(tasks)]
        means = []
        for method in FLOW_METHODS:
            if method not in subset.index:
                row.append("—")
                means.append(float("-inf"))
                continue
            item = subset.loc[method]
            row.append(
                f"{100 * item.class_accuracy_mean:.1f} ± "
                f"{100 * item.class_accuracy_std:.1f}"
            )
            means.append(float(item.class_accuracy_mean))
        bold_cells.add((row_index, 2 + int(np.argmax(means))))
        rows.append(row)

    columns = ["Dataset", "Tasks", "Flow 64D", "Flow 128D", "Flow 256D", "Flow 1024D"]
    with plt.rc_context(paper_style()):
        figure, axis = plt.subplots(figsize=(11.4, 3.1))
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=[0.18, 0.09, 0.18, 0.18, 0.18, 0.18],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1.0, 1.65)
        for (row, column), cell in table.get_celld().items():
            cell.set_edgecolor("#B8B8B8")
            cell.set_linewidth(0.55)
            if row == 0:
                cell.set_facecolor("#E8EBEF")
                cell.get_text().set_fontweight("bold")
            elif row in (1, 2):
                cell.set_facecolor("#FAFAFA")
            else:
                cell.set_facecolor("#F3F3F3")
            if (row, column) in bold_cells:
                cell.get_text().set_fontweight("bold")
        axis.set_title(
            f"{BACKBONES[backbone]}: final exact-class accuracy by flow capacity",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        figure.text(
            0.02,
            0.025,
            "Mean ± sample SD across three seeds. Bold indicates the best flow dimension.",
            fontsize=8.5,
        )
        export(figure, destination)
        plt.close(figure)


def generate_flow_routing_figures(
    experiment_root: Path,
    tsne_root: Path,
    output_dir: Path,
    *,
    tsne_dataset: str = "cifar100",
    tsne_tasks: int = 20,
    tsne_seed: int = 1993,
    decomposition_tasks: int = 20,
    decomposition_dim: int = 1024,
) -> list[Path]:
    """Render the six flow-routing figures and return their paths."""

    if tsne_dataset not in DATASETS:
        raise ValueError(f"Unsupported t-SNE dataset: {tsne_dataset}")
    if decomposition_dim not in (64, 128, 256, 1024):
        raise ValueError(f"Unsupported decomposition dimension: {decomposition_dim}")
    args = FlowFigureSpec(
        Path(experiment_root).resolve(),
        Path(tsne_root).resolve(),
        Path(output_dir).resolve(),
        tsne_dataset,
        tsne_tasks,
        tsne_seed,
        decomposition_tasks,
        decomposition_dim,
    )
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results = load_capacity_results(args.experiment_root)

    tsne_destination = output / "frozen_latent_tsne"
    decomposition_destination = output / "flow_prediction_error_decomposition"
    destinations = [
        tsne_destination.with_suffix(".png"),
        decomposition_destination.with_suffix(".png"),
    ]
    render_tsne(args, tsne_destination)
    render_flow_decomposition(args, decomposition_destination)
    for backbone, short in (("vit_b16", "vit"), ("resnet50", "resnet")):
        curves = output / "dimensionality" / short
        table = output / f"flow_dimension_final_exact_class_{short}"
        destinations.extend(render_dimension_curves(results, backbone, curves))
        render_dimension_table(
            results, backbone, table
        )
        destinations.append(table.with_suffix(".png"))
    return destinations

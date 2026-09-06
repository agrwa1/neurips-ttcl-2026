"""Create paper-ready exact-class comparisons using one selected flow dimension."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .results import (
    DATASETS,
    CapacityResults,
    load_capacity_results,
)

SELECTED_FLOW = {"cifar100": 128, "imagenet_r": 1024}
SETTINGS = (
    ("cifar100", 10),
    ("cifar100", 20),
    ("imagenet_r", 10),
    ("imagenet_r", 20),
)
METHOD_STYLE = {
    "task_global_softmax": ("Logit-based routing", "#0072B2", "o", "-"),
    "task_local_softmax": ("Confidence-based routing", "#CC79A7", "D", ":"),
    "selected_flow": ("Flow routing", "#D55E00", "s", "-"),
    "offline_joint": ("Offline joint", "#4D4D4D", "X", "--"),
}


def paper_style() -> dict[str, object]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


def export(figure: plt.Figure, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination.with_suffix(".pdf"), bbox_inches="tight")


def selected_subset(results: CapacityResults, dataset: str, backbone: str, tasks: int):
    selected_dimension = SELECTED_FLOW[dataset]
    allowed = [
        "task_global_softmax",
        "task_local_softmax",
        f"flow_{selected_dimension}",
        "offline_joint",
    ]
    subset = results.summary[
        (results.summary.dataset == dataset)
        & (results.summary.backbone == backbone)
        & (results.summary.num_tasks == tasks)
        & (results.summary.display_method.isin(allowed))
    ].copy()
    subset["paper_method"] = subset.display_method.replace(
        {f"flow_{selected_dimension}": "selected_flow"}
    )
    missing = set(METHOD_STYLE) - set(subset.paper_method)
    if missing:
        raise ValueError(
            f"Missing methods for {dataset}/{backbone}/{tasks}: {sorted(missing)}"
        )
    return subset


def render_curves(
    results: CapacityResults, backbone: str, destination: Path
) -> list[Path]:
    y_min = 40 if backbone == "vit_b16" else 20
    destinations: list[Path] = []
    with plt.rc_context(paper_style()):
        for dataset, tasks in SETTINGS:
            figure, axis = plt.subplots(figsize=(3.25, 2.35))
            subset = selected_subset(results, dataset, backbone, tasks)
            for method, (label, color, marker, linestyle) in METHOD_STYLE.items():
                curve = subset[subset.paper_method == method].sort_values("tasks_seen")
                x = curve.tasks_seen.to_numpy(dtype=int)
                mean = 100 * curve.class_accuracy_mean.to_numpy(dtype=float)
                sd = 100 * curve.class_accuracy_std.fillna(0).to_numpy(dtype=float)
                axis.plot(
                    x,
                    mean,
                    color=color,
                    marker=marker,
                    markevery=max(1, tasks // 10),
                    linewidth=1.25,
                    markersize=3.2,
                    markeredgewidth=0.55,
                    linestyle=linestyle,
                    label=label,
                )
                axis.fill_between(
                    x,
                    np.maximum(0, mean - sd),
                    np.minimum(100, mean + sd),
                    color=color,
                    alpha=0.11,
                    linewidth=0,
                )
            axis.set_xlim(0.7, tasks + 0.3)
            axis.set_ylim(y_min, 101)
            axis.set_xlabel("Tasks observed")
            axis.set_ylabel("Exact-class accuracy (%)")
            axis.set_xticks(np.arange(1, tasks + 1, 1 if tasks == 10 else 2))
            axis.tick_params(labelsize=7.5)
            axis.xaxis.label.set_size(8.5)
            axis.yaxis.label.set_size(8.5)
            axis.grid(axis="y", color="#D8D8D8", linewidth=0.45)
            axis.set_axisbelow(True)
            handles, labels = axis.get_legend_handles_labels()
            figure.legend(
                handles,
                labels,
                loc="lower center",
                ncol=2,
                frameon=False,
                bbox_to_anchor=(0.5, 0.0),
                fontsize=7.2,
                handlelength=2.2,
                handletextpad=0.55,
                columnspacing=1.15,
                labelspacing=0.35,
                borderaxespad=0.1,
            )
            figure.tight_layout(rect=(0, 0.245, 1, 1), pad=0.45)
            panel_destination = destination / f"{dataset}_{tasks}tasks"
            export(figure, panel_destination)
            plt.close(figure)
            destinations.append(panel_destination.with_suffix(".pdf"))
    return destinations


def format_value(mean: float, sd: float, marker: str = "") -> str:
    return f"{100 * mean:.1f} ± {100 * sd:.1f}{marker}"


def render_table(results: CapacityResults, backbone: str, destination: Path) -> None:
    rows: list[list[str]] = []
    bold_cells: set[tuple[int, int]] = set()
    for row_index, (dataset, tasks) in enumerate(SETTINGS, start=1):
        subset = selected_subset(results, dataset, backbone, tasks)
        subset = subset[subset.tasks_seen == tasks].set_index("paper_method")
        global_row = subset.loc["task_global_softmax"]
        local_row = subset.loc["task_local_softmax"]
        flow_row = subset.loc["selected_flow"]
        joint_row = subset.loc["offline_joint"]
        cross = "†" if dataset == "imagenet_r" else ""
        rows.append(
            [
                DATASETS[dataset],
                str(tasks),
                format_value(
                    global_row.class_accuracy_mean,
                    global_row.class_accuracy_std,
                ),
                format_value(local_row.class_accuracy_mean, local_row.class_accuracy_std),
                format_value(flow_row.class_accuracy_mean, flow_row.class_accuracy_std, cross),
                format_value(joint_row.class_accuracy_mean, joint_row.class_accuracy_std),
            ]
        )
        # Bold the best implemented continual method; offline joint is a reference.
        continual_means = [
            global_row.class_accuracy_mean,
            local_row.class_accuracy_mean,
            flow_row.class_accuracy_mean,
        ]
        bold_cells.add((row_index, 2 + int(np.argmax(continual_means))))

    columns = [
        "Dataset",
        "Tasks",
        "Logit-based\nrouting",
        "Confidence-based\nrouting",
        "Flow routing",
        "Offline joint",
    ]
    with plt.rc_context(paper_style()):
        figure, axis = plt.subplots(figsize=(12.0, 3.3))
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=columns,
            cellLoc="center",
            colLoc="center",
            loc="center",
            colWidths=[0.17, 0.08, 0.18, 0.19, 0.18, 0.18],
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
        figure.text(
            0.02,
            0.08,
            "Mean ± sample SD across three seeds. Bold indicates the best implemented "
            "continual method; offline joint is a reference.",
            fontsize=8.5,
        )
        figure.text(
            0.02,
            0.025,
            "† ImageNet-R uses a 1024D flow; CIFAR-100 uses a 128D flow.",
            fontsize=8.5,
        )
        export(figure, destination)
        plt.close(figure)


def generate_method_comparison_figures(
    experiment_root: Path, output_dir: Path
) -> list[Path]:
    """Render eight standalone curves and two summary tables."""

    results = load_capacity_results(Path(experiment_root).resolve())
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    destinations: list[Path] = []
    for backbone, short in (("vit_b16", "vit"), ("resnet50", "resnet")):
        curves = output / short
        table = output / f"method_comparison_final_exact_class_{short}"
        destinations.extend(render_curves(results, backbone, curves))
        render_table(
            results,
            backbone,
            table,
        )
        destinations.append(table.with_suffix(".pdf"))
    return destinations

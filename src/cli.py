"""Unified command-line interface for the implemented experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .bias_accumulation.experiment import enumerate_runs, run_bias_accumulation
from .config import load_config, resolved_config
from .core.aggregation import aggregate_core_results
from .core.dimension_sweep import run_dimension_sweep
from .core.experiment import run_core
from .data import DATASET_NUM_CLASSES, create_imagenet_r_split, extract_feature_caches
from .figures import (
    generate_bias_accumulation_figure,
    generate_canonical_position_figure,
    generate_flow_routing_figures,
    generate_method_comparison_figures,
)

DATASETS = tuple(sorted(DATASET_NUM_CLASSES))
BACKBONES = ("vit_b16", "resnet50")
DIMENSIONS = (64, 128, 256, 1024)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    parser.add_argument("--config", default="configs/experiments.yaml")
    commands = parser.add_subparsers(dest="command", required=True)

    split = commands.add_parser("prepare-imagenet-r")
    split.add_argument("--overwrite", action="store_true")

    bias = commands.add_parser("bias")
    bias_commands = bias.add_subparsers(dest="bias_command", required=True)
    bias_run = bias_commands.add_parser("run")
    _add_partition_filters(bias_run, canonical_position=True)
    bias_run.add_argument("--max-runs", type=int)
    bias_run.add_argument("--overwrite", action="store_true")
    bias_run.add_argument("--dry-run", action="store_true")
    bias_list = bias_commands.add_parser("list")
    _add_partition_filters(bias_list, canonical_position=True)

    core = commands.add_parser("core")
    core_commands = core.add_subparsers(dest="core_command", required=True)
    cache = core_commands.add_parser("cache")
    cache.add_argument("--dataset", action="append", choices=DATASETS)
    cache.add_argument("--backbone", action="append", choices=BACKBONES)
    cache.add_argument("--overwrite", action="store_true")
    core_run = core_commands.add_parser("run")
    _add_partition_filters(core_run)
    core_run.add_argument("--projection-dim", type=int, choices=DIMENSIONS, default=64)
    core_run.add_argument("--max-flow-epochs", type=int)
    core_run.add_argument("--max-stages", type=int)
    core_run.add_argument("--no-aggregate", action="store_true")
    core_run.add_argument("--overwrite", action="store_true")
    dimensions = core_commands.add_parser("dimensions")
    _add_partition_filters(dimensions)
    dimensions.add_argument(
        "--projection-dim", action="append", type=int, choices=DIMENSIONS
    )
    dimensions.add_argument("--max-flow-epochs", type=int)
    dimensions.add_argument("--max-stages", type=int)
    dimensions.add_argument("--no-aggregate", action="store_true")
    dimensions.add_argument("--overwrite", action="store_true")
    core_commands.add_parser("aggregate")

    figures = commands.add_parser("figures")
    figure_commands = figures.add_subparsers(dest="figure_command", required=True)
    bias_figure = figure_commands.add_parser("bias-accumulation")
    _add_bias_figure_arguments(bias_figure, canonical_position=True)
    position_figure = figure_commands.add_parser("canonical-position")
    _add_bias_figure_arguments(position_figure)
    method_figure = figure_commands.add_parser("method-comparison")
    method_figure.add_argument("--experiment-root", type=Path)
    method_figure.add_argument("--output-dir", type=Path)
    flow_figure = figure_commands.add_parser("flow-routing")
    flow_figure.add_argument("--experiment-root", type=Path)
    flow_figure.add_argument("--tsne-root", type=Path)
    flow_figure.add_argument("--output-dir", type=Path)
    flow_figure.add_argument("--tsne-dataset", choices=DATASETS, default="cifar100")
    flow_figure.add_argument("--tsne-tasks", type=int, choices=(10, 20), default=20)
    flow_figure.add_argument("--tsne-seed", type=int, default=1993)
    flow_figure.add_argument(
        "--decomposition-tasks", type=int, choices=(10, 20), default=20
    )
    flow_figure.add_argument(
        "--decomposition-dim", type=int, choices=DIMENSIONS, default=1024
    )
    all_figures = figure_commands.add_parser("all")
    all_figures.add_argument("--bias-runs-root", type=Path)
    all_figures.add_argument("--experiment-root", type=Path)
    all_figures.add_argument("--tsne-root", type=Path)
    all_figures.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    root = repository_root()
    config = load_config(args.config)

    if args.command == "prepare-imagenet-r":
        dataset = resolved_config(config, root)["datasets"]["imagenet_r"]
        path = create_imagenet_r_split(
            dataset["root"],
            dataset["split_manifest"],
            dataset["train_fraction"],
            dataset["split_seed"],
            overwrite=args.overwrite,
        )
        print(path)
        return 0

    if args.command == "bias":
        runs = enumerate_runs(
            config,
            datasets=args.dataset,
            backbones=args.backbone,
            task_counts=args.tasks,
            seeds=args.seed,
            canonical_positions=args.canonical_position,
        )
        if args.bias_command == "list":
            for run in runs:
                print(run.relative_directory().as_posix())
            return 0
        summary = run_bias_accumulation(
            config,
            root,
            runs=runs,
            resume=not args.overwrite,
            max_runs=args.max_runs,
            dry_run=args.dry_run,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["failed"] == 0 else 1

    if args.command == "figures":
        paths = _generate_figures(args, root)
        print(json.dumps({"figures": [str(path) for path in paths]}, indent=2))
        return 0

    if args.core_command == "cache":
        caches = extract_feature_caches(
            config,
            root,
            datasets=args.dataset,
            backbones=args.backbone,
            overwrite=args.overwrite,
        )
        print(json.dumps({"caches": [str(path) for path in caches]}, indent=2))
        return 0
    if args.core_command == "run":
        summary = run_core(
            config,
            root,
            datasets=args.dataset,
            backbones=args.backbone,
            task_counts=args.tasks,
            seeds=args.seed,
            projection_dim=args.projection_dim,
            max_flow_epochs=args.max_flow_epochs,
            max_stages=args.max_stages,
            aggregate=not args.no_aggregate,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary["failed"] == 0 else 1
    if args.core_command == "dimensions":
        summary = run_dimension_sweep(
            config,
            root,
            datasets=args.dataset,
            backbones=args.backbone,
            task_counts=args.tasks,
            seeds=args.seed,
            dimensions=args.projection_dim,
            max_flow_epochs=args.max_flow_epochs,
            max_stages=args.max_stages,
            aggregate=not args.no_aggregate,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2))
        return 0
    if args.core_command == "aggregate":
        output_root = Path(resolved_config(config, root)["project"]["output_root"])
        artifacts = aggregate_core_results(output_root / "core")
        print(
            json.dumps({key: str(value) for key, value in artifacts.items()}, indent=2)
        )
        return 0
    raise AssertionError("Unhandled command")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _add_partition_filters(
    parser: argparse.ArgumentParser, canonical_position: bool = False
) -> None:
    parser.add_argument("--dataset", action="append", choices=DATASETS)
    parser.add_argument("--backbone", action="append", choices=BACKBONES)
    parser.add_argument("--tasks", action="append", type=int, choices=(10, 20))
    parser.add_argument("--seed", action="append", type=int)
    if canonical_position:
        parser.add_argument("--canonical-position", action="append", type=int)


def _add_bias_figure_arguments(
    parser: argparse.ArgumentParser, *, canonical_position: bool = False
) -> None:
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset", choices=DATASETS, default="cifar100")
    parser.add_argument("--backbone", choices=BACKBONES, default="vit_b16")
    parser.add_argument("--tasks", type=int, choices=(10, 20), default=10)
    parser.add_argument("--seed", action="append", type=int)
    if canonical_position:
        parser.add_argument("--canonical-position", type=int, default=0)


def _generate_figures(args: argparse.Namespace, root: Path) -> list[Path]:
    imported_data = root / "figure_data"
    figure_output = root / "output"
    bias_root = imported_data / "bias_accumulation" / "runs"
    core_root = imported_data / "core"
    tsne_root = imported_data / "tsne"

    if args.figure_command == "bias-accumulation":
        return generate_bias_accumulation_figure(
            args.runs_root or bias_root,
            args.output_dir or figure_output / "bias_accumulation",
            dataset=args.dataset,
            backbone=args.backbone,
            tasks=args.tasks,
            canonical_position=args.canonical_position,
            seeds=tuple(args.seed or (1993, 1994, 1995)),
        )
    if args.figure_command == "canonical-position":
        return [
            generate_canonical_position_figure(
                args.runs_root or bias_root,
                args.output_dir or figure_output / "bias_accumulation",
                dataset=args.dataset,
                backbone=args.backbone,
                tasks=args.tasks,
                seeds=tuple(args.seed or (1993, 1994, 1995)),
            )
        ]
    if args.figure_command == "method-comparison":
        return generate_method_comparison_figures(
            args.experiment_root or core_root,
            args.output_dir or figure_output / "method_comparison",
        )
    if args.figure_command == "flow-routing":
        return generate_flow_routing_figures(
            args.experiment_root or core_root,
            args.tsne_root or tsne_root,
            args.output_dir or figure_output / "flow_routing",
            tsne_dataset=args.tsne_dataset,
            tsne_tasks=args.tsne_tasks,
            tsne_seed=args.tsne_seed,
            decomposition_tasks=args.decomposition_tasks,
            decomposition_dim=args.decomposition_dim,
        )
    if args.figure_command == "all":
        bias_root = args.bias_runs_root or bias_root
        core_root = args.experiment_root or core_root
        tsne_root = args.tsne_root or tsne_root
        figure_output = args.output_root or figure_output
        paths = generate_bias_accumulation_figure(
            bias_root, figure_output / "bias_accumulation"
        )
        paths.append(
            generate_canonical_position_figure(
                bias_root, figure_output / "bias_accumulation"
            )
        )
        paths.extend(
            generate_method_comparison_figures(
                core_root, figure_output / "method_comparison"
            )
        )
        paths.extend(
            generate_flow_routing_figures(
                core_root, tsne_root, figure_output / "flow_routing"
            )
        )
        return paths
    raise AssertionError("Unhandled figure command")


if __name__ == "__main__":
    raise SystemExit(main())

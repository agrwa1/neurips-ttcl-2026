"""Resumable canonical-position bias accumulation experiment."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from ..artifacts import (
    RunIdentity,
    ensure_backbone_checkpoint,
    is_done,
    mark_done,
    save_npz,
    save_torch,
    write_dataframe,
    write_json,
)
from ..config import resolved_config, task_counts_for_dataset
from ..data import DATASET_NUM_CLASSES, ClassSubset, load_dataset_pair
from ..models import build_frozen_backbone
from ..reproducibility import environment_metadata, seed_everything
from ..tasks import canonical_insertion_sequence, create_partition
from .evaluation import canonical_capture_table, run_inference
from .training import train_sequence

LOGGER = logging.getLogger(__name__)


def enumerate_runs(
    config: dict,
    datasets: Iterable[str] | None = None,
    backbones: Iterable[str] | None = None,
    task_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    canonical_positions: Iterable[int] | None = None,
) -> list[RunIdentity]:
    dataset_values = list(datasets or config["datasets"].keys())
    backbone_values = list(backbones or config["backbones"])
    task_override = (
        None if task_counts is None else [int(value) for value in task_counts]
    )
    seed_values = [int(value) for value in (seeds or config["seeds"])]
    position_filter = (
        None if canonical_positions is None else set(map(int, canonical_positions))
    )
    runs = []
    # Group by backbone so a complete sweep needs only one large feature extractor
    # resident on the GPU at a time.
    for backbone in backbone_values:
        for dataset in dataset_values:
            for num_tasks in task_counts_for_dataset(config, dataset, task_override):
                for seed in seed_values:
                    for position in range(num_tasks):
                        if position_filter is None or position in position_filter:
                            runs.append(
                                RunIdentity(
                                    dataset=dataset,
                                    backbone=backbone,
                                    num_tasks=num_tasks,
                                    seed=seed,
                                    canonical_position=position,
                                )
                            )
    return runs


def run_bias_accumulation(
    config: dict,
    repository_root: str | Path,
    runs: list[RunIdentity] | None = None,
    resume: bool = True,
    max_runs: int | None = None,
    dry_run: bool = False,
) -> dict[str, int]:
    root = Path(repository_root).resolve()
    config = resolved_config(config, root)
    selected_runs = enumerate_runs(config) if runs is None else runs
    if max_runs is not None:
        selected_runs = selected_runs[: int(max_runs)]
    summary = {"planned": len(selected_runs), "completed": 0, "skipped": 0, "failed": 0}
    if dry_run:
        for identity in selected_runs:
            print(identity.relative_directory().as_posix())
        return summary

    _configure_torch_multiprocessing(config["runtime"])
    output_root = Path(config["project"]["output_root"])
    experiment_root = output_root / "bias_accumulation"
    run_root = experiment_root / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(experiment_root / "resolved_config.json", config)
    write_json(experiment_root / "environment.json", environment_metadata(root))

    dataset_cache: dict[str, tuple[object, object]] = {}
    backbone_provenance: dict[str, dict[str, str]] = {}
    active_backbone_name: str | None = None
    active_backbone = None

    for index, identity in enumerate(selected_runs, start=1):
        run_directory = run_root / identity.relative_directory()
        if resume and _run_is_complete(
            run_directory,
            require_logits=bool(config["evaluation"].get("save_logits", True)),
        ):
            LOGGER.info(
                "[%d/%d] skipping completed run %s",
                index,
                len(selected_runs),
                identity.run_id,
            )
            summary["skipped"] += 1
            continue
        run_directory.mkdir(parents=True, exist_ok=True)
        (run_directory / "DONE.json").unlink(missing_ok=True)
        _configure_run_logging(run_directory)
        LOGGER.info("[%d/%d] starting %s", index, len(selected_runs), asdict(identity))

        try:
            if identity.dataset not in dataset_cache:
                dataset_cache[identity.dataset] = load_dataset_pair(
                    identity.dataset, config["datasets"][identity.dataset]
                )
            if identity.backbone != active_backbone_name:
                if active_backbone is not None:
                    active_backbone.to("cpu")
                    _clear_cuda_cache()
                active_backbone = build_frozen_backbone(identity.backbone)
                active_backbone_name = identity.backbone
                backbone_provenance[identity.backbone] = ensure_backbone_checkpoint(
                    active_backbone, output_root
                )

            seed_everything(
                identity.seed, bool(config["runtime"].get("deterministic", True))
            )
            partition = create_partition(
                dataset=identity.dataset,
                num_classes=DATASET_NUM_CLASSES[identity.dataset],
                num_tasks=identity.num_tasks,
                seed=identity.seed,
            )
            task_sequence = canonical_insertion_sequence(
                identity.num_tasks, identity.canonical_position
            )
            train_dataset, test_dataset = dataset_cache[identity.dataset]
            backbone = active_backbone

            run_manifest = {
                "identity": asdict(identity),
                "run_id": identity.run_id,
                "partition": partition.as_dict(),
                "task_sequence": list(task_sequence),
                "backbone": backbone_provenance[identity.backbone],
                "test_time_updates": False,
                "training": config["training"],
                "evaluation": config["evaluation"],
                "runtime": config["runtime"],
            }
            write_json(run_directory / "run_manifest.json", run_manifest)
            _write_partition_tables(run_directory, partition, task_sequence)

            training = train_sequence(
                backbone=backbone,
                train_dataset=train_dataset,
                partition=partition,
                task_sequence=task_sequence,
                training_config=config["training"],
                runtime_config=config["runtime"],
            )
            write_dataframe(
                run_directory / "training_history.csv",
                _attach(training.history, identity),
            )
            write_dataframe(
                run_directory / "classifier_statistics.csv",
                _attach(training.classifier_statistics, identity),
            )

            checkpoint_payload = {
                "identity": asdict(identity),
                "run_id": identity.run_id,
                "classifier_state_dict": training.classifier.state_dict(),
                "optimizer_state_dict": (
                    training.optimizer_state
                    if config["checkpointing"].get("save_optimizer", True)
                    else None
                ),
                "output_to_original_class": training.output_to_original_class,
                "partition": partition.as_dict(),
                "task_sequence": list(task_sequence),
                "backbone": backbone_provenance[identity.backbone],
            }
            save_torch(run_directory / "final_classifier.pt", checkpoint_payload)

            canonical_test = ClassSubset(test_dataset, partition.tasks[0])
            canonical_result = run_inference(
                backbone,
                training.classifier,
                canonical_test,
                partition,
                training.output_to_original_class,
                config["runtime"],
                config["evaluation"],
            )
            canonical_predictions = _attach(canonical_result.predictions, identity)
            write_dataframe(
                run_directory / "canonical_predictions.csv.gz",
                canonical_predictions,
                compression="gzip",
            )
            capture = canonical_capture_table(
                canonical_result.predictions,
                task_sequence,
                identity.canonical_position,
                identity.num_tasks,
            )
            write_dataframe(
                run_directory / "capture_by_task.csv", _attach(capture, identity)
            )
            if config["evaluation"].get("save_logits", True):
                save_npz(
                    run_directory / "canonical_logits.npz",
                    logits=canonical_result.logits,
                    output_to_original_class=canonical_result.output_to_original_class,
                    sample_id=canonical_result.predictions["sample_id"].to_numpy(
                        dtype=str
                    ),
                )

            mark_done(
                run_directory,
                {
                    "identity": asdict(identity),
                    "run_id": identity.run_id,
                    "training_wall_time_seconds": training.wall_time_seconds,
                },
            )
            (run_directory / "FAILED.json").unlink(missing_ok=True)
            summary["completed"] += 1
            LOGGER.info("completed run %s", identity.run_id)
        except (
            Exception
        ) as error:  # keep a structured failure record before respecting policy
            summary["failed"] += 1
            write_json(
                run_directory / "FAILED.json",
                {
                    "identity": asdict(identity),
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            LOGGER.exception("run failed: %s", identity.run_id)
            if not config["runtime"].get("continue_on_error", False):
                raise
    write_json(experiment_root / "last_run_summary.json", summary)
    return summary


def _attach(dataframe: pd.DataFrame, identity: RunIdentity) -> pd.DataFrame:
    result = dataframe.copy()
    metadata = asdict(identity)
    metadata["run_id"] = identity.run_id
    for key, value in metadata.items():
        if key in result.columns:
            if not result[key].eq(value).all():
                raise ValueError(
                    f"Artifact column {key!r} conflicts with run metadata for {identity.run_id}."
                )
            continue
        result.insert(len(result.columns), key, value)
    return result


def _write_partition_tables(
    run_directory: Path, partition, sequence: tuple[int, ...]
) -> None:
    class_rows = []
    sequence_position = {
        semantic_task: index for index, semantic_task in enumerate(sequence)
    }
    for semantic_task, classes in enumerate(partition.tasks):
        for within_task_index, original_class in enumerate(classes):
            class_rows.append(
                {
                    "original_class": original_class,
                    "semantic_task": semantic_task,
                    "within_task_index": within_task_index,
                    "sequence_position": sequence_position[semantic_task],
                }
            )
    write_dataframe(run_directory / "class_partition.csv", pd.DataFrame(class_rows))
    write_dataframe(
        run_directory / "task_sequence.csv",
        pd.DataFrame(
            {
                "sequence_position": range(len(sequence)),
                "semantic_task": sequence,
                "is_canonical": [task == 0 for task in sequence],
            }
        ),
    )


def _configure_run_logging(run_directory: Path) -> None:
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for old_handler in list(root.handlers):
        if getattr(old_handler, "_bias_accumulation_run_handler", False):
            root.removeHandler(old_handler)
            old_handler.close()
    log_path = run_directory / "run.log"
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler._bias_accumulation_run_handler = True
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root.addHandler(handler)


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _configure_torch_multiprocessing(runtime_config: dict) -> None:
    """Configure tensor IPC once, before any DataLoader workers are created."""

    sharing_strategy = runtime_config.get("sharing_strategy")
    if sharing_strategy:
        import torch

        torch.multiprocessing.set_sharing_strategy(str(sharing_strategy))


def _run_is_complete(run_directory: Path, *, require_logits: bool) -> bool:
    required = [
        "run_manifest.json",
        "class_partition.csv",
        "task_sequence.csv",
        "training_history.csv",
        "classifier_statistics.csv",
        "final_classifier.pt",
        "canonical_predictions.csv.gz",
        "capture_by_task.csv",
    ]
    if require_logits:
        required.append("canonical_logits.npz")
    return is_done(run_directory) and all(
        (run_directory / filename).is_file() for filename in required
    )

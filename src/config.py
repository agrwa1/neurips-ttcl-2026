"""Configuration loading and validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .data import DATASET_NUM_CLASSES


class ConfigError(ValueError):
    """Raised when the experiment configuration is internally inconsistent."""


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError(f"Expected a mapping in {config_path}.")
    config = deepcopy(config)
    config["_config_path"] = str(config_path)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    for key in ("project", "datasets", "training", "evaluation", "runtime", "core"):
        if key not in config:
            raise ConfigError(f"Missing required configuration section: {key}")

    seeds = config.get("seeds", [])
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ConfigError("Exactly three distinct seeds are required.")

    expected_backbones = {"vit_b16", "resnet50"}
    if set(config.get("backbones", [])) != expected_backbones:
        raise ConfigError(f"backbones must be {sorted(expected_backbones)}.")

    for name in ("cifar100", "imagenet_r"):
        if name not in config["datasets"]:
            raise ConfigError(f"Missing dataset configuration: {name}")

    unknown_datasets = set(config["datasets"]) - set(DATASET_NUM_CLASSES)
    if unknown_datasets:
        raise ConfigError(f"Unknown datasets: {sorted(unknown_datasets)}")
    for name in config["datasets"]:
        task_counts_for_dataset(config, name)
    dimensions = [int(value) for value in config["core"].get("projection_dims", [])]
    if dimensions != [64, 128, 256, 1024]:
        raise ConfigError("core.projection_dims must be [64, 128, 256, 1024].")


def task_counts_for_dataset(
    config: dict[str, Any],
    dataset_name: str,
    override: Any = None,
) -> list[int]:
    """Resolve and validate task counts, allowing dataset-specific defaults."""

    if dataset_name not in DATASET_NUM_CLASSES:
        raise ConfigError(f"Unknown dataset: {dataset_name}")
    if dataset_name not in config.get("datasets", {}):
        raise ConfigError(f"Missing dataset configuration: {dataset_name}")
    raw_values = (
        override
        if override is not None
        else config["datasets"][dataset_name].get(
            "task_counts", config.get("task_counts", [])
        )
    )
    values = [int(value) for value in raw_values]
    if (
        not values
        or any(value < 1 for value in values)
        or len(values) != len(set(values))
    ):
        raise ConfigError(
            f"Task counts for {dataset_name} must be distinct positive integers."
        )
    num_classes = DATASET_NUM_CLASSES[dataset_name]
    invalid = [value for value in values if value > num_classes or num_classes % value]
    if invalid:
        raise ConfigError(
            f"Task counts {invalid} do not divide the {num_classes} classes in {dataset_name}."
        )
    return values


def resolved_config(
    config: dict[str, Any], repository_root: str | Path
) -> dict[str, Any]:
    """Return a copy with project-relative paths resolved against the repository."""

    root = Path(repository_root).resolve()
    result = deepcopy(config)
    result["project"]["output_root"] = str(
        _resolve(root, result["project"]["output_root"])
    )
    for dataset in result["datasets"].values():
        dataset["root"] = str(_resolve(root, dataset["root"]))
        if "split_manifest" in dataset:
            dataset["split_manifest"] = str(_resolve(root, dataset["split_manifest"]))
    return result


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()

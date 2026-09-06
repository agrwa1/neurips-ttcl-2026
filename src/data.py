"""Dataset preparation and image transforms."""

from __future__ import annotations

import csv
import os
import tempfile
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DATASET_NUM_CLASSES = {
    "cifar100": 100,
    "imagenet_r": 200,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class ImageNetRSplitRow:
    sample_id: str
    relative_path: str
    class_id: int
    class_name: str
    split: str


@dataclass(frozen=True)
class FeatureCachePaths:
    """Files comprising one complete frozen-feature cache."""

    directory: Path
    train_features: Path
    train_labels: Path
    train_sample_ids: Path
    test_features: Path
    test_labels: Path
    test_sample_ids: Path
    metadata: Path
    done: Path


def feature_cache_paths(
    output_root: str | Path,
    dataset: str,
    backbone: str,
) -> FeatureCachePaths:
    directory = Path(output_root) / "features" / dataset / backbone
    return FeatureCachePaths(
        directory=directory,
        train_features=directory / "train_features.npy",
        train_labels=directory / "train_classes.npy",
        train_sample_ids=directory / "train_sample_ids.npy",
        test_features=directory / "test_features.npy",
        test_labels=directory / "test_classes.npy",
        test_sample_ids=directory / "test_sample_ids.npy",
        metadata=directory / "metadata.json",
        done=directory / "DONE.json",
    )


def extract_feature_caches(
    config: dict[str, Any],
    repository_root: str | Path,
    datasets: Iterable[str] | None = None,
    backbones: Iterable[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Extract deterministic train/test features for the core experiment."""

    from .artifacts import ensure_backbone_checkpoint, mark_done, write_json
    from .config import resolved_config
    from .models import build_frozen_backbone
    from .reproducibility import environment_metadata, seed_everything

    root = Path(repository_root).resolve()
    resolved = resolved_config(config, root)
    output_root = Path(resolved["project"]["output_root"])
    feature_config = resolved.get("features", {})
    runtime = resolved["runtime"]
    completed: list[Path] = []
    for backbone_name in list(backbones or resolved["backbones"]):
        backbone = build_frozen_backbone(backbone_name)
        backbone_provenance = ensure_backbone_checkpoint(backbone, output_root)
        for dataset_name in list(datasets or resolved["datasets"]):
            paths = feature_cache_paths(output_root, dataset_name, backbone_name)
            if not overwrite and feature_cache_complete(paths):
                completed.append(paths.directory)
                continue
            seed_everything(
                int(feature_config.get("random_seed", 2026)),
                bool(runtime.get("deterministic", True)),
            )
            train_dataset, test_dataset = load_feature_extraction_pair(
                dataset_name,
                resolved["datasets"][dataset_name],
            )
            started = time.monotonic()
            train = _encode_dataset(
                backbone,
                train_dataset,
                runtime,
                int(feature_config.get("extraction_batch_size", 256)),
            )
            test = _encode_dataset(
                backbone,
                test_dataset,
                runtime,
                int(feature_config.get("extraction_batch_size", 256)),
            )
            _atomic_save_npy(paths.train_features, train["features"])
            _atomic_save_npy(paths.train_labels, train["labels"])
            _atomic_save_npy(paths.train_sample_ids, train["sample_ids"])
            _atomic_save_npy(paths.test_features, test["features"])
            _atomic_save_npy(paths.test_labels, test["labels"])
            _atomic_save_npy(paths.test_sample_ids, test["sample_ids"])
            write_json(
                paths.metadata,
                {
                    "dataset": dataset_name,
                    "backbone": backbone_name,
                    "feature_dim": int(backbone.spec.feature_dim),
                    "train_examples": int(len(train["labels"])),
                    "test_examples": int(len(test["labels"])),
                    "transform": "evaluation_transform_for_both_splits",
                    "backbone_provenance": backbone_provenance,
                    "runtime": runtime,
                    "environment": environment_metadata(root),
                    "wall_time_seconds": time.monotonic() - started,
                },
            )
            mark_done(
                paths.directory, {"dataset": dataset_name, "backbone": backbone_name}
            )
            completed.append(paths.directory)
        backbone.to("cpu")
    return completed


def load_feature_cache(
    output_root: str | Path,
    dataset: str,
    backbone: str,
    *,
    test_only: bool = False,
) -> dict[str, np.ndarray]:
    """Memory-map a complete frozen-feature cache."""

    paths = feature_cache_paths(output_root, dataset, backbone)
    required = [paths.test_features, paths.test_labels, paths.done]
    if not test_only:
        required.extend([paths.train_features, paths.train_labels])
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing frozen-feature cache. Run `python -m src.cli core cache` first. "
            f"Missing: {missing}"
        )
    result = {
        "test_features": np.load(paths.test_features, mmap_mode="r"),
        "test_classes": np.load(paths.test_labels),
    }
    if not test_only:
        result.update(
            {
                "train_features": np.load(paths.train_features, mmap_mode="r"),
                "train_classes": np.load(paths.train_labels),
            }
        )
    return result


def feature_cache_complete(paths: FeatureCachePaths) -> bool:
    return paths.done.is_file() and all(
        path.is_file()
        for path in (
            paths.train_features,
            paths.train_labels,
            paths.train_sample_ids,
            paths.test_features,
            paths.test_labels,
            paths.test_sample_ids,
            paths.metadata,
        )
    )


def train_transform():
    """Return the non-normalized 224px stochastic training transform."""

    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3 / 4, 4 / 3)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )


def evaluation_transform():
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def create_imagenet_r_split(
    dataset_root: str | Path,
    manifest_path: str | Path,
    train_fraction: float = 0.8,
    seed: int = 2026,
    overwrite: bool = False,
) -> Path:
    """Create a deterministic per-class train/test split without moving image files."""

    root = Path(dataset_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if manifest.exists() and not overwrite:
        validate_imagenet_r_manifest(root, manifest)
        return manifest
    if not root.is_dir():
        raise FileNotFoundError(f"ImageNet-R directory does not exist: {root}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between zero and one.")

    class_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    # A raw ImageNet-R release contains one directory per class. Avoid silently treating an
    # existing train/test layout as two classes.
    if {path.name for path in class_dirs}.issubset({"train", "test", "val"}):
        raise ValueError(
            "Expected raw ImageNet-R class directories, but found a pre-split layout. "
            "Point datasets.imagenet_r.root at the directory containing the 200 class folders."
        )
    if len(class_dirs) != 200:
        raise ValueError(
            f"Expected 200 ImageNet-R class directories under {root}, found {len(class_dirs)}."
        )

    rows: list[ImageNetRSplitRow] = []
    rng = np.random.RandomState(seed)
    for class_id, class_dir in enumerate(class_dirs):
        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(images) < 2:
            raise ValueError(f"Class {class_dir.name} contains fewer than two images.")
        permutation = rng.permutation(len(images))
        train_count = int(np.floor(len(images) * train_fraction))
        train_count = min(max(train_count, 1), len(images) - 1)
        train_indices = set(int(index) for index in permutation[:train_count])
        for image_index, image_path in enumerate(images):
            relative = image_path.relative_to(root).as_posix()
            rows.append(
                ImageNetRSplitRow(
                    sample_id=relative,
                    relative_path=relative,
                    class_id=class_id,
                    class_name=class_dir.name,
                    split="train" if image_index in train_indices else "test",
                )
            )

    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ImageNetRSplitRow.__annotations__)
        )
        writer.writeheader()
        writer.writerows(row.__dict__ for row in rows)
    validate_imagenet_r_manifest(root, manifest)
    return manifest


def validate_imagenet_r_manifest(
    dataset_root: str | Path, manifest_path: str | Path
) -> None:
    root = Path(dataset_root).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    class_ids: set[int] = set()
    split_counts = {"train": 0, "test": 0}
    missing: list[str] = []
    with manifest.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_ids.add(int(row["class_id"]))
            split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
            if len(missing) < 10 and not (root / row["relative_path"]).is_file():
                missing.append(row["relative_path"])
    if class_ids != set(range(200)):
        raise ValueError(
            "ImageNet-R manifest must contain exactly the contiguous class IDs 0..199."
        )
    if not split_counts["train"] or not split_counts["test"]:
        raise ValueError(
            "ImageNet-R manifest must contain both train and test examples."
        )
    if missing:
        raise FileNotFoundError(
            f"Manifest references missing images (first entries): {missing}"
        )


class IndexedDataset:
    """Attach stable sample IDs and original class IDs to a torchvision-style dataset."""

    def __init__(
        self,
        dataset,
        sample_ids: Iterable[str],
        original_targets: Iterable[int],
        transform: Callable | None = None,
    ) -> None:
        self.dataset = dataset
        self.sample_ids = list(sample_ids)
        self.original_targets = [int(value) for value in original_targets]
        self.transform = transform
        if len(self.dataset) != len(self.sample_ids) or len(self.dataset) != len(
            self.original_targets
        ):
            raise ValueError("Dataset, sample ID, and target lengths must match.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _ = self.dataset[index]
        if self.transform is not None:
            image = self.transform(image)
        return self.sample_ids[index], image, self.original_targets[index]


class ImageNetRManifestDataset:
    def __init__(
        self, root: str | Path, manifest: str | Path, split: str, transform=None
    ) -> None:
        from PIL import Image

        self._image_type = Image.Image
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        with Path(manifest).open("r", newline="", encoding="utf-8") as handle:
            self.rows = [row for row in csv.DictReader(handle) if row["split"] == split]
        if not self.rows:
            raise ValueError(f"No ImageNet-R samples found for split={split!r}.")
        self.targets = [int(row["class_id"]) for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        with Image.open(self.root / row["relative_path"]) as handle:
            image = handle.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return row["sample_id"], image, int(row["class_id"])


class ClassSubset:
    """A lightweight class-filtered view that retains original labels."""

    def __init__(self, dataset, class_ids: Iterable[int]) -> None:
        allowed = set(int(value) for value in class_ids)
        targets = _targets_for(dataset)
        self.dataset = dataset
        self.indices = [
            index for index, target in enumerate(targets) if int(target) in allowed
        ]
        if not self.indices:
            raise ValueError(f"Class subset is empty for classes {sorted(allowed)}.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


def load_dataset_pair(name: str, dataset_config: dict):
    """Return train/test datasets with stable IDs and original labels."""

    if name == "cifar100":
        return _load_cifar_pair(
            name,
            dataset_config,
            train_transform=train_transform(),
            test_transform=evaluation_transform(),
        )

    if name == "imagenet_r":
        manifest = create_imagenet_r_split(
            dataset_config["root"],
            dataset_config["split_manifest"],
            train_fraction=float(dataset_config.get("train_fraction", 0.8)),
            seed=int(dataset_config.get("split_seed", 2026)),
        )
        return (
            ImageNetRManifestDataset(
                dataset_config["root"],
                manifest,
                "train",
                transform=train_transform(),
            ),
            ImageNetRManifestDataset(
                dataset_config["root"], manifest, "test", transform=evaluation_transform()
            ),
        )
    raise KeyError(f"Unknown dataset: {name}")


def load_feature_extraction_pair(name: str, dataset_config: dict):
    """Return deterministic train/test views for frozen-feature caching.

    The ordinary training path intentionally augments training images.  A geometry
    cache must not give one sample a different feature every time it is read,
    so both splits use the deterministic evaluation transform here.
    """

    transform = evaluation_transform()
    if name == "cifar100":
        return _load_cifar_pair(
            name,
            dataset_config,
            train_transform=transform,
            test_transform=transform,
        )

    if name == "imagenet_r":
        manifest = create_imagenet_r_split(
            dataset_config["root"],
            dataset_config["split_manifest"],
            train_fraction=float(dataset_config.get("train_fraction", 0.8)),
            seed=int(dataset_config.get("split_seed", 2026)),
        )
        return (
            ImageNetRManifestDataset(
                dataset_config["root"], manifest, "train", transform=transform
            ),
            ImageNetRManifestDataset(
                dataset_config["root"], manifest, "test", transform=transform
            ),
        )
    raise KeyError(f"Unknown dataset: {name}")


def _load_cifar_pair(
    name: str,
    dataset_config: dict,
    train_transform,
    test_transform,
):
    from torchvision.datasets import CIFAR100

    if name != "cifar100":
        raise KeyError(f"Unknown CIFAR dataset: {name}")
    dataset_class = CIFAR100
    root = Path(dataset_config["root"])
    download = bool(dataset_config.get("download", True))
    train_base = dataset_class(root=root, train=True, download=download)
    test_base = dataset_class(root=root, train=False, download=download)
    train = IndexedDataset(
        train_base,
        (f"train/{index:05d}" for index in range(len(train_base))),
        train_base.targets,
        transform=train_transform,
    )
    test = IndexedDataset(
        test_base,
        (f"test/{index:05d}" for index in range(len(test_base))),
        test_base.targets,
        transform=test_transform,
    )
    train.targets = [int(value) for value in train_base.targets]
    test.targets = [int(value) for value in test_base.targets]
    return train, test


def _targets_for(dataset) -> list[int]:
    if hasattr(dataset, "targets"):
        return [int(value) for value in dataset.targets]
    if hasattr(dataset, "original_targets"):
        return [int(value) for value in dataset.original_targets]
    raise TypeError(f"Cannot obtain targets from {type(dataset).__name__}.")


def _encode_dataset(
    backbone,
    dataset,
    runtime: dict[str, Any],
    batch_size: int,
) -> dict[str, np.ndarray]:
    import torch
    from torch.utils.data import DataLoader

    from .bias_accumulation.training import _autocast, _shutdown_dataloader

    device = torch.device(runtime.get("device", "cuda"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA feature extraction was requested but CUDA is unavailable."
        )
    worker_count = int(runtime.get("num_workers", 0))
    options: dict[str, Any] = {}
    if worker_count:
        options["prefetch_factor"] = int(runtime.get("prefetch_factor", 1))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=worker_count,
        pin_memory=device.type == "cuda",
        persistent_workers=worker_count > 0,
        timeout=int(runtime.get("dataloader_timeout_seconds", 120))
        if worker_count
        else 0,
        **options,
    )
    backbone.to(device).eval()
    features = np.empty((len(dataset), backbone.spec.feature_dim), dtype=np.float32)
    labels = np.empty(len(dataset), dtype=np.int64)
    sample_ids: list[str] = []
    offset = 0
    iterator = iter(loader)
    try:
        for batch_sample_ids, images, batch_labels in iterator:
            images = images.to(device, non_blocking=True)
            with torch.no_grad(), _autocast(device, runtime):
                batch_features = backbone(images)
            count = batch_features.shape[0]
            features[offset : offset + count] = batch_features.float().cpu().numpy()
            labels[offset : offset + count] = np.asarray(batch_labels, dtype=np.int64)
            sample_ids.extend(map(str, batch_sample_ids))
            offset += count
    finally:
        _shutdown_dataloader(loader, iterator)
    if offset != len(dataset):
        raise RuntimeError(f"Encoded {offset} examples, expected {len(dataset)}.")
    return {
        "features": features,
        "labels": labels,
        "sample_ids": np.asarray(sample_ids, dtype=str),
    }


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp.npy", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.save(temporary, array, allow_pickle=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

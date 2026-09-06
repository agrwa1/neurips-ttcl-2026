"""Atomic, tabular-first experiment artifact storage."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .reproducibility import sha256_file, stable_hash


@dataclass(frozen=True)
class RunIdentity:
    dataset: str
    backbone: str
    num_tasks: int
    seed: int
    canonical_position: int

    @property
    def run_id(self) -> str:
        return stable_hash(self.__dict__)

    def relative_directory(self) -> Path:
        return Path(
            self.dataset,
            self.backbone,
            f"tasks_{self.num_tasks:02d}",
            f"seed_{self.seed}",
            f"canonical_position_{self.canonical_position:02d}_{self.run_id}",
        )


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n"
    )
    _atomic_write_bytes(destination, encoded.encode("utf-8"))
    return destination


def write_dataframe(
    path: str | Path, dataframe, compression: str | None = None
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=destination.name,
        suffix=suffix,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        if compression == "gzip":
            with gzip.open(
                temporary_path, "wt", encoding="utf-8", newline=""
            ) as handle:
                dataframe.to_csv(handle, index=False)
        else:
            dataframe.to_csv(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def save_npz(path: str | Path, **arrays) -> Path:
    import numpy as np

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name, suffix=".tmp.npz", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def save_torch(path: str | Path, payload: Any) -> Path:
    import torch

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name, suffix=".tmp.pt", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
    return destination


def ensure_backbone_checkpoint(backbone, output_root: str | Path) -> dict[str, str]:
    """Store one immutable copy of each frozen backbone and return provenance."""

    destination = Path(output_root) / "backbones" / f"{backbone.spec.name}.pt"
    if not destination.exists():
        save_torch(
            destination,
            {
                "name": backbone.spec.name,
                "feature_dim": backbone.spec.feature_dim,
                "source": backbone.spec.source,
                "state_dict": backbone.state_dict(),
            },
        )
    return {
        "path": str(destination.relative_to(Path(output_root))),
        "sha256": sha256_file(destination),
        "source": backbone.spec.source,
    }


def mark_done(run_directory: str | Path, payload: dict[str, Any]) -> None:
    write_json(Path(run_directory) / "DONE.json", payload)


def is_done(run_directory: str | Path) -> bool:
    return (Path(run_directory) / "DONE.json").is_file()


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=destination.name,
        suffix=".tmp",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")

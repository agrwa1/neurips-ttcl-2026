"""Data-independent projections used by flow routing."""

from __future__ import annotations

import hashlib

import numpy as np


def fixed_random_projection(
    input_dim: int,
    output_dim: int,
    seed: int,
    dataset: str,
    backbone: str,
) -> np.ndarray:
    """Create the reproducible Gaussian map used by the original experiment.

    Dimensions larger than 64 retain the exact historical 64 directions and
    append deterministically generated columns. The complete matrix is scaled
    for its requested output dimension, matching the reference implementation.
    """

    input_dim, output_dim = int(input_dim), int(output_dim)
    if output_dim < 2:
        raise ValueError("Projection dimension must be at least two.")
    digest = hashlib.sha256(f"{seed}:{dataset}:{backbone}".encode()).digest()
    generator = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    base_width = min(output_dim, 64)
    base_raw = generator.standard_normal((input_dim, base_width)).astype(np.float32)
    if output_dim <= 64:
        return base_raw / np.sqrt(float(output_dim))

    columns = [base_raw]
    for column in range(64, output_dim):
        column_digest = hashlib.sha256(
            f"{seed}:{dataset}:{backbone}:column:{column}".encode()
        ).digest()
        column_generator = np.random.default_rng(
            int.from_bytes(column_digest[:8], "little")
        )
        columns.append(
            column_generator.standard_normal((input_dim, 1)).astype(np.float32)
        )
    return np.concatenate(columns, axis=1) / np.sqrt(float(output_dim))

"""Pure NumPy reference kernels (always available)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def scale_minmax(x: NDArray[np.float32], lo: float = 0.0, hi: float = 1.0) -> NDArray[np.float32]:
    d_min, d_max = float(x.min()), float(x.max())
    if d_max == d_min:
        return np.full_like(x, lo)
    k = (hi - lo) / (d_max - d_min)
    return np.asarray((x - d_min) * k + lo, dtype=np.float32)


def normalize_l2(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Row-wise L2 normalisation; zero rows stay zero. Norms are accumulated in float64."""
    sq = np.einsum("ij,ij->i", x, x, dtype=np.float64)
    inv = np.divide(1.0, np.sqrt(sq), out=np.ones_like(sq), where=sq > 0).astype(np.float32)
    return np.asarray(x * inv[:, None], dtype=np.float32)


def dot(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)))

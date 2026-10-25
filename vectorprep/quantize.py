"""Symmetric int8 quantisation of embeddings, per tensor or per row, with error metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def quantize_int8(x: NDArray[np.floating], per_row: bool = False) -> tuple[NDArray[np.int8], NDArray[np.float32]]:
    """Map ``x`` to int8 in [-127, 127] with ``q = round(x / scale)``.

    Returns ``(q, scale)``; ``scale`` has shape ``()`` (per tensor) or ``(rows, 1)`` (per row).
    """
    x = np.asarray(x, dtype=np.float32)
    if per_row:
        if x.ndim != 2:
            raise ValueError("per_row quantisation needs a 2-D array")
        absmax = np.abs(x).max(axis=1, keepdims=True)
    else:
        absmax = np.asarray(np.abs(x).max())
    scale = np.where(absmax > 0, absmax / 127.0, 1.0).astype(np.float32)
    q = np.clip(np.rint(x / scale), -127, 127).astype(np.int8)
    return q, scale


def dequantize(q: NDArray[np.int8], scale: NDArray[np.float32]) -> NDArray[np.float32]:
    return (q.astype(np.float32) * scale).astype(np.float32)


def sqnr_db(x: NDArray[np.floating], x_hat: NDArray[np.floating]) -> float:
    """Signal-to-quantisation-noise ratio in dB (inf when exact)."""
    x64, e = np.asarray(x, np.float64), np.asarray(x, np.float64) - np.asarray(x_hat, np.float64)
    noise = float((e * e).sum())
    return float("inf") if noise == 0 else float(10 * np.log10((x64 * x64).sum() / noise))


def cosine_error(x: NDArray[np.floating], x_hat: NDArray[np.floating], pairs: int = 2000, seed: int = 0) -> float:
    """Max |cos(x_i, x_j) - cos(x̂_i, x̂_j)| over random row pairs (all-zero rows count as cos = 0)."""
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    i, j = rng.integers(0, n, pairs), rng.integers(0, n, pairs)

    def cos(m: NDArray[np.floating]) -> NDArray[np.float64]:
        m = np.asarray(m, np.float64)
        num = (m[i] * m[j]).sum(1)
        den = np.linalg.norm(m[i], axis=1) * np.linalg.norm(m[j], axis=1)
        # a row quantised to all zeros has undefined cosine; count it as 0
        return np.asarray(np.divide(num, den, out=np.zeros_like(num), where=den > 0), dtype=np.float64)

    return float(np.abs(cos(x) - cos(x_hat)).max())

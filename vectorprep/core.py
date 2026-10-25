"""``VectorPreprocessor``: one API over the NumPy kernels and the optional C++ backend.

v1 compatibility: same class name, method names and return types; ``has_cpp = False``
still forces the NumPy path. Behaviour changes in v2 (see README, "Bugs fixed"):
inputs are no longer silently modified in place (pass ``inplace=True`` to opt in), the
C++ path uses 64-bit sizes and float64 accumulators, and PCA is seeded and converges
to a tolerance instead of running a fixed 50 iterations from an unseeded start.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from vectorprep import kernels, native
from vectorprep.pca import pca_power
from vectorprep.quantize import quantize_int8

log = logging.getLogger(__name__)
Backend = Literal["auto", "numpy", "native"]


class VectorPreprocessor:
    """Preprocesses vectors/matrices with the C++ backend when built, else NumPy.

    Args:
        backend: ``"auto"`` (native if a library is found), ``"numpy"`` or ``"native"``
            (raises if no library is available).
        seed: seed for :meth:`batch_pca`.
    """

    def __init__(self, backend: Backend = "auto", seed: int = 0) -> None:
        self.seed = seed
        self._native = None if backend == "numpy" else native.find()
        if backend == "native" and self._native is None:
            raise RuntimeError("native backend requested but not built; run `vectorprep build-native`")
        self.has_cpp = self._native is not None
        log.debug("VectorPreprocessor backend=%s", "native" if self.has_cpp else "numpy")

    @property
    def backend(self) -> str:
        return "native" if self._use_native() else "numpy"

    def _use_native(self) -> bool:
        return bool(self.has_cpp and self._native is not None)

    @staticmethod
    def _f32(x: NDArray[np.floating], inplace: bool) -> NDArray[np.float32]:
        if inplace:
            if not (isinstance(x, np.ndarray) and x.dtype == np.float32 and x.flags.c_contiguous):
                raise ValueError("inplace=True needs a C-contiguous float32 array")
            return x
        return np.array(x, dtype=np.float32, order="C", copy=True)

    def scale_minmax(
        self, data: NDArray[np.floating], target_min: float = 0.0, target_max: float = 1.0, inplace: bool = False
    ) -> NDArray[np.float32]:
        """Affinely map ``data`` onto ``[target_min, target_max]`` (constant input → ``target_min``)."""
        x = self._f32(data, inplace)
        if x.size == 0:
            return x
        if self._use_native():
            assert self._native is not None
            self._native.scale_minmax_(x, target_min, target_max)
            return x
        out = kernels.scale_minmax(x, target_min, target_max)
        if inplace:
            x[...] = out
            return x
        return out

    def normalize_l2(self, data: NDArray[np.floating], inplace: bool = False) -> NDArray[np.float32]:
        """Scale every row of a 2-D matrix to unit L2 norm; all-zero rows stay zero."""
        if np.ndim(data) != 2:
            raise ValueError("Input data must be a 2D matrix.")
        x = self._f32(data, inplace)
        if self._use_native():
            assert self._native is not None
            self._native.normalize_l2_(x)
            return x
        out = kernels.normalize_l2(x)
        if inplace:
            x[...] = out
            return x
        return out

    def dot_product(self, a: NDArray[np.floating], b: NDArray[np.floating]) -> float:
        """Dot product accumulated in float64."""
        a32 = np.ascontiguousarray(a, dtype=np.float32).ravel()
        b32 = np.ascontiguousarray(b, dtype=np.float32).ravel()
        if a32.size != b32.size:
            raise ValueError("Vector sizes must match.")
        if self._use_native():
            assert self._native is not None
            return self._native.dot(a32, b32)
        return kernels.dot(a32, b32)

    def batch_pca(
        self, data: NDArray[np.floating], n_components: int
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Seeded subspace-iteration PCA. Returns ``(projected (N, k), components (k, D))``."""
        comps, _, proj = pca_power(data, n_components, seed=self.seed)
        return proj.astype(np.float32), comps.astype(np.float32)

    def quantize_embeddings(self, data: NDArray[np.floating]) -> tuple[NDArray[np.int8], float]:
        """Per-tensor symmetric int8 quantisation. Returns ``(q, scale)``; ``x ≈ q * scale``."""
        q, scale = quantize_int8(data)
        return q, float(scale)

    def batch_cosine_similarity(self, X: NDArray[np.floating], Y: NDArray[np.floating]) -> NDArray[np.float32]:
        """Pairwise cosine similarity matrix of shape ``(len(X), len(Y))``."""
        return np.asarray(self.normalize_l2(X) @ self.normalize_l2(Y).T)

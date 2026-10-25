"""v1 entry point, kept for compatibility: build the C++ backend if possible, then time it.

The full, reproducible benchmark is ``vectorprep bench`` (writes docs/assets/bench.json).
"""

from __future__ import annotations

import time

import numpy as np

from vectorprep import native
from vectorprep.core import VectorPreprocessor


def compile_cpp_library() -> bool:
    """Compile the C++ backend into the cache dir. Returns False if no compiler is available."""
    try:
        print(f"Built {native.build()}")
        return True
    except Exception as e:  # noqa: BLE001 - report any toolchain failure and fall back
        print(f"[WARNING] C++ build failed, using NumPy: {e}")
        return False


def run_performance_benchmarks(seed: int = 42) -> dict[str, float]:
    compile_cpp_library()
    prep = VectorPreprocessor()
    print(f"Backend: {prep.backend}")
    rng = np.random.default_rng(seed)
    data = rng.standard_normal(5_000_000).astype(np.float32)
    t0 = time.perf_counter()
    res = prep.scale_minmax(data, 0.0, 100.0)
    t_minmax = (time.perf_counter() - t0) * 1e3
    print(f"MinMax 5M floats: {t_minmax:.2f} ms (min {res.min():.2f}, max {res.max():.2f})")
    matrix = rng.standard_normal((10000, 200)).astype(np.float32)
    t0 = time.perf_counter()
    out = prep.normalize_l2(matrix)
    t_l2 = (time.perf_counter() - t0) * 1e3
    print(f"L2 10000x200: {t_l2:.2f} ms (row-0 norm {np.linalg.norm(out[0]):.4f})")
    return {"minmax_ms": t_minmax, "l2_ms": t_l2}


if __name__ == "__main__":
    run_performance_benchmarks()

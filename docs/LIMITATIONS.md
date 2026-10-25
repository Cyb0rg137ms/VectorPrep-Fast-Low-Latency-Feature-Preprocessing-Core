# Limitations

* The C++ kernels are single-threaded scalar loops compiled with `-O3`. There is no explicit
  SIMD or OpenMP, so the speedups over NumPy are modest (1.0–1.6× with copy, 1.5–3.6× in
  place, measured; see README).
* v2's NumPy L2 path accumulates norms in float64 for accuracy. It is **slower** than v1's
  `np.linalg.norm` path (0.75–0.95× of v1 throughput at N ≥ 1000; measured), which is a
  deliberate accuracy-for-speed trade.
* The ctypes call overhead (~µs) dominates for tiny inputs.
* Native builds are tested on Linux (g++). The macOS and Windows code paths exist but are untested in CI.
* PCA forms the D×D covariance, which suits D up to a few thousand. For very wide data, use
  randomized SVD on X directly.
* Quantisation is symmetric int8 only, with no zero-point, int4 or product quantisation.
  Per-tensor scaling degrades badly when row norms vary (measured 16.7 dB vs 42.3 dB per-row).
* Benchmarks use synthetic Gaussian data, not real embedding corpora.

## Roadmap
* OpenMP plus explicit AVX2 kernels, with a wheel build via scikit-build-core.
* A fused normalise-and-quantise kernel.
* Retrieval-recall benchmark on a public embedding set.

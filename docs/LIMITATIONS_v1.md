"""
LIMITATIONS.md — VectorPrep-Fast
================================

# Known Limitations and Future Work

## C++ / SIMD AVX2 Intrinsics

### 1. Lack of Precompiled Binaries on Windows
Although the C++ library in `src/preprocessor.cpp` is fully optimized, it requires compilation to a shared library (`preprocessor.dll` or `libpreprocessor.so`).
- Without a native compiler (like MSVC or GCC) on the target Windows system, the library falls back to the NumPy-based implementation.
- While the NumPy implementation is highly optimized, it does not exploit direct hand-crafted AVX2/FMA register loops that can be custom-tailored for cache-aligned layouts.

### 2. Thread Safety and Parallelism
The C++ normalization loop `normalize_l2` processes rows sequentially:
- For very large matrices ($N > 10^6$ vectors), it is bounded by single-core speed.
- In production, one should use **OpenMP** to parallelize the row loop across all available cores (`#pragma omp parallel for`).

---

## PCA & Quantization

### 1. Naive Power Iteration PCA
Our `batch_pca` uses a power iteration method with Gram-Schmidt orthogonalisation to find the top-$k$ principal components:
- This works well for small $D$.
- For extremely high-dimensional vectors ($D > 1000$ dimensions, e.g. text/image embeddings), computing the full covariance matrix ($D \times D$) takes $O(N D^2)$ time and $O(D^2)$ space, which is very expensive.
- In production, a randomized SVD algorithm (like Halko's method) or incremental PCA should be used to avoid storing the covariance matrix.

### 2. Simple Linear Scalar Quantization
`quantize_embeddings` uses a simple symmetric linear map to range $[-127, 127]$:
- This is susceptible to outliers (one very large activation value will squash all normal values, causing massive precision loss).
- To solve this, a production-grade quantizer should use **percentile-based clipping** (e.g. clipping the top 0.1% outliers before computing the scale factor) or k-means product quantization (PQ).
"""

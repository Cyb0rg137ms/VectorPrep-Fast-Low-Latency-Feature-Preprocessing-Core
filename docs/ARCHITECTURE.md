# Architecture (v2)

```mermaid
flowchart LR
  U[user code / CLI] --> VP[core.VectorPreprocessor<br/>backend = auto · numpy · native]
  VP -->|numpy| K[kernels.py<br/>float64-accumulated L2, min-max, dot]
  VP -->|native| N[native.py<br/>ctypes NativeLib, ABI check]
  N --> SO[(libpreprocessor.so<br/>csrc/preprocessor.cpp, -O3)]
  B[vectorprep build-native] --> SO
  VP --> Q[quantize.py<br/>int8 per-tensor / per-row, SQNR, cosine error]
  VP --> P[pca.py<br/>seeded subspace iteration + Rayleigh-Ritz, SVD reference]
  BEN[benchmarks.py] --> VP & Q & P
```

* **Copy by default.** Both backends return a new array unless `inplace=True`, so the
  native and NumPy paths have the same semantics. In v1 the C++ path silently overwrote
  the caller's float32 array and the NumPy path did not.
* **ABI version symbol.** `vectorprep_abi_version()` lets the loader reject stale v1
  builds, whose 32-bit `int` signatures would be called with 64-bit sizes.
* **Library lookup:** `$VECTORPREP_LIB` → `$VECTORPREP_CACHE` or `~/.cache/vectorprep/` → `./build/` (v1).

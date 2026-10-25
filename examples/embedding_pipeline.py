"""End-to-end: normalise embeddings, reduce with PCA, quantise to int8, check retrieval quality.

Run: python examples/embedding_pipeline.py
"""

import numpy as np

from vectorprep import VectorPreprocessor, dequantize, quantize_int8, sqnr_db

rng = np.random.default_rng(0)
docs = rng.standard_normal((2000, 256)).astype(np.float32)
queries = docs[:50] + 0.1 * rng.standard_normal((50, 256)).astype(np.float32)

prep = VectorPreprocessor(seed=0)
print("backend:", prep.backend)
d, q = prep.normalize_l2(docs), prep.normalize_l2(queries)

q8, scale = quantize_int8(d, per_row=True)
d_hat = dequantize(q8, scale)
print(f"int8 per-row SQNR: {sqnr_db(d, d_hat):.1f} dB, {d.nbytes / (q8.nbytes + scale.nbytes):.2f}x smaller")

top1_fp32 = (q @ d.T).argmax(1)
top1_int8 = (q @ d_hat.T).argmax(1)
print(f"top-1 agreement fp32 vs int8: {np.mean(top1_fp32 == top1_int8):.2%}")
print(f"top-1 correct (query i -> doc i): fp32 {np.mean(top1_fp32 == np.arange(50)):.2%}")

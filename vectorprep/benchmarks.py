"""Measured benchmarks (``vectorprep bench``): backend throughput, int8 error, PCA accuracy."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np

from vectorprep.core import VectorPreprocessor
from vectorprep.pca import explained_variance_ratio, pca_power, pca_svd
from vectorprep.quantize import cosine_error, dequantize, quantize_int8, sqnr_db

log = logging.getLogger(__name__)


def _median_ms(fn: Callable[[], object], repeats: int) -> float:
    fn()  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) * 1e3)


def _v1_numpy_l2(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.asarray(x / norms)


def throughput(shapes: list[tuple[int, int]], repeats: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    preps = {"numpy": VectorPreprocessor("numpy")}
    try:
        preps["native"] = VectorPreprocessor("native")
    except RuntimeError:
        log.warning("native backend not built; benchmarking numpy only")
    rows = []
    for n, d in shapes:
        x = rng.standard_normal((n, d)).astype(np.float32)
        row: dict[str, Any] = {"n": n, "d": d}
        row["v1_numpy_ms"] = _median_ms(partial(_v1_numpy_l2, x), repeats)
        for name, p in preps.items():
            row[f"{name}_ms"] = _median_ms(partial(p.normalize_l2, x), repeats)
            buf = x.copy()
            row[f"{name}_inplace_ms"] = _median_ms(partial(p.normalize_l2, buf, inplace=True), repeats)
        for k in list(row):
            if k.endswith("_ms"):
                row[k.replace("_ms", "_vecs_per_s")] = n / (row[k] / 1e3)
        rows.append(row)
    return rows


def quantization(n: int, d: int, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    gauss = rng.standard_normal((n, d)).astype(np.float32)
    outlier = gauss.copy()
    outlier[:, :4] *= 20.0  # a few high-magnitude channels, as seen in transformer activations
    rownorm = gauss * rng.lognormal(0.0, 1.0, (n, 1)).astype(np.float32)  # rows of very different norms
    out = []
    for name, x in [("gaussian", gauss), ("outlier_channels", outlier), ("varying_row_norms", rownorm)]:
        for per_row in (False, True):
            q, s = quantize_int8(x, per_row=per_row)
            xh = dequantize(q, s)
            out.append(
                {
                    "data": name,
                    "scheme": "per_row" if per_row else "per_tensor",
                    "sqnr_db": sqnr_db(x, xh),
                    "max_cosine_error": cosine_error(x, xh),
                    "bytes_ratio": (x.nbytes) / (q.nbytes + s.nbytes),
                }
            )
    return out


def pca_accuracy(n: int, d: int, ks: list[int], seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    spectrum = 1.0 / np.arange(1, d + 1) ** 1.0  # decaying variances
    basis, _ = np.linalg.qr(rng.standard_normal((d, d)))
    x = (rng.standard_normal((n, d)) * np.sqrt(spectrum)) @ basis.T
    rows = []
    for k in ks:
        comps, evals, _ = pca_power(x, k, seed=seed)
        ref, evr_ref = pca_svd(x, k)
        align = np.abs(np.sum(comps * ref, axis=1))
        rows.append(
            {
                "k": k,
                "explained_variance": float(explained_variance_ratio(x, evals).sum()),
                "explained_variance_svd": float(evr_ref.sum()),
                "min_axis_alignment": float(align.min()),
                "power_ms": _median_ms(partial(pca_power, x, k, seed=seed), 3),
                "svd_ms": _median_ms(partial(pca_svd, x, k), 3),
            }
        )
    return rows


def run(out_dir: Path | None = None, quick: bool = False) -> dict[str, Any]:
    shapes = [(1000, 128), (5000, 256)] if quick else [(1000, 128), (10000, 512), (100000, 768), (200000, 1536)]
    res = {
        "machine": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "cpus": os.cpu_count(),
        },
        "l2_throughput": throughput(shapes, repeats=3 if quick else 7),
        "quantization": quantization(2000 if quick else 20000, 768),
        "pca": pca_accuracy(2000 if quick else 5000, 100, [2, 5] if quick else [2, 5, 10, 20]),
    }
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "bench.json").write_text(json.dumps(res, indent=2))
        plot(res, out_dir / "benchmark.png")
    return res


def plot(res: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        log.warning("matplotlib not installed; skipping plot")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    rows = res["l2_throughput"]
    labels = [f"{r['n']:,}×{r['d']}" for r in rows]
    series = [
        ("v1_numpy", "v1 NumPy (linalg.norm)"),
        ("numpy", "v2 NumPy (copy)"),
        ("native", "C++ (copy)"),
        ("native_inplace", "C++ in place"),
    ]
    series = [s for s in series if f"{s[0]}_vecs_per_s" in rows[0]]
    w = 0.8 / len(series)
    for i, (key, lab) in enumerate(series):
        a1.bar(np.arange(len(rows)) + i * w, [r[f"{key}_vecs_per_s"] / 1e6 for r in rows], w, label=lab)
    a1.set_xticks(np.arange(len(rows)) + w * (len(series) - 1) / 2, labels)
    a1.set_yscale("log")
    a1.set(ylabel="million vectors / s (log)", xlabel="matrix (N × D)", title="Row L2 normalisation throughput")
    a1.legend(fontsize=8)
    q = res["quantization"]
    names = list(dict.fromkeys(r["data"] for r in q))
    for i, scheme in enumerate(["per_tensor", "per_row"]):
        vals = [next(r["sqnr_db"] for r in q if r["data"] == n and r["scheme"] == scheme) for n in names]
        a2.bar(np.arange(len(names)) + i * 0.4, vals, 0.4, label=scheme.replace("_", "-"))
    a2.set_xticks(np.arange(len(names)) + 0.2, [n.replace("_", "\n") for n in names])
    a2.set(ylabel="SQNR (dB, higher is better)", title="int8 quantisation error (D = 768)")
    a2.legend(fontsize=8)
    for ax in (a1, a2):
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)

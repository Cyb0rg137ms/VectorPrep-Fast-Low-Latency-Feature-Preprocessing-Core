"""``vectorprep`` CLI: build-native, info, bench, and file transforms on .npy arrays."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from vectorprep import native
from vectorprep.core import VectorPreprocessor


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="vectorprep", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-native", help="compile the C++ backend into the cache dir")
    b.add_argument("--out-dir", type=Path)
    sub.add_parser("info", help="show which backend is active")
    t = sub.add_parser("transform", help="apply an op to a .npy file")
    t.add_argument("op", choices=["l2", "minmax", "int8", "pca"])
    t.add_argument("input", type=Path)
    t.add_argument("output", type=Path)
    t.add_argument("--k", type=int, default=8, help="PCA components")
    t.add_argument("--backend", choices=["auto", "numpy", "native"], default="auto")
    t.add_argument("--seed", type=int, default=0)
    be = sub.add_parser("bench", help="run measured benchmarks")
    be.add_argument("--out", type=Path, default=Path("docs/assets"))
    be.add_argument("--quick", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING)

    if a.cmd == "build-native":
        print(native.build(a.out_dir))
    elif a.cmd == "info":
        lib = native.find()
        print(json.dumps({"backend": "native" if lib else "numpy", "library": str(lib.path) if lib else None}))
    elif a.cmd == "transform":
        p = VectorPreprocessor(a.backend, seed=a.seed)
        x = np.load(a.input)
        if a.op == "l2":
            np.save(a.output, p.normalize_l2(x))
        elif a.op == "minmax":
            np.save(a.output, p.scale_minmax(x))
        elif a.op == "int8":
            q, s = p.quantize_embeddings(x)
            np.savez(a.output, q=q, scale=np.float32(s))
        else:
            np.save(a.output, p.batch_pca(x, a.k)[0])
        print(f"{a.op} ({p.backend}) -> {a.output}")
    else:
        from vectorprep.benchmarks import run

        res = run(a.out, quick=a.quick)
        print(json.dumps({k: v for k, v in res.items() if k != "machine"}, indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

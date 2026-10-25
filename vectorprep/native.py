"""Optional C++ backend: build with ``vectorprep build-native`` and load via ctypes.

Lookup order for the shared library:
1. ``$VECTORPREP_LIB`` (explicit path),
2. ``$VECTORPREP_CACHE`` or ``~/.cache/vectorprep/`` (where ``build()`` writes),
3. ``<repo>/build/`` (v1 location, kept for compatibility).
"""

from __future__ import annotations

import ctypes
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)

ABI_VERSION = 2
SOURCE = Path(__file__).parent / "csrc" / "preprocessor.cpp"
_F32P = ctypes.POINTER(ctypes.c_float)


def lib_name() -> str:
    if sys.platform.startswith("win32"):
        return "preprocessor.dll"
    if sys.platform.startswith("darwin"):
        return "libpreprocessor.dylib"
    return "libpreprocessor.so"


def cache_dir() -> Path:
    return Path(os.environ.get("VECTORPREP_CACHE", Path.home() / ".cache" / "vectorprep"))


def candidate_paths() -> list[Path]:
    out = []
    if env := os.environ.get("VECTORPREP_LIB"):
        out.append(Path(env))
    out.append(cache_dir() / lib_name())
    out.append(Path(__file__).resolve().parent.parent / "build" / lib_name())
    return out


@dataclass
class NativeLib:
    """Typed wrapper around the C ABI. All functions mutate the passed buffer in place."""

    cdll: ctypes.CDLL
    path: Path

    @classmethod
    def load(cls, path: Path) -> NativeLib:
        lib = ctypes.CDLL(str(path))
        lib.vectorprep_abi_version.restype = ctypes.c_int32
        if lib.vectorprep_abi_version() != ABI_VERSION:
            raise OSError(f"{path}: ABI {lib.vectorprep_abi_version()} != {ABI_VERSION}; rebuild")
        lib.scale_minmax.argtypes = [_F32P, ctypes.c_int64, ctypes.c_float, ctypes.c_float]
        lib.scale_minmax.restype = None
        lib.normalize_l2.argtypes = [_F32P, ctypes.c_int64, ctypes.c_int64]
        lib.normalize_l2.restype = None
        lib.vector_dot_product.argtypes = [_F32P, _F32P, ctypes.c_int64]
        lib.vector_dot_product.restype = ctypes.c_double
        return cls(lib, path)

    def scale_minmax_(self, x: NDArray[np.float32], lo: float, hi: float) -> None:
        self.cdll.scale_minmax(x.ctypes.data_as(_F32P), x.size, lo, hi)

    def normalize_l2_(self, x: NDArray[np.float32]) -> None:
        self.cdll.normalize_l2(x.ctypes.data_as(_F32P), x.shape[0], x.shape[1])

    def dot(self, a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
        return float(self.cdll.vector_dot_product(a.ctypes.data_as(_F32P), b.ctypes.data_as(_F32P), a.size))


def find() -> NativeLib | None:
    """Load the first valid library from :func:`candidate_paths`, or ``None``."""
    for p in candidate_paths():
        if p.exists():
            try:
                return NativeLib.load(p)
            except (OSError, AttributeError) as e:  # stale v1 build has no ABI symbol
                log.warning("ignoring %s: %s", p, e)
    return None


def build(out_dir: Path | None = None, compiler: str | None = None) -> Path:
    """Compile :data:`SOURCE` with ``g++``/``clang++`` (-O3, -fPIC). Returns the library path."""
    cxx = compiler or os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        raise RuntimeError("no C++ compiler found (set CXX or install g++)")
    out_dir = out_dir or cache_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / lib_name()
    cmd = [cxx, "-O3", "-shared", "-fPIC", "-std=c++17", "-o", str(out), str(SOURCE)]
    log.info("building native backend: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return out

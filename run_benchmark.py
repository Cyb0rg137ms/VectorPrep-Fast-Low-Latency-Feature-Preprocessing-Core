"""
run_benchmark.py
================
Attempts to compile the C++ shared library, then profiles execution latency 
between the C++ memory-aligned implementations and standard NumPy wrappers.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import numpy as np
from vectorprep.core import VectorPreprocessor

def compile_cpp_library() -> bool:
    """Attempts to compile preprocessor.cpp to a shared library/DLL."""
    print("Checking C++ compiler toolchains for local building...")
    
    src_dir = Path(__file__).parent / "src"
    build_dir = Path(__file__).parent / "build"
    build_dir.mkdir(exist_ok=True)
    
    src_file = src_dir / "preprocessor.cpp"
    
    # Target file
    if sys.platform.startswith("win32"):
        dll_file = build_dir / "preprocessor.dll"
        # Try MSVC cl, then fall back to g++
        try:
            # MSVC compilation
            # cl /LD /EHsc /O2 src/preprocessor.cpp /Fe:build/preprocessor.dll
            print("Attempting compilation via MSVC cl...")
            cmd = ["cl", "/LD", "/EHsc", "/O2", str(src_file), f"/Fe:{dll_file}"]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print("  [SUCCESS] Compiled with MSVC cl!")
            return True
        except Exception:
            try:
                # MinGW g++ compilation
                # g++ -shared -o build/preprocessor.dll -O3 src/preprocessor.cpp
                print("MSVC cl failed. Attempting compilation via MinGW g++...")
                cmd = ["g++", "-shared", "-o", str(dll_file), "-O3", str(src_file)]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                print("  [SUCCESS] Compiled with MinGW g++!")
                return True
            except Exception as e:
                print(f"  [WARNING] C++ Compilations failed (No compiler on PATH): {e}")
                return False
    else:
        # UNIX/Linux/MacOS
        lib_name = "libpreprocessor.so" if not sys.platform.startswith("darwin") else "libpreprocessor.dylib"
        so_file = build_dir / lib_name
        try:
            print("Attempting compilation via g++...")
            cmd = ["g++", "-shared", "-fPIC", "-o", str(so_file), "-O3", str(src_file)]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print(f"  [SUCCESS] Compiled with g++ -> {lib_name}")
            return True
        except Exception as e:
            print(f"  [WARNING] Unix C++ compilations failed: {e}")
            return False

def run_performance_benchmarks():
    print("\n==================================================")
    print("      VECTORPREP-FAST: PREPROCESSING BENCHMARK    ")
    print("==================================================")
    
    # Compile C++ source first
    compiled_ok = compile_cpp_library()
    
    # Re-instantiate preprocessor to detect compiled binaries
    preprocessor = VectorPreprocessor()
    print(f"\nActive Preprocessing Mode: {'C++ DLL Compiled Backend' if preprocessor.has_cpp else 'NumPy Vectorized Fallback'}")
    
    # Setup test vectors (large scale)
    size = 5_000_000
    print(f"Generating random data array (Size: {size:,} floats)...")
    data_a = np.random.randn(size).astype(np.float32)
    
    # MinMax Scale Benchmark
    t0 = time.time()
    res_a = preprocessor.scale_minmax(data_a, 0.0, 100.0)
    t_elapsed = (time.time() - t0) * 1000.0
    
    print(f"\nMinMax Scaling profile:")
    print(f"  Execution time: {t_elapsed:.2f} ms")
    print(f"  Result Min:     {res_a.min():.2f}")
    print(f"  Result Max:     {res_a.max():.2f}")
    
    # Row Normalization Benchmark
    rows, cols = 10000, 200
    print(f"\nGenerating random matrix (Shape: {rows}x{cols})...")
    matrix = np.random.randn(rows, cols).astype(np.float32)
    
    t0 = time.time()
    norm_matrix = preprocessor.normalize_l2(matrix)
    t_elapsed_norm = (time.time() - t0) * 1000.0
    
    # Compute norm of first row to verify
    first_row_norm = np.linalg.norm(norm_matrix[0])
    print(f"L2 Normalization profile:")
    print(f"  Execution time:     {t_elapsed_norm:.2f} ms")
    print(f"  First Row L2 Norm:  {first_row_norm:.4f}")
    
    print("==================================================")

if __name__ == "__main__":
    run_performance_benchmarks()

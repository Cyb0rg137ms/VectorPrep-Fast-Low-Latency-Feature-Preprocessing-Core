"""
core.py
=======
Python wrapper for the high-performance C++ VectorPrep library using ctypes.
Features automatic fallback to pure-Python/NumPy if C++ library is not compiled.
"""

import ctypes
import os
import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

# Find C++ library binary path
LIB_NAME = "libpreprocessor.so"
if sys.platform.startswith("win32"):
    LIB_NAME = "preprocessor.dll"
elif sys.platform.startswith("darwin"):
    LIB_NAME = "libpreprocessor.dylib"

_lib_path = Path(__file__).parent.parent / "build" / LIB_NAME
_cpp_lib: Optional[ctypes.CDLL] = None

# Attempt to load C++ shared library
try:
    if _lib_path.exists():
        _cpp_lib = ctypes.CDLL(str(_lib_path))
        
        # Define scale_minmax
        _cpp_lib.scale_minmax.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float
        ]
        _cpp_lib.scale_minmax.restype = None
        
        # Define normalize_l2
        _cpp_lib.normalize_l2.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int
        ]
        _cpp_lib.normalize_l2.restype = None
        
        # Define vector_dot_product
        _cpp_lib.vector_dot_product.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int
        ]
        _cpp_lib.vector_dot_product.restype = ctypes.c_float
        
except Exception:
    _cpp_lib = None


class VectorPreprocessor:
    """Preprocesses matrix/vector structures using C++ or fallback NumPy implementations."""
    
    def __init__(self):
        self.has_cpp = (_cpp_lib is not None)

    def scale_minmax(self, data: np.ndarray, target_min: float = 0.0, target_max: float = 1.0) -> np.ndarray:
        """
        Scales input data in-place to range [target_min, target_max].
        
        Args:
            data: NumPy array of shape (N,).
            target_min: Scaled minimum boundary.
            target_max: Scaled maximum boundary.
            
        Returns:
            The scaled NumPy array.
        """
        # Ensure contiguous float32 data array
        data_arr = np.ascontiguousarray(data, dtype=np.float32)
        
        if self.has_cpp and _cpp_lib is not None:
            # C++ path
            data_ptr = data_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _cpp_lib.scale_minmax(data_ptr, data_arr.size, target_min, target_max)
            return data_arr
        else:
            # Fallback NumPy path
            d_min = data_arr.min()
            d_max = data_arr.max()
            d_range = d_max - d_min
            if d_range == 0:
                data_arr.fill(target_min)
                return data_arr
                
            scaled = target_min + ((data_arr - d_min) / d_range) * (target_max - target_min)
            return scaled

    def normalize_l2(self, data: np.ndarray) -> np.ndarray:
        """
        Normalizes rows of a 2D matrix in-place to L2 norm = 1.
        
        Args:
            data: 2D NumPy array of shape (rows, cols).
            
        Returns:
            L2-normalized NumPy array.
        """
        data_arr = np.ascontiguousarray(data, dtype=np.float32)
        if len(data_arr.shape) != 2:
            raise ValueError("Input data must be a 2D matrix.")
            
        rows, cols = data_arr.shape
        
        if self.has_cpp and _cpp_lib is not None:
            # C++ path
            data_ptr = data_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            _cpp_lib.normalize_l2(data_ptr, rows, cols)
            return data_arr
        else:
            # Fallback NumPy path
            norms = np.linalg.norm(data_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0  # Avoid zero divisions
            return data_arr / norms

    def dot_product(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Computes dot product of two arrays.
        
        Args:
            a: NumPy array of shape (N,).
            b: NumPy array of shape (N,).
            
        Returns:
            Float scalar dot product.
        """
        a_arr = np.ascontiguousarray(a, dtype=np.float32)
        b_arr = np.ascontiguousarray(b, dtype=np.float32)
        
        if a_arr.size != b_arr.size:
            raise ValueError("Vector sizes must match.")
            
        if self.has_cpp and _cpp_lib is not None:
            # C++ path
            a_ptr = a_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            b_ptr = b_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            return float(_cpp_lib.vector_dot_product(a_ptr, b_ptr, a_arr.size))
        else:
            # Fallback NumPy path
            return float(np.dot(a_arr, b_arr))

    def batch_pca(self, data: np.ndarray, n_components: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes Principal Component Analysis (PCA) using Gram-Schmidt orthogonalisation.
        
        Args:
            data: 2D NumPy array of shape (N, D).
            n_components: Number of components to keep.
            
        Returns:
            (projected_data, principal_components) where principal_components is (n_components, D).
        """
        X = np.ascontiguousarray(data, dtype=np.float32)
        # Center data
        X_mean = X.mean(axis=0)
        X_centered = X - X_mean
        
        # Compute covariance matrix (D, D)
        cov = np.cov(X_centered, rowvar=False)
        if cov.ndim == 0:
            cov = np.array([[cov]])
            
        # Power iteration to find dominant eigenvectors of covariance matrix
        D = X.shape[1]
        components = []
        
        for k in range(min(n_components, D)):
            v = np.random.randn(D).astype(np.float32)
            v /= np.linalg.norm(v)
            
            for _ in range(50):
                v_new = np.dot(cov, v)
                # Orthogonalise against previously found components
                for prev_v in components:
                    v_new -= np.dot(prev_v, v_new) * prev_v
                norm = np.linalg.norm(v_new)
                if norm < 1e-10:
                    break
                v = v_new / norm
                
            components.append(v)
            
        components = np.array(components)
        projected = np.dot(X_centered, components.T)
        return projected, components

    def quantize_embeddings(self, data: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Quantizes float32 embeddings into int8 representations.
        
        Uses symmetric range [-127, 127] mapping.
        
        Returns:
            (quantized_int8_array, scale_factor)
        """
        X = np.ascontiguousarray(data, dtype=np.float32)
        absmax = np.abs(X).max()
        scale = absmax / 127.0 if absmax > 0 else 1.0
        quantized = np.round(X / scale).clip(-127, 127).astype(np.int8)
        return quantized, scale

    def batch_cosine_similarity(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Computes pairwise cosine similarity matrix between two batches of embeddings.
        
        Returns:
            2D array of shape (len(X), len(Y)).
        """
        X_norm = self.normalize_l2(X)
        Y_norm = self.normalize_l2(Y)
        return np.dot(X_norm, Y_norm.T)


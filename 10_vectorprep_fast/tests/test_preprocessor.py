import pytest
import numpy as np
from vectorprep.core import VectorPreprocessor

def test_minmax_scaling():
    prep = VectorPreprocessor()
    data = np.array([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float32)
    
    # Scale to [0, 1]
    scaled = prep.scale_minmax(data, 0.0, 1.0)
    assert np.allclose(scaled, [0.0, 0.25, 0.5, 0.75, 1.0])

    # Test edge case: constant array
    constant_data = np.array([5.0, 5.0, 5.0], dtype=np.float32)
    scaled_const = prep.scale_minmax(constant_data, 1.0, 10.0)
    assert np.allclose(scaled_const, [1.0, 1.0, 1.0])

def test_l2_normalization():
    prep = VectorPreprocessor()
    matrix = np.array([
        [3.0, 4.0],
        [6.0, 8.0]
    ], dtype=np.float32)
    
    normalized = prep.normalize_l2(matrix)
    
    # Rows should have L2 norm = 1
    # First row: sqrt(3^2 + 4^2) = 5. Normalized = [3/5, 4/5] = [0.6, 0.8]
    assert np.allclose(normalized[0], [0.6, 0.8])
    assert np.allclose(normalized[1], [0.6, 0.8])
    assert np.allclose(np.linalg.norm(normalized, axis=1), [1.0, 1.0])

def test_dot_product():
    prep = VectorPreprocessor()
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([4.0, 5.0, 6.0], dtype=np.float32)
    
    dot = prep.dot_product(a, b)
    # 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert abs(dot - 32.0) < 1e-5

def test_numpy_fallback_toggle():
    prep = VectorPreprocessor()
    # Force NumPy fallback pathway
    prep.has_cpp = False
    
    data = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    scaled = prep.scale_minmax(data, 0.0, 100.0)
    assert np.allclose(scaled, [0.0, 50.0, 100.0])

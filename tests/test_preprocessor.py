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

def test_batch_pca():
    prep = VectorPreprocessor()
    np.random.seed(42)
    # Generate some correlated 2D data
    x = np.random.randn(50)
    y = x * 2.0 + np.random.randn(50) * 0.1
    data = np.column_stack([x, y])
    
    projected, components = prep.batch_pca(data, n_components=1)
    assert projected.shape == (50, 1)
    assert components.shape == (1, 2)
    # The principal component should point roughly in the direction of the line y = 2x
    # (i.e. first component has ratio ~2/sqrt(5) and 1/sqrt(5))
    ratio = abs(components[0][1] / components[0][0])
    assert 1.5 < ratio < 2.5

def test_quantize_embeddings():
    prep = VectorPreprocessor()
    data = np.array([-10.0, 0.0, 10.0, 5.0], dtype=np.float32)
    quantized, scale = prep.quantize_embeddings(data)
    
    assert quantized.dtype == np.int8
    assert quantized[0] == -127
    assert quantized[2] == 127
    assert abs(scale - (10.0 / 127.0)) < 1e-6

def test_batch_cosine_similarity():
    prep = VectorPreprocessor()
    X = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0]
    ], dtype=np.float32)
    Y = np.array([
        [1.0, 0.0],
        [0.0, 1.0]
    ], dtype=np.float32)
    
    sims = prep.batch_cosine_similarity(X, Y)
    assert sims.shape == (3, 2)
    assert abs(sims[0, 0] - 1.0) < 1e-6  # Identical
    assert abs(sims[0, 1] - 0.0) < 1e-6  # Orthogonal
    assert abs(sims[2, 0] - 1.0 / np.sqrt(2)) < 1e-6  # 45 degrees


"""VectorPrep: embedding preprocessing (min-max, L2, int8 quantisation, PCA) in NumPy with an optional C++ backend."""

from vectorprep.core import VectorPreprocessor
from vectorprep.pca import explained_variance_ratio, pca_power, pca_svd
from vectorprep.quantize import cosine_error, dequantize, quantize_int8, sqnr_db

__version__ = "2.0.0"
__all__ = [
    "VectorPreprocessor",
    "__version__",
    "cosine_error",
    "dequantize",
    "explained_variance_ratio",
    "pca_power",
    "pca_svd",
    "quantize_int8",
    "sqnr_db",
]

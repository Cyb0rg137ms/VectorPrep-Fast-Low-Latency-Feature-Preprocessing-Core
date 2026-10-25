"""PCA by seeded block power iteration, with an exact SVD reference."""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def pca_power(
    x: NDArray[np.floating], k: int, seed: int = 0, max_iter: int = 500, tol: float = 1e-7
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Top-``k`` principal axes via orthogonal (subspace) iteration on the covariance.

    Returns ``(components (k, D), eigenvalues (k,), projected (N, k))``. Converges when the
    subspace stops rotating (``tol``); logs a warning if ``max_iter`` is reached.
    """
    x = np.asarray(x, np.float64)
    xc = x - x.mean(axis=0)
    d = x.shape[1]
    k = min(k, d)
    cov = xc.T @ xc / max(x.shape[0] - 1, 1)
    q, _ = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, k)))
    for it in range(max_iter):
        q_new, _ = np.linalg.qr(cov @ q)
        # subspace distance: 1 - smallest singular value of q^T q_new
        if 1.0 - np.linalg.svd(q.T @ q_new, compute_uv=False).min() < tol:
            q = q_new
            log.debug("pca_power converged in %d iterations", it + 1)
            break
        q = q_new
    else:
        log.warning("pca_power: no convergence after %d iterations", max_iter)
    # Rayleigh-Ritz to order and rotate within the subspace
    evals, v = np.linalg.eigh(q.T @ cov @ q)
    order = np.argsort(evals)[::-1]
    comps = (q @ v[:, order]).T
    comps *= np.sign(comps[np.arange(k), np.abs(comps).argmax(axis=1)])[:, None]
    return comps, evals[order], xc @ comps.T


def pca_svd(x: NDArray[np.floating], k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact reference: ``(components (k, D), explained variance ratio (k,))``."""
    xc = np.asarray(x, np.float64) - np.mean(x, axis=0)
    _, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s**2
    comps = vt[:k]
    comps *= np.sign(comps[np.arange(len(comps)), np.abs(comps).argmax(axis=1)])[:, None]
    return comps, var[:k] / var.sum()


def explained_variance_ratio(x: NDArray[np.floating], eigenvalues: NDArray[np.float64]) -> NDArray[np.float64]:
    xc = np.asarray(x, np.float64) - np.mean(x, axis=0)
    total = float((xc * xc).sum() / max(xc.shape[0] - 1, 1))
    return np.asarray(eigenvalues / total)

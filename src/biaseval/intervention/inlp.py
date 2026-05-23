"""Linear concept erasure: INLP (Ravfogel et al. 2020) and LEACE (Belrose et
al. 2023).

Both fit a projection P that removes a linear attribute from activations.
INLP iterates: train probe, project onto its nullspace, repeat until probe
hits chance. LEACE is the closed-form least-squares projection that nullifies
the cross-covariance between activations and labels.

Projections are stored in row-vector convention (x' = x @ P) so the forward
hook in `hooks.ProjectionHook` can apply either method without branching.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class NullspaceResult:
    projection: np.ndarray
    n_iterations: int
    accuracy_curve: list[float]
    converged: bool
    method: str = "inlp"


def _train_probe_get_w(X: np.ndarray, y: np.ndarray, *, seed: int) -> tuple[np.ndarray, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
    acc = float(cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1).mean())
    clf.fit(X, y)
    w = clf.coef_.reshape(-1).astype(np.float32)
    return w, acc


def _projection_for(w: np.ndarray) -> np.ndarray:
    """P = I - w w^T / ||w||^2 -- orthogonal projection onto null(w)."""
    norm_sq = float(w @ w)
    if norm_sq < 1e-12:
        return np.eye(w.shape[0], dtype=np.float32)
    outer = np.outer(w, w).astype(np.float32) / norm_sq
    return np.eye(w.shape[0], dtype=np.float32) - outer


def fit_inlp(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 10,
    chance_threshold: float = 0.55,
    seed: int = 42,
) -> NullspaceResult:
    """Iterative Nullspace Projection.

    Stops when 5-fold CV probe accuracy on the projected X drops below
    `chance_threshold`, or after `max_iter`.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    H = X.shape[1]
    P = np.eye(H, dtype=np.float32)
    X_curr = X.copy()
    curve: list[float] = []
    converged = False

    for it in range(max_iter):
        w, acc = _train_probe_get_w(X_curr, y, seed=seed + it)
        curve.append(acc)
        if acc <= chance_threshold:
            converged = True
            logger.info("INLP converged at iter %d (acc=%.3f <= %.3f)", it, acc, chance_threshold)
            break
        P_i = _projection_for(w)
        X_curr = X_curr @ P_i
        P = P @ P_i

    if not converged:
        logger.warning("INLP did not converge after %d iters; last acc=%.3f", max_iter, curve[-1])

    return NullspaceResult(
        projection=P.astype(np.float32),
        n_iterations=len(curve),
        accuracy_curve=curve,
        converged=converged,
        method="inlp",
    )


@dataclass
class LeaceResult:
    projection: np.ndarray
    bias: np.ndarray
    method: str = "leace"


def fit_leace(X: np.ndarray, y: np.ndarray) -> LeaceResult:
    """LEACE -- closed-form least-squares concept erasure.

    Computes P that nullifies Cov(X, Y) with minimum mean-squared
    perturbation. Returns (P, mean) so the application is
        x' = (x - mean) @ P + mean
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y).reshape(-1)
    H = X.shape[1]

    classes = np.unique(y)
    Y = np.zeros((X.shape[0], len(classes)), dtype=np.float64)
    for i, c in enumerate(classes):
        Y[y == c, i] = 1.0

    mean_x = X.mean(axis=0)
    Xc = X - mean_x
    Yc = Y - Y.mean(axis=0)

    # Whiten with a small ridge for stability.
    cov_x = (Xc.T @ Xc) / max(X.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov_x + 1e-6 * np.eye(H))
    eigvals = np.clip(eigvals, 1e-10, None)
    W = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    W_inv = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

    cov_xy = (Xc.T @ Yc) / max(X.shape[0] - 1, 1)
    M = W @ cov_xy

    U, s, _ = np.linalg.svd(M, full_matrices=False)
    rank = int((s > 1e-8).sum())
    U = U[:, :rank]

    P_whitened = np.eye(H) - U @ U.T
    P = W @ P_whitened @ W_inv
    return LeaceResult(
        projection=P.astype(np.float32),
        bias=mean_x.astype(np.float32),
        method="leace",
    )


def standardise_for_probe(X: np.ndarray) -> np.ndarray:
    """Match the StandardScaler used in `linear_probe.train_layer_probe`."""
    return StandardScaler().fit_transform(X)

"""Train logistic-regression probes per layer."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def train_layer_probe(
    X: np.ndarray, y: np.ndarray, *, cv_folds: int = 5, seed: int = 42
) -> dict[str, float]:
    """Stratified k-fold CV accuracy on a logistic-regression probe."""
    if len(np.unique(y)) < 2:
        return {"mean_accuracy": float("nan"), "std_accuracy": float("nan")}

    X = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    return {
        "mean_accuracy": float(scores.mean()),
        "std_accuracy": float(scores.std()),
    }


def mean_difference_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Unit vector from class 0 centroid to class 1 centroid.

    Useful as a steering / ablation axis (Sun et al. 2025); saved per layer
    alongside the probe so downstream code can read off the direction.
    """
    if len(np.unique(y)) < 2:
        return np.zeros(X.shape[1], dtype=np.float32)
    mu1 = X[y == 1].mean(axis=0)
    mu0 = X[y == 0].mean(axis=0)
    diff = (mu1 - mu0).astype(np.float32)
    nrm = float(np.linalg.norm(diff))
    return diff / nrm if nrm > 0 else diff


def train_probes_all_layers(
    activation_dir: str | Path,
    labels: np.ndarray,
    num_layers: int,
    attribute_name: str,
    *,
    cv_folds: int = 5,
    seed: int = 42,
    save_directions: bool = True,
    direction_save_dir: str | Path | None = None,
) -> list[dict]:
    """Train a per-layer probe and save the mean-difference direction.

    Reads `activation_dir/layer_<i>.npy` for i in [0, num_layers). When
    `activation_dir` is a per-attribute slice, pass `direction_save_dir` to
    write `direction_<attribute>.npy` somewhere the aggregator can find it.
    """
    activation_dir = Path(activation_dir)
    save_dir = Path(direction_save_dir) if direction_save_dir is not None else activation_dir
    results: list[dict] = []
    if save_directions:
        directions = np.zeros((num_layers, np.load(activation_dir / "layer_0.npy").shape[1]),
                              dtype=np.float32)
    for layer_idx in range(num_layers):
        X = np.load(activation_dir / f"layer_{layer_idx}.npy")
        scores = train_layer_probe(X, labels, cv_folds=cv_folds, seed=seed)
        results.append(
            {
                "layer": layer_idx,
                "layer_normalized": layer_idx / max(num_layers - 1, 1),
                "attribute": attribute_name,
                **scores,
            }
        )
        if save_directions:
            directions[layer_idx] = mean_difference_direction(X, labels)
    if save_directions:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / f"direction_{attribute_name}.npy", directions)
    return results

from __future__ import annotations

import numpy as np
import pandas as pd


def make_binary_target(y_weak: np.ndarray, positive_threshold: float = 1.0) -> np.ndarray:
    """Map weak labels to binary classes; by default only label 1.0 is positive."""
    return (y_weak >= positive_threshold).astype(int)


def class_counts(y: np.ndarray) -> dict[str, int]:
    counts = np.bincount(y.astype(int), minlength=2)
    return {"class_0": int(counts[0]), "class_1": int(counts[1])}


def make_balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Balanced weights for weighted binary cross-entropy/log-loss."""
    counts = np.bincount(y.astype(int), minlength=2)
    if np.any(counts == 0):
        raise RuntimeError(
            "Training split produced a single class. Increase --n-rows or adjust --test-size."
        )
    total = counts.sum()
    class_weight = total / (2.0 * counts)
    return class_weight[y.astype(int)]


def undersample_majority_class(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Return a balanced training subset by downsampling the majority class."""
    rng = np.random.default_rng(seed)
    class_0_idx = np.flatnonzero(y == 0)
    class_1_idx = np.flatnonzero(y == 1)
    target_count = min(len(class_0_idx), len(class_1_idx))
    if target_count == 0:
        raise RuntimeError("Cannot undersample a single-class training split.")

    selected_0 = rng.choice(class_0_idx, size=target_count, replace=False)
    selected_1 = rng.choice(class_1_idx, size=target_count, replace=False)
    selected = np.sort(np.concatenate([selected_0, selected_1]))
    return X.iloc[selected].reset_index(drop=True), y[selected]

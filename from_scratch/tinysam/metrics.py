"""Segmentation metrics implemented with NumPy."""

import numpy as np


def intersection_over_union(prediction: np.ndarray, target: np.ndarray) -> float:
    """Return binary IoU; an empty union is treated as a perfect match."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    union = np.logical_or(prediction, target).sum()
    return float(np.logical_and(prediction, target).sum() / union) if union else 1.0


def mean_iou(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Average per-image binary IoU."""
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length")
    if not len(predictions):
        return 0.0
    return float(np.mean([
        intersection_over_union(prediction, target)
        for prediction, target in zip(predictions, targets)
    ]))

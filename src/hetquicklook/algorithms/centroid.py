"""Spatial centroid operations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Centroid:
    """A weighted centroid and the signal supporting it."""

    x: float
    y: float
    total_weight: float
    n_used: int


def weighted_centroid(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Centroid:
    """Calculate a centroid from finite, non-negative weights.

    Returns ``NaN`` coordinates when no positive finite signal is available.
    Negative weights are ignored because they are not meaningful for the
    source-location diagnostic.
    """

    x_values, y_values, weight_values = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(weights, dtype=float)
    )
    valid = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(weight_values)
    valid &= weight_values > 0
    if not np.any(valid):
        return Centroid(float("nan"), float("nan"), 0.0, 0)
    x_valid = x_values[valid]
    y_valid = y_values[valid]
    weights_valid = weight_values[valid]
    total = float(np.sum(weights_valid))
    return Centroid(
        x=float(np.sum(x_valid * weights_valid) / total),
        y=float(np.sum(y_valid * weights_valid) / total),
        total_weight=total,
        n_used=int(weights_valid.size),
    )


def centroid_2d(image: np.ndarray) -> Centroid:
    """Calculate a centroid for a two-dimensional image array."""

    values = np.asarray(image, dtype=float)
    if values.ndim != 2:
        raise ValueError("image must be a two-dimensional array")
    y, x = np.indices(values.shape, dtype=float)
    return weighted_centroid(x, y, values)


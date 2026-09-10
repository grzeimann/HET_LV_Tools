"""Collapse detector or extracted fiber data into quick-look products."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..fibers import FiberTopology
from .extraction import extract_trace


def collapse_fiber_signal(signal: np.ndarray, *, statistic: str = "mean") -> float:
    """Collapse one extracted fiber signal to a scalar.

    ``NaN`` and infinite values are ignored. If no finite values remain,
    ``NaN`` is returned.
    """

    values = np.asarray(signal, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    if statistic == "mean":
        return float(np.mean(finite))
    if statistic == "median":
        return float(np.median(finite))
    if statistic == "sum":
        return float(np.sum(finite))
    raise ValueError("statistic must be 'mean', 'median', or 'sum'")


def collapse_fibers(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    aperture: int = 1,
    statistic: str = "mean",
) -> dict[str, float]:
    """Extract and collapse every fiber in an amplifier topology."""

    return {
        fiber_id: collapse_fiber_signal(
            extract_trace(detector, topology.traces[fiber_id], aperture=aperture),
            statistic=statistic,
        )
        for fiber_id in topology.fiber_ids
    }


def spatial_image_from_fiber_values(
    fiber_values: Mapping[str, float],
    topology: FiberTopology,
    *,
    fill_value: float = float("nan"),
) -> np.ndarray:
    """Place fiber values on the integer IFU-plane grid.

    Fiber positions are rounded to the nearest grid pixel. If multiple fibers
    map to one pixel, their finite values are averaged. The returned array is
    indexed as ``[ifu_y - y_min, ifu_x - x_min]``.
    """

    if not topology.locations:
        return np.empty((0, 0), dtype=float)
    x_positions = np.array([location.ifu_x for location in topology.locations.values()])
    y_positions = np.array([location.ifu_y for location in topology.locations.values()])
    x_indices = np.rint(x_positions).astype(int)
    y_indices = np.rint(y_positions).astype(int)
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())

    image = np.full((y_max - y_min + 1, x_max - x_min + 1), fill_value, dtype=float)
    buckets: dict[tuple[int, int], list[float]] = {}
    for fiber_id, location in topology.locations.items():
        value = float(fiber_values.get(fiber_id, fill_value))
        if np.isfinite(value):
            key = (int(np.rint(location.ifu_y)) - y_min, int(np.rint(location.ifu_x)) - x_min)
            buckets.setdefault(key, []).append(value)
    for key, values in buckets.items():
        image[key] = float(np.mean(values))
    return image


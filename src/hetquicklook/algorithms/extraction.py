"""Detector-to-trace extraction operations."""

from __future__ import annotations

import numpy as np

from ..fibers import FiberTrace


def extract_trace(
    detector: np.ndarray,
    trace: FiberTrace,
    *,
    aperture: int = 1,
) -> np.ndarray:
    """Sample detector data along a fiber trace.

    Args:
        detector: Two-dimensional detector image indexed as ``[y, x]``.
        trace: Detector coordinates for one fiber.
        aperture: Odd positive number of pixels sampled across the trace. The
            samples are averaged with non-finite values ignored.

    Returns:
        One-dimensional array with one value per trace sample. A sample is
        ``NaN`` when no finite detector pixels fall in its aperture.
    """

    image = np.asarray(detector, dtype=float)
    if image.ndim != 2:
        raise ValueError("detector must be a two-dimensional array")
    if aperture < 1 or aperture % 2 == 0:
        raise ValueError("aperture must be a positive odd integer")

    result = np.full(trace.detector_x.shape, np.nan, dtype=float)
    radius = aperture // 2
    height, width = image.shape
    for index, (x_value, y_value) in enumerate(
        zip(trace.detector_x, trace.detector_y)
    ):
        x_center = int(np.rint(x_value))
        y_center = int(np.rint(y_value))
        x_low = max(0, x_center - radius)
        x_high = min(width, x_center + radius + 1)
        y_low = max(0, y_center - radius)
        y_high = min(height, y_center + radius + 1)
        if x_low >= x_high or y_low >= y_high:
            continue
        values = image[y_low:y_high, x_low:x_high]
        finite = values[np.isfinite(values)]
        if finite.size:
            result[index] = float(np.mean(finite))
    return result


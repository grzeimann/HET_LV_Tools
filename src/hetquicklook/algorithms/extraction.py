"""Shared fractional-aperture extraction for VIRUS and LRS2."""

from __future__ import annotations

import numpy as np

from ..fibers import FiberTrace
from .results import AlgorithmResult


EXTRACTION_VERSION = "fractional-sum-aperture-1.0"


def fractional_aperture_geometry(
    traces: np.ndarray,
    detector_rows: int,
    *,
    width: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact detector-row overlaps for continuous top-hat apertures.

    Args:
        traces: Dense trace map with shape ``(nfiber, ncolumn)``.
        detector_rows: Number of detector rows available for sampling.
        width: Continuous aperture width in detector pixels.

    Returns:
        ``(rows, weights, valid)``.  ``rows`` and ``weights`` have shape
        ``(nfiber, ncolumn, ceil(width)+1)`` and ``valid`` has shape
        ``(nfiber, ncolumn)``.  A valid aperture is wholly contained in the
        detector; partial edge apertures are retained as invalid evidence,
        following the supplied VIRUSFlow contract.
    """

    trace = np.asarray(traces, dtype=float)
    if (
        trace.ndim != 2
        or int(detector_rows) <= 0
        or not np.isfinite(width)
        or float(width) <= 0.0
    ):
        raise ValueError(
            "fractional aperture requires 2D traces, positive rows, and positive width"
        )
    aperture_width = float(width)
    sample_count = int(np.ceil(aperture_width)) + 1
    left = trace - aperture_width / 2.0
    right = trace + aperture_width / 2.0
    safe_left = np.where(np.isfinite(left), left, 0.0)
    start = np.floor(safe_left).astype(np.int32)
    rows = start[..., None] + np.arange(sample_count, dtype=np.int32)
    weights = np.maximum(
        0.0,
        np.minimum(rows + 1.0, right[..., None])
        - np.maximum(rows, left[..., None]),
    )
    valid = (
        np.isfinite(trace)
        & (left >= 0.0)
        & (right <= float(detector_rows))
    )
    weights[~valid] = 0.0
    return rows, weights.astype(np.float32), valid


def extract_fractional_aperture(
    image: np.ndarray,
    variance: np.ndarray,
    traces: np.ndarray,
    *,
    pixel_mask: np.ndarray | None = None,
    width: float = 5.0,
) -> AlgorithmResult:
    """Sum detector signal in exact fractional top-hat apertures.

    The routine is independent of instrument name and takes its fiber count
    from ``traces.shape[0]``.  Diagonal detector variance is propagated with
    the square of the same actual weights used for the flux.  ``pixel_mask``
    is optional and defaults to no masked pixels for quick-look use.
    """

    data = np.asarray(image, dtype=float)
    detector_variance = np.asarray(variance, dtype=float)
    trace = np.asarray(traces, dtype=float)
    if (
        data.ndim != 2
        or detector_variance.shape != data.shape
        or trace.ndim != 2
        or trace.shape[1] != data.shape[1]
    ):
        raise ValueError("image, variance, and trace shapes are incompatible")
    mask = (
        np.zeros(data.shape, dtype=bool)
        if pixel_mask is None
        else np.asarray(pixel_mask, dtype=bool)
    )
    if mask.shape != data.shape:
        raise ValueError("pixel_mask must match image")

    rows, weights, aperture_valid = fractional_aperture_geometry(
        trace, data.shape[0], width=width
    )
    clipped_rows = np.clip(rows, 0, data.shape[0] - 1)
    columns = np.broadcast_to(
        np.arange(data.shape[1], dtype=np.int32)[None, :, None], rows.shape
    )
    samples = data[clipped_rows, columns]
    sample_variance = detector_variance[clipped_rows, columns]
    sample_valid = (
        aperture_valid[..., None]
        & ~mask[clipped_rows, columns]
        & np.isfinite(samples)
        & np.isfinite(sample_variance)
        & (sample_variance >= 0.0)
    )
    actual_weights = np.where(sample_valid, weights, 0.0)
    spectrum = np.sum(
        actual_weights * np.where(sample_valid, samples, 0.0), axis=-1
    )
    extracted_variance = np.sum(
        np.square(actual_weights)
        * np.where(sample_valid, sample_variance, 0.0),
        axis=-1,
    )
    effective_width = np.sum(actual_weights, axis=-1)
    valid_fraction = effective_width / float(width)
    extraction_valid = aperture_valid & (effective_width > 0.0)
    spectrum = np.where(extraction_valid, spectrum, np.nan).astype(np.float32)
    extracted_variance = np.where(
        extraction_valid, extracted_variance, np.nan
    ).astype(np.float32)
    return AlgorithmResult(
        kind="fractional_aperture_extraction",
        version=EXTRACTION_VERSION,
        arrays={
            "spectrum": spectrum,
            "variance": extracted_variance,
            "valid_pixel_fraction": valid_fraction.astype(np.float32),
            "effective_aperture_width": effective_width.astype(np.float32),
            "aperture_start_row": rows[..., 0].astype(np.int16),
            "fractional_weights": actual_weights.astype(np.float32),
            "extraction_valid": extraction_valid.astype(np.uint8),
        },
        scalars={"aperture_width_pixels": float(width)},
    )


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

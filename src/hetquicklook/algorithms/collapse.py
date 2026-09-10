"""Central detector-column collapse for extracted fiber spectra."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ..fibers import FiberTopology
from .extraction import extract_trace
from .results import AlgorithmResult


DEFAULT_COLLAPSE_COLUMNS = 200


def _central_column_bounds(n_columns: int, requested: int) -> tuple[int, int]:
    """Return the actual centered half-open column bounds."""

    selected_width = min(requested, n_columns)
    start = max(0, n_columns // 2 - selected_width // 2)
    start = min(start, n_columns - selected_width)
    return start, start + selected_width


def select_central_columns(
    spectra: np.ndarray,
    *,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
) -> np.ndarray:
    """Select a centered detector-column window from ``(nfiber, ncolumn)`` data.

    If the requested width exceeds the available spectrum, all available
    columns are selected.  The selected width remains explicit in the result
    shape, and no wavelength array is required.
    """

    values = np.asarray(spectra)
    if values.ndim != 2:
        raise ValueError("spectra must be a two-dimensional (fiber, column) array")
    try:
        requested = int(collapse_columns)
    except (TypeError, ValueError) as error:
        raise ValueError("collapse_columns must be a positive integer") from error
    if requested <= 0:
        raise ValueError("collapse_columns must be a positive integer")
    start, stop = _central_column_bounds(values.shape[1], requested)
    return values[:, start:stop]


def _reduce_rows(values: np.ndarray, statistic: str) -> tuple[np.ndarray, np.ndarray]:
    """Reduce rows and return values plus finite-sample counts."""

    finite = np.isfinite(values)
    counts = np.count_nonzero(finite, axis=1).astype(np.int32)
    with np.errstate(all="ignore"):
        if statistic == "mean":
            reduced = np.nanmean(np.where(finite, values, np.nan), axis=1)
        elif statistic == "median":
            reduced = np.nanmedian(np.where(finite, values, np.nan), axis=1)
        elif statistic == "sum":
            reduced = np.nansum(np.where(finite, values, 0.0), axis=1)
        else:
            raise ValueError("statistic must be 'mean', 'median', or 'sum'")
    return np.where(counts > 0, reduced, np.nan), counts


def collapse_extracted_spectra(
    spectra: np.ndarray,
    variance: np.ndarray | None = None,
    *,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    statistic: str = "mean",
) -> AlgorithmResult:
    """Collapse extracted ``(fiber, detector-column)`` spectra spatially.

    The default retains the current quick-look ``mean`` statistic after
    selecting the central 200 detector columns.  The statistic is explicit so
    that the remaining scientific choice can be changed without changing
    column selection or extraction.
    """

    values = np.asarray(spectra, dtype=float)
    if values.ndim != 2:
        raise ValueError("spectra must be a two-dimensional array")
    selected = select_central_columns(values, collapse_columns=collapse_columns)
    selected_variance: np.ndarray | None = None
    if variance is not None:
        variance_array = np.asarray(variance, dtype=float)
        if variance_array.shape != values.shape:
            raise ValueError("variance must match spectra shape")
        selected_variance = select_central_columns(
            variance_array, collapse_columns=collapse_columns
        )
    fiber_values, counts = _reduce_rows(selected, statistic)
    central_start, _ = _central_column_bounds(values.shape[1], int(collapse_columns))
    fiber_variance: np.ndarray | None = None
    if selected_variance is not None and statistic in {"mean", "sum"}:
        valid_variance = np.isfinite(selected) & np.isfinite(selected_variance)
        with np.errstate(all="ignore"):
            summed = np.sum(
                np.where(valid_variance, np.maximum(selected_variance, 0.0), 0.0),
                axis=1,
            )
        denominator = np.square(counts) if statistic == "mean" else np.ones_like(counts)
        fiber_variance = np.where(counts > 0, summed / denominator, np.nan)
    arrays: dict[str, np.ndarray] = {
        "selected_spectra": selected,
        "fiber_values": fiber_values,
        "valid_column_count": counts,
    }
    if selected_variance is not None:
        arrays["selected_variance"] = selected_variance
    if fiber_variance is not None:
        arrays["fiber_variance"] = fiber_variance
        arrays["fiber_errors"] = np.sqrt(fiber_variance)
    return AlgorithmResult(
        kind="central_column_collapse",
        version="central-column-collapse-1.0",
        arrays=arrays,
        scalars={
            "collapse_columns_requested": int(collapse_columns),
            "collapse_columns_selected": int(selected.shape[1]),
            "collapse_statistic": statistic,
            "central_start": central_start,
        },
    )


# Short aliases for callers that describe the operation in terms of spectra.
collapse_spectra = collapse_extracted_spectra
collapse_fiber_spectra = collapse_extracted_spectra


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

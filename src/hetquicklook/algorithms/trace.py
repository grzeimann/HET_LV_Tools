"""Shared continuum-flat fiber trace construction for VIRUS and LRS2."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import numpy as np
from scipy.ndimage import percentile_filter

from .results import AlgorithmResult


TRACE_ALGORITHM_VERSION = "trace-1.2"
DEFAULT_TRACE_CHUNKS = 40
DEFAULT_TRACE_DEGREE = 4


@contextmanager
def _timed_trace_stage(
    timings: dict[str, float] | None, name: str
):
    """Accumulate one optional trace-fitting stage measurement."""

    if timings is None:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        timings[name] = timings.get(name, 0.0) + perf_counter() - started


def robust_polyfit_predict(
    x_observed: np.ndarray,
    y_observed: np.ndarray,
    x_prediction: np.ndarray,
    *,
    degree: int = DEFAULT_TRACE_DEGREE,
) -> np.ndarray:
    """Fit a robust low-order polynomial and evaluate it on ``x_prediction``.

    The fit is performed in a normalized coordinate for numerical stability.
    A short iteratively reweighted Huber fit supplies the robust behavior of
    the reference implementation without making this package depend on a
    second machine-learning library.
    """

    x_obs = np.asarray(x_observed, dtype=float).ravel()
    y_obs = np.asarray(y_observed, dtype=float).ravel()
    x_pred = np.asarray(x_prediction, dtype=float).ravel()
    valid = np.isfinite(x_obs) & np.isfinite(y_obs)
    if np.count_nonzero(valid) < 2:
        return np.full(x_pred.shape, np.nan, dtype=float)
    x_obs = x_obs[valid]
    y_obs = y_obs[valid]
    fit_degree = max(1, min(int(degree), 4, x_obs.size - 1))
    minimum = float(np.min(x_obs))
    maximum = float(np.max(x_obs))
    span = maximum - minimum
    if not np.isfinite(span) or span <= 0.0:
        return np.full(x_pred.shape, float(np.median(y_obs)), dtype=float)

    scaled = 2.0 * (x_obs - minimum) / span - 1.0
    prediction_scaled = 2.0 * (x_pred - minimum) / span - 1.0
    weights = np.ones(x_obs.shape, dtype=float)
    coefficients: np.ndarray | None = None
    for _ in range(8):
        try:
            coefficients = np.polynomial.polynomial.polyfit(
                scaled, y_obs, fit_degree, w=weights
            )
        except (ValueError, np.linalg.LinAlgError):
            coefficients = None
            break
        residuals = y_obs - np.polynomial.polynomial.polyval(scaled, coefficients)
        scale = 1.4826 * float(np.median(np.abs(residuals)))
        if not np.isfinite(scale) or scale <= np.finfo(float).eps:
            break
        threshold = 1.35 * scale
        absolute = np.abs(residuals)
        weights = np.minimum(1.0, threshold / np.maximum(absolute, np.finfo(float).eps))
    if coefficients is None:
        return np.full(x_pred.shape, np.nan, dtype=float)
    return np.polynomial.polynomial.polyval(prediction_scaled, coefficients)


def _ordinary_polyfit_predict(
    x_observed: np.ndarray,
    y_observed: np.ndarray,
    x_prediction: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Fit one ordinary polynomial, preserving the trace validity rules."""

    x_obs = np.asarray(x_observed, dtype=float).ravel()
    y_obs = np.asarray(y_observed, dtype=float).ravel()
    x_pred = np.asarray(x_prediction, dtype=float).ravel()
    valid = np.isfinite(x_obs) & np.isfinite(y_obs) & (y_obs > 0.0)
    if np.count_nonzero(valid) < 2:
        return np.full(x_pred.shape, np.nan, dtype=float)
    x_obs = x_obs[valid]
    y_obs = y_obs[valid]
    fit_degree = max(1, min(int(degree), 4, x_obs.size - 1))
    minimum = float(np.min(x_obs))
    maximum = float(np.max(x_obs))
    span = maximum - minimum
    if not np.isfinite(span) or span <= 0.0:
        return np.full(x_pred.shape, float(np.median(y_obs)), dtype=float)
    scaled = 2.0 * (x_obs - minimum) / span - 1.0
    prediction_scaled = 2.0 * (x_pred - minimum) / span - 1.0
    design = np.polynomial.polynomial.polyvander(scaled, fit_degree)
    try:
        coefficients, _, _, _ = np.linalg.lstsq(design, y_obs, rcond=None)
    except (ValueError, np.linalg.LinAlgError):
        return np.full(x_pred.shape, np.nan, dtype=float)
    return np.polynomial.polynomial.polyval(prediction_scaled, coefficients)


def _fast_polyfit_predict(
    x_observed: np.ndarray,
    y_observed: np.ndarray,
    x_prediction: np.ndarray,
    *,
    degree: int,
) -> np.ndarray:
    """Fit common complete trace samples in one batched least-squares solve."""

    x_obs = np.asarray(x_observed, dtype=float).ravel()
    samples = np.asarray(y_observed, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != x_obs.size:
        raise ValueError("batched trace samples must be shaped (fiber, sample)")
    x_pred = np.asarray(x_prediction, dtype=float).ravel()
    dense = np.zeros((samples.shape[0], x_pred.size), dtype=float)
    valid = np.isfinite(samples) & (samples > 0.0)
    valid &= np.isfinite(x_obs)[None, :]
    counts = np.count_nonzero(valid, axis=1)
    fit_degree = max(1, min(int(degree), 4, x_obs.size - 1))
    complete = np.all(valid, axis=1) & (counts >= fit_degree + 1)

    if np.any(complete):
        minimum = float(np.min(x_obs))
        maximum = float(np.max(x_obs))
        span = maximum - minimum
        if np.isfinite(span) and span > 0.0:
            scaled = 2.0 * (x_obs - minimum) / span - 1.0
            prediction_scaled = 2.0 * (x_pred - minimum) / span - 1.0
            design = np.polynomial.polynomial.polyvander(scaled, fit_degree)
            try:
                coefficients, _, _, _ = np.linalg.lstsq(
                    design, samples[complete].T, rcond=None
                )
            except (ValueError, np.linalg.LinAlgError):
                complete = np.zeros_like(complete)
            else:
                prediction = np.polynomial.polynomial.polyvander(
                    prediction_scaled, fit_degree
                ) @ coefficients
                dense[complete] = prediction.T
        else:
            dense[complete] = np.median(samples[complete], axis=1)[:, None]

    fallback = np.flatnonzero((counts > 0) & ~complete)
    for fiber in fallback:
        dense[fiber] = _ordinary_polyfit_predict(
            x_obs[valid[fiber]],
            samples[fiber, valid[fiber]],
            x_pred,
            degree=degree,
        )
    return dense


def _percentile_filter_1d(values: np.ndarray, window: int, percentile: float) -> np.ndarray:
    """Apply a nearest-edge one-dimensional finite percentile filter."""

    data = np.asarray(values, dtype=float).ravel()
    if data.size == 0:
        return data.copy()
    size = max(3, int(window))
    if size % 2 == 0:
        size += 1
    size = min(size, 2 * data.size - 1) if data.size > 1 else 1
    if not np.all(np.isfinite(data)):
        raise ValueError("flat trace-detection profile contains non-finite values")
    if size <= 1:
        return data.copy()
    return percentile_filter(
        data,
        percentile=percentile,
        size=size,
        mode="nearest",
    )


def _gaussian_smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth a one-dimensional profile with a normalized Gaussian kernel."""

    data = np.asarray(values, dtype=float).ravel()
    if data.size == 0:
        return data.copy()
    width = max(1, int(np.ceil(4.0 * max(0.5, float(sigma)))))
    coordinate = np.arange(-width, width + 1, dtype=float)
    kernel = np.exp(-0.5 * np.square(coordinate / max(0.5, float(sigma))))
    kernel /= np.sum(kernel)
    finite = np.isfinite(data)
    if not np.all(finite):
        replacement = float(np.nanmedian(data)) if np.any(finite) else 0.0
        data = np.where(finite, data, replacement)
    padded = np.pad(data, width, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def preprocess_flat_for_detection(
    profile: np.ndarray,
    *,
    percentile_window: int = 201,
    percentile: float = 5.0,
    background_degree: int = 2,
    gaussian_sigma: float = 1.5,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """Remove broad cross-dispersion structure and smooth a flat profile."""

    values = np.asarray(profile, dtype=float).ravel()
    if values.size == 0:
        return values.copy()
    with _timed_trace_stage(timings, "percentile_filter"):
        broad_background = _percentile_filter_1d(
            values, percentile_window, percentile
        )
    with _timed_trace_stage(timings, "background_polynomial"):
        x = np.arange(values.size, dtype=float)
        finite = np.isfinite(broad_background)
        degree = min(
            max(0, int(background_degree)),
            max(0, int(np.count_nonzero(finite) - 1)),
        )
        if np.count_nonzero(finite) >= degree + 1:
            coefficients = np.polynomial.polynomial.polyfit(
                x[finite], broad_background[finite], degree
            )
            background = np.polynomial.polynomial.polyval(x, coefficients)
        else:
            background = np.nan_to_num(broad_background, nan=0.0)
    with _timed_trace_stage(timings, "gaussian_smoothing"):
        return _gaussian_smooth_1d(values - background, gaussian_sigma)


def _subpixel_peak_positions(profile: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Localize integer peak candidates with the reference three-point parabola."""

    values = np.asarray(profile, dtype=float).ravel()
    candidates = np.asarray(candidates, dtype=int).ravel()
    valid = (candidates >= 1) & (candidates < values.size - 1)
    candidates = candidates[valid]
    if candidates.size == 0:
        return np.empty((0,), dtype=float)
    left = values[candidates - 1]
    center = values[candidates]
    right = values[candidates + 1]
    denominator = 2.0 * (right - 2.0 * center + left)
    with np.errstate(divide="ignore", invalid="ignore"):
        result = candidates - (right - left) / denominator
    return np.where(np.isfinite(result), result, candidates.astype(float))


def _trace_from_flat_chunk(
    profile: np.ndarray,
    n_fibers: int,
    reference: np.ndarray,
    *,
    timings: dict[str, float] | None = None,
) -> np.ndarray:
    """Detect one cross-dispersion chunk and fill configured dead fibers."""

    with _timed_trace_stage(timings, "flat_profile_preprocessing"):
        processed = preprocess_flat_for_detection(profile, timings=timings)
    with _timed_trace_stage(timings, "peak_detection_assignment"):
        finite = np.where(np.isfinite(processed), processed, -np.inf)
        differences = np.diff(finite)
        candidates = np.where(
            (differences[:-1] > 0.0) & (differences[1:] < 0.0)
        )[0] + 1
        heights = finite[candidates]
        active_count = int(np.count_nonzero(np.asarray(reference)[:, 1] == 0.0))
        if candidates.size > active_count:
            selected = np.argsort(heights)[::-1][:active_count]
            candidates = np.sort(candidates[selected])
        observed = _subpixel_peak_positions(profile, candidates)
        trace = np.zeros(n_fibers, dtype=float)
        good = np.flatnonzero(np.asarray(reference)[:, 1] == 0.0)
        if observed.size == good.size and observed.size:
            trace[good] = observed
            for missing in np.flatnonzero(np.asarray(reference)[:, 1] != 0.0):
                nearest = good[np.argmin(np.abs(missing - good))]
                trace[missing] = (
                    trace[nearest]
                    + reference[missing, 0]
                    - reference[nearest, 0]
                )
        elif observed.size == n_fibers:
            trace[:] = observed
    return trace


def _mad_std(values: np.ndarray) -> float:
    """Return the reference scaled median absolute deviation."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(finite - np.median(finite))))


def fit_fiber_traces(
    continuum_flat: np.ndarray | None = None,
    trace_reference: np.ndarray | None = None,
    *,
    specid: str | None = None,
    ifuid: str | None = None,
    amplifier: str | None = None,
    amp: str | None = None,
    zipcode: Any | None = None,
    master_ldls_array: np.ndarray | None = None,
    master_flat_array: np.ndarray | None = None,
    n_chunks: int = DEFAULT_TRACE_CHUNKS,
    degree: int = DEFAULT_TRACE_DEGREE,
    fit_method: str = "robust",
    detector_column_start: int = 0,
    timings: dict[str, float] | None = None,
) -> AlgorithmResult:
    """Fit a dense detector trace map from a loaded continuum flat.

    Args:
        continuum_flat: Detector image after detector preparation, shaped
            ``(detector_row, detector_column)``.
        trace_reference: Dated ``fiber_loc`` table shaped ``(nfiber, 2)``;
            column 0 supplies reference row offsets and column 1 marks dead
            fibers with a nonzero value.
        specid, ifuid, amplifier: Identity fields needed only for the known
            ``504/018/RU`` hardware exception.
        zipcode: Optional object with ``specid``, ``ifuid``, and ``amp``
            attributes, retained as a convenient identity adapter.
        fit_method: ``"robust"`` for the existing Huber fit or ``"fast"``
            for ordinary polynomial least squares used by compact quick looks.
        detector_column_start: Original prepared-detector column corresponding
            to local column zero. This records provenance only; all returned
            trace coordinates remain local to ``continuum_flat``.
        timings: Optional diagnostic accumulator for trace-fitting substages.

    Returns:
        Named dense trace and trace QA arrays.  The first dimension follows
        the supplied reference and is therefore naturally 112 for VIRUS or
        140 for LRS2, except for the demonstrated ``504/018/RU`` exception.
    """

    if continuum_flat is None:
        continuum_flat = master_ldls_array if master_ldls_array is not None else master_flat_array
    if continuum_flat is None or trace_reference is None:
        raise TypeError("fit_fiber_traces requires a loaded continuum flat and trace_reference")
    if zipcode is not None:
        specid = getattr(zipcode, "specid", specid)
        ifuid = getattr(zipcode, "ifuid", ifuid)
        amplifier = getattr(zipcode, "amp", amplifier)
    amplifier = amp if amp is not None else amplifier
    image = np.asarray(continuum_flat, dtype=float)
    reference = np.atleast_2d(np.asarray(trace_reference, dtype=float))
    if image.ndim != 2 or reference.ndim != 2 or reference.shape[1] < 2:
        raise ValueError("continuum_flat must be 2D and trace_reference must be Nx2")
    normalized_fit_method = str(fit_method).strip().casefold()
    if normalized_fit_method not in {"fast", "robust"}:
        raise ValueError("fit_method must be 'fast' or 'robust'")
    try:
        column_start = int(detector_column_start)
    except (TypeError, ValueError) as error:
        raise ValueError("detector_column_start must be an integer") from error
    if column_start < 0:
        raise ValueError("detector_column_start must be nonnegative")
    if image.shape[1] <= 0:
        raise ValueError("continuum_flat must contain at least one detector column")
    if reference.shape[0] == 0:
        raise ValueError("trace_reference must contain at least one fiber")
    chunks = int(n_chunks)
    if chunks <= 0:
        raise ValueError("n_chunks must be positive")
    x_chunks_array = np.array_split(np.arange(image.shape[1]), min(chunks, image.shape[1]))
    x_chunks = np.array([np.mean(chunk) for chunk in x_chunks_array], dtype=float)
    sampled = np.zeros((reference.shape[0], len(x_chunks)), dtype=float)
    for index, chunk in enumerate(x_chunks_array):
        with _timed_trace_stage(timings, "chunk_profile_collapse"):
            profile = np.nanmedian(image[:, chunk], axis=1)
        if timings is None:
            sampled[:, index] = _trace_from_flat_chunk(
                profile, reference.shape[0], reference
            )
        else:
            sampled[:, index] = _trace_from_flat_chunk(
                profile, reference.shape[0], reference, timings=timings
            )

    x = np.arange(image.shape[1], dtype=float)
    with _timed_trace_stage(timings, "polynomial_trace_fit"):
        if normalized_fit_method == "fast":
            dense = _fast_polyfit_predict(
                x_chunks, sampled, x, degree=degree
            )
        else:
            dense = np.zeros((reference.shape[0], image.shape[1]), dtype=float)
            for fiber in range(reference.shape[0]):
                valid = np.isfinite(sampled[fiber]) & (sampled[fiber] > 0.0)
                if np.any(valid):
                    dense[fiber] = robust_polyfit_predict(
                        x_chunks[valid], sampled[fiber, valid], x, degree=degree
                    )

    qa_started = perf_counter() if timings is not None else 0.0
    normalized_specid = None if specid is None else str(specid).strip().zfill(3)
    normalized_ifuid = None if ifuid is None else str(ifuid).strip().zfill(3)
    normalized_amp = None if amplifier is None else str(amplifier).strip().upper()
    if (
        normalized_specid == "504"
        and normalized_ifuid == "018"
        and normalized_amp == "RU"
    ):
        dense = dense[:-1]
        reference = reference[:-1]
        sampled = sampled[:-1]

    x_indices = np.clip(np.rint(x_chunks).astype(int), 0, image.shape[1] - 1)
    modeled = dense[:, x_indices]
    valid_samples = (
        np.isfinite(sampled)
        & (sampled > 0.0)
        & np.isfinite(modeled)
    )
    residuals = np.where(valid_samples, sampled - modeled, np.nan)
    valid_counts = np.count_nonzero(valid_samples, axis=1).astype(np.int16)
    residual_rms = np.full((dense.shape[0],), np.nan, dtype=float)
    for fiber in range(dense.shape[0]):
        if valid_counts[fiber] >= 2:
            residual_rms[fiber] = _mad_std(residuals[fiber, valid_samples[fiber]])
    interpolated = (reference[:, 1] != 0.0).astype(np.uint8)
    result = AlgorithmResult(
        kind="trace",
        version=TRACE_ALGORITHM_VERSION,
        arrays={
            "fiber_trace_map": dense,
            "per_fiber_trace_residual_rms": residual_rms,
            "trace_sample_columns": x_chunks,
            "sampled_trace_positions": sampled,
            "trace_sample_valid_mask": valid_samples.astype(np.uint8),
            "trace_fit_residuals": residuals.astype(np.float32),
            "per_fiber_valid_sample_count": valid_counts,
            "trace_interpolated_fiber_mask": interpolated,
            "trace_reference": reference,
        },
        scalars={
            "trace_len": int(image.shape[1]),
            "trace_n_chunks": chunks,
            "trace_degree_requested": int(degree),
            "trace_fit_method": normalized_fit_method,
            "trace_column_start": column_start,
            "trace_column_stop": column_start + int(image.shape[1]),
        },
        metadata={
            "trace_map_shape": list(dense.shape),
            "trace_model": (
                "per_fiber_huber_polynomial"
                if normalized_fit_method == "robust"
                else "per_fiber_ordinary_least_squares_polynomial"
            ),
            "trace_fit_method": normalized_fit_method,
            "trace_n_chunks": chunks,
            "trace_degree_requested": int(degree),
            "trace_column_bounds": [
                column_start,
                column_start + int(image.shape[1]),
            ],
            "trace_sample_state": (
                "measured_active_fibers_or_reference_offset_for_configured_dead_fibers"
            ),
        },
    )
    if timings is not None:
        timings["trace_result_qa"] = (
            timings.get("trace_result_qa", 0.0) + perf_counter() - qa_started
        )
    return result


# Clearer name for callers that do not need the historical pipeline spelling.
build_trace_map = fit_fiber_traces

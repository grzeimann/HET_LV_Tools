"""Minimal shared detector preparation for VIRUS and LRS2 amplifiers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .results import AlgorithmResult


DETECTOR_REDUCTION_VERSION = "detector-2.0"
DEFAULT_GAIN = 0.85
DEFAULT_READ_NOISE = 3.0


def orient_amplifier_image(
    image: np.ndarray,
    amplifier: str,
    ampname: str | None = None,
) -> np.ndarray:
    """Orient an already trimmed amplifier image using the raw conventions.

    ``LU`` and ``RL`` are flipped in both axes.  ``AMPNAME`` values ``LR`` and
    ``UL`` apply an additional column flip.  The input is never modified.
    """

    result = np.array(image, dtype=float, copy=True)
    amp = str(amplifier).replace(" ", "").upper()
    if amp in {"LU", "RL"}:
        result = result[::-1, ::-1]
    normalized_ampname = (
        None if ampname is None else str(ampname).replace(" ", "").upper()
    )
    if normalized_ampname in {"LR", "UL"}:
        result = result[:, ::-1]
    return result


def _biweight_location_rows(values: np.ndarray) -> np.ndarray:
    """Return the reference fixed-median biweight estimate for each row."""

    data = np.asarray(values, dtype=float)
    center = np.nanmedian(data, axis=1)
    delta = data - center[:, None]
    mad = np.nanmedian(np.abs(delta), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        u = delta / (6.0 * mad[:, None])
    rejected = np.abs(u) >= 1.0
    weights = np.square(1.0 - np.square(u))
    weights[rejected] = 0.0
    weights[~np.isfinite(delta)] = 0.0
    delta_weighted = np.where(np.isfinite(delta), delta * weights, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        location = center + np.sum(delta_weighted, axis=1) / np.sum(weights, axis=1)
    return np.where(mad == 0.0, center, location)


def _header_float(header: Mapping[str, Any], key: str, fallback: float) -> float:
    """Read a positive finite detector scalar, retaining the reference fallback."""

    try:
        value = float(header.get(key, fallback))
    except (TypeError, ValueError):
        return fallback
    return value if np.isfinite(value) and value > 0.0 else fallback


def reduce_amplifier_array(
    data: np.ndarray,
    header: Mapping[str, Any] | None = None,
) -> AlgorithmResult:
    """Prepare one loaded raw amplifier image for quick-look algorithms.

    The operation follows the supplied VIRUSFlow/Panacea contract: estimate a
    robust row-wise overscan level, subtract and trim the overscan columns,
    orient the detector, apply gain, and form diagonal detector variance from
    read noise plus positive signal.  No bias, dark, flat, cosmic-ray, or
    scattered-light correction is performed here.

    Args:
        data: Raw primary-HDU detector image with shape ``(ny, nx_raw)``.
        header: Header containing ``CCDPOS``, ``CCDHALF``, ``AMPNAME``,
            ``GAIN``, and ``RDNOISE`` when available.

    Returns:
        An :class:`AlgorithmResult` containing the prepared detector image,
        variance/error, overscan model, and QA evidence.
    """

    image = np.asarray(data, dtype=float)
    if image.ndim != 2:
        raise ValueError("raw amplifier data must be a two-dimensional array")
    header_values = dict(header or {})
    raw_width = image.shape[1]
    overscan_columns = int(32 * (raw_width / 1064.0))
    overscan_columns = max(0, min(raw_width // 4, overscan_columns))
    if overscan_columns:
        sample_columns = (
            overscan_columns - 2 if overscan_columns >= 3 else overscan_columns
        )
        overscan_region = image[:, -sample_columns:]
        overscan_model = _biweight_location_rows(overscan_region)
        corrected = image - overscan_model[:, None]
        corrected = corrected[:, : raw_width - overscan_columns]
    else:
        overscan_model = np.zeros(image.shape[0], dtype=float)
        corrected = image.copy()

    ccdpos = str(header_values.get("CCDPOS", "")).replace(" ", "").upper()
    ccdhalf = str(header_values.get("CCDHALF", "")).replace(" ", "").upper()
    amplifier = ccdpos + ccdhalf
    ampname_value = header_values.get("AMPNAME")
    ampname = None if ampname_value is None else str(ampname_value)
    gain = _header_float(header_values, "GAIN", DEFAULT_GAIN)
    read_noise = _header_float(header_values, "RDNOISE", DEFAULT_READ_NOISE)

    oriented = orient_amplifier_image(corrected, amplifier, ampname)
    prepared = oriented * gain
    positive_signal = np.where(prepared > 0.0, prepared, 0.0)
    error = np.sqrt(read_noise**2 + positive_signal)
    variance = np.square(error)
    return AlgorithmResult(
        kind="detector_reduction",
        version=DETECTOR_REDUCTION_VERSION,
        arrays={
            "overscan_model": overscan_model,
            "overscan_corrected_image": corrected,
            "oriented_detector_image": prepared.astype(np.float32),
            "detector_error": error.astype(np.float32),
            "detector_variance": variance.astype(np.float32),
        },
        scalars={
            "gain": gain,
            "read_noise": read_noise,
            "overscan_columns": overscan_columns,
        },
        metadata={
            "header": header_values,
            "amp": amplifier,
            "ampname": ampname,
        },
    )


prepare_detector = reduce_amplifier_array

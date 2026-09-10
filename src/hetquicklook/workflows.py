"""Operational quick-look workflows built from small array algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .algorithms.centroid import Centroid, weighted_centroid
from .algorithms.collapse import (
    DEFAULT_COLLAPSE_COLUMNS,
    DEFAULT_COLLAPSE_STATISTIC,
    collapse_extracted_spectra,
    collapse_fibers,
    spatial_image_from_fiber_values,
)
from .algorithms.extraction import extract_fractional_aperture
from .algorithms.spatial import gaussian_splat
from .fibers import FiberTopology
from .instrument import (
    Instrument,
    lrs2_amplifier_tokens_for_channel,
    lrs2_component_for_channel,
)


VIRUS_SPATIAL_DEFAULTS = {
    "gaussian_fwhm_arcsec": 1.5,
    "pixel_scale_arcsec": 1.0,
    "intended_fiducial": (0.0, 0.0),
}
LRS2_SPATIAL_DEFAULTS = {
    "gaussian_fwhm_arcsec": 1.2,
    "pixel_scale_arcsec": 0.4,
    "intended_fiducial": (0.0, 0.0),
}
INSTRUMENT_SPATIAL_DEFAULTS = {
    Instrument.VIRUS: VIRUS_SPATIAL_DEFAULTS,
    Instrument.LRS2: LRS2_SPATIAL_DEFAULTS,
}
LRS2_STANDARD_FIDUCIAL = LRS2_SPATIAL_DEFAULTS["intended_fiducial"]
VIRUS_STANDARD_FIDUCIAL = VIRUS_SPATIAL_DEFAULTS["intended_fiducial"]


def spatial_defaults_for(
    instrument: Instrument | str,
) -> Mapping[str, object]:
    """Return the workflow defaults for one supported instrument."""

    return INSTRUMENT_SPATIAL_DEFAULTS[Instrument.from_value(instrument)]


@dataclass(frozen=True)
class SpatialQuicklook:
    """Collapsed values, spatial image, coordinates, and QA evidence."""

    fiber_values: Mapping[str, float]
    image: np.ndarray
    instrument: Instrument | None = None
    fiber_positions: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    fiber_errors: Mapping[str, float] = field(default_factory=dict)
    spatial_weight: np.ndarray | None = None
    spatial_support: np.ndarray | None = None
    spatial_x_coordinates: np.ndarray | None = None
    spatial_y_coordinates: np.ndarray | None = None
    spatial_gaussian_fwhm_arcsec: float | None = None
    spatial_pixel_scale_arcsec: float | None = None
    intended_fiducial: tuple[float, float] | None = None
    collapse_columns: int | None = None
    collapse_statistic: str | None = None
    extraction_width: float | None = None
    extracted_spectra: np.ndarray | None = None
    extraction_variance: np.ndarray | None = None
    extraction_valid_fraction: np.ndarray | None = None
    effective_aperture_width: np.ndarray | None = None
    extraction_valid: np.ndarray | None = None


@dataclass(frozen=True)
class PointingQuicklook(SpatialQuicklook):
    """Standard-star image with intended and measured positions as evidence."""

    measured_centroid: Centroid = field(
        default_factory=lambda: Centroid(float("nan"), float("nan"), 0.0, 0)
    )
    requested_position: tuple[float, float] | None = None

    @property
    def offset(self) -> tuple[float, float] | None:
        """Return measured-minus-requested ``(x, y)`` when available."""

        if self.requested_position is None:
            return None
        return (
            self.measured_centroid.x - self.requested_position[0],
            self.measured_centroid.y - self.requested_position[1],
        )


@dataclass(frozen=True)
class LRS2ChannelQuicklook:
    """Spatial evidence for one complete two-amplifier LRS2 channel.

    The component products retain the detector, trace, extraction, and
    amplifier-specific evidence.  The channel-level arrays begin only after
    each amplifier has been reduced to collapsed fiber values and physical
    IFU positions.
    """

    channel: str
    amplifier_products: Mapping[str, SpatialQuicklook]
    fiber_positions: np.ndarray
    fiber_values: np.ndarray
    image: np.ndarray
    spatial_x_coordinates: np.ndarray
    spatial_y_coordinates: np.ndarray
    spatial_support: np.ndarray
    spatial_weight: np.ndarray
    intended_fiducial: tuple[float, float]
    measured_centroid: Centroid | None
    spatial_gaussian_fwhm_arcsec: float
    spatial_pixel_scale_arcsec: float
    fiber_errors: np.ndarray | None = None
    spatial_variance: np.ndarray | None = None

    @property
    def gaussian_fwhm_arcsec(self) -> float:
        """Return the configured Gaussian FWHM using the short result name."""

        return self.spatial_gaussian_fwhm_arcsec

    @property
    def pixel_scale_arcsec(self) -> float:
        """Return the configured output pixel scale using the short name."""

        return self.spatial_pixel_scale_arcsec

    @property
    def offset(self) -> tuple[float, float] | None:
        """Return measured-minus-intended position when a centroid exists."""

        if self.measured_centroid is None:
            return None
        return (
            self.measured_centroid.x - self.intended_fiducial[0],
            self.measured_centroid.y - self.intended_fiducial[1],
        )


def _dense_trace_map(topology: FiberTopology, detector_columns: int) -> np.ndarray:
    """Convert topology trace samples to a dense ``(fiber, column)`` map."""

    if detector_columns <= 0:
        raise ValueError("detector must contain at least one detector column")
    x_target = np.arange(detector_columns, dtype=float)
    rows: list[np.ndarray] = []
    for fiber_id in topology.fiber_ids:
        trace = topology.traces[fiber_id]
        x_values = np.asarray(trace.detector_x, dtype=float)
        y_values = np.asarray(trace.detector_y, dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(valid):
            rows.append(np.full(detector_columns, np.nan))
            continue
        if np.count_nonzero(valid) == 1:
            rows.append(np.full(detector_columns, y_values[valid][0]))
            continue
        x_values = x_values[valid]
        y_values = y_values[valid]
        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        unique_x, inverse = np.unique(x_values, return_inverse=True)
        if unique_x.size != x_values.size:
            sums = np.zeros(unique_x.size, dtype=float)
            counts = np.zeros(unique_x.size, dtype=float)
            np.add.at(sums, inverse, y_values)
            np.add.at(counts, inverse, 1.0)
            y_values = sums / counts
            x_values = unique_x
        rows.append(np.interp(x_target, x_values, y_values))
    return np.asarray(rows, dtype=float)


def _has_dense_trace_geometry(topology: FiberTopology, detector_columns: int) -> bool:
    """Whether topology samples already describe every detector column."""

    return all(
        trace.detector_x.size == detector_columns
        and np.all(np.isfinite(trace.detector_y))
        for trace in topology.traces.values()
    )


def _resolve_spatial_parameters(
    instrument: Instrument | str,
    *,
    gaussian_fwhm: float | None,
    pixel_scale: float | None,
    spatial_defaults: Mapping[str, object] | None,
) -> tuple[Instrument, float, float, tuple[float, float]]:
    """Resolve workflow defaults and caller overrides before splatting."""

    parsed_instrument = Instrument.from_value(instrument)
    defaults = (
        spatial_defaults
        if spatial_defaults is not None
        else spatial_defaults_for(parsed_instrument)
    )
    resolved_fwhm = defaults["gaussian_fwhm_arcsec"] if gaussian_fwhm is None else gaussian_fwhm
    resolved_pixel_scale = defaults["pixel_scale_arcsec"] if pixel_scale is None else pixel_scale
    fiducial = defaults.get("intended_fiducial", (0.0, 0.0))
    if len(fiducial) != 2:
        raise ValueError("intended_fiducial must contain two coordinates")
    return (
        parsed_instrument,
        float(resolved_fwhm),
        float(resolved_pixel_scale),
        (float(fiducial[0]), float(fiducial[1])),
    )


def _spatial_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    aperture: int | float | None = None,
    gaussian_fwhm: float | None = None,
    pixel_scale: float | None = None,
    instrument: Instrument | str = Instrument.VIRUS,
    spatial_defaults: Mapping[str, object] | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
    intended_fiducial: tuple[float, float] | None = None,
) -> SpatialQuicklook:
    """Run the in-memory detector-to-fiber-to-spatial quick-look path."""

    image = np.asarray(detector, dtype=float)
    if image.ndim != 2:
        raise ValueError("detector must be a two-dimensional array")
    parsed_instrument, resolved_fwhm, resolved_pixel_scale, _ = (
        _resolve_spatial_parameters(
            instrument,
            gaussian_fwhm=gaussian_fwhm,
            pixel_scale=pixel_scale,
            spatial_defaults=spatial_defaults,
        )
    )
    dense = trace_map is not None or _has_dense_trace_geometry(topology, image.shape[1])
    if dense:
        traces = _dense_trace_map(topology, image.shape[1]) if trace_map is None else np.asarray(trace_map, dtype=float)
        if traces.shape != (len(topology.fiber_ids), image.shape[1]):
            raise ValueError("trace_map must have shape (nfiber, detector_columns)")
        variance = (
            np.ones_like(image, dtype=float)
            if detector_variance is None
            else np.asarray(detector_variance, dtype=float)
        )
        extraction = extract_fractional_aperture(
            image,
            variance,
            traces,
            width=extraction_width if aperture is None else float(aperture),
        )
        collapsed = collapse_extracted_spectra(
            extraction.get_array("spectrum"),
            extraction.get_array("variance"),
            collapse_columns=collapse_columns,
            statistic=statistic,
        )
        values = collapsed.get_array("fiber_values")
        errors_array = collapsed.arrays.get("fiber_errors")
        fiber_errors = (
            {}
            if errors_array is None
            else {
                fiber_id: float(errors_array[index])
                for index, fiber_id in enumerate(topology.fiber_ids)
            }
        )
        positions = np.asarray(
            [
                [topology.locations[fiber_id].ifu_x, topology.locations[fiber_id].ifu_y]
                for fiber_id in topology.fiber_ids
            ],
            dtype=float,
        )
        fiber_positions = {
            fiber_id: (float(positions[index, 0]), float(positions[index, 1]))
            for index, fiber_id in enumerate(topology.fiber_ids)
        }
        spatial = gaussian_splat(
            positions,
            values,
            None if errors_array is None else errors_array,
            fwhm=resolved_fwhm,
            pixel_scale=resolved_pixel_scale,
            output_shape=output_shape,
            origin=origin,
        )
        return SpatialQuicklook(
            fiber_values={
                fiber_id: float(values[index])
                for index, fiber_id in enumerate(topology.fiber_ids)
            },
            image=spatial.image,
            instrument=parsed_instrument,
            fiber_positions=fiber_positions,
            fiber_errors=fiber_errors,
            spatial_weight=spatial.weight,
            spatial_support=spatial.support,
            spatial_x_coordinates=spatial.x_coordinates,
            spatial_y_coordinates=spatial.y_coordinates,
            spatial_gaussian_fwhm_arcsec=resolved_fwhm,
            spatial_pixel_scale_arcsec=resolved_pixel_scale,
            intended_fiducial=intended_fiducial,
            collapse_columns=int(collapsed.scalars["collapse_columns_requested"]),
            collapse_statistic=str(collapsed.scalars["collapse_statistic"]),
            extraction_width=float(extraction.scalars["aperture_width_pixels"]),
            extracted_spectra=extraction.get_array("spectrum"),
            extraction_variance=extraction.get_array("variance"),
            extraction_valid_fraction=extraction.get_array("valid_pixel_fraction"),
            effective_aperture_width=extraction.get_array("effective_aperture_width"),
            extraction_valid=extraction.get_array("extraction_valid"),
        )

    # Compatibility adapter for the original sparse FiberTopology examples:
    # their trace samples are individual detector points rather than a dense
    # detector-column trace map.  New pipeline data take the shared path above.
    legacy_aperture = 1 if aperture is None else aperture
    fiber_values = collapse_fibers(
        image, topology, aperture=int(legacy_aperture), statistic=statistic
    )
    return SpatialQuicklook(
        fiber_values=fiber_values,
        image=spatial_image_from_fiber_values(fiber_values, topology),
        instrument=parsed_instrument,
        fiber_positions={
            fiber_id: (
                float(topology.locations[fiber_id].ifu_x),
                float(topology.locations[fiber_id].ifu_y),
            )
            for fiber_id in topology.fiber_ids
        },
        spatial_gaussian_fwhm_arcsec=resolved_fwhm,
        spatial_pixel_scale_arcsec=resolved_pixel_scale,
        intended_fiducial=intended_fiducial,
        collapse_columns=None,
        collapse_statistic=statistic,
        extraction_width=(None if aperture is None else float(aperture)),
    )


def run_ldls_flat_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    aperture: int | float | None = None,
    gaussian_fwhm: float | None = None,
    pixel_scale: float | None = None,
    instrument: Instrument | str = Instrument.VIRUS,
    spatial_defaults: Mapping[str, object] | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> SpatialQuicklook:
    """Create a collapsed spatial product for an LDLS-flat detector image."""

    return _spatial_quicklook(
        detector,
        topology,
        detector_variance=detector_variance,
        trace_map=trace_map,
        extraction_width=extraction_width,
        collapse_columns=collapse_columns,
        statistic=statistic,
        aperture=aperture,
        gaussian_fwhm=gaussian_fwhm,
        pixel_scale=pixel_scale,
        instrument=instrument,
        spatial_defaults=spatial_defaults,
        output_shape=output_shape,
        origin=origin,
    )


def run_standard_star_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    requested_position: tuple[float, float] | None = None,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    aperture: int | float | None = None,
    gaussian_fwhm: float | None = None,
    pixel_scale: float | None = None,
    instrument: Instrument | str = Instrument.VIRUS,
    spatial_defaults: Mapping[str, object] | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> PointingQuicklook:
    """Create a spatial product with measured and intended positions.

    The existing weighted centroid is exposed as a measured location. It is
    evidence for an observer and does not produce an automatic quality
    decision.
    """

    _, _, _, default_fiducial = _resolve_spatial_parameters(
        instrument,
        gaussian_fwhm=gaussian_fwhm,
        pixel_scale=pixel_scale,
        spatial_defaults=spatial_defaults,
    )
    intended_position = (
        default_fiducial if requested_position is None else requested_position
    )

    spatial = _spatial_quicklook(
        detector,
        topology,
        detector_variance=detector_variance,
        trace_map=trace_map,
        extraction_width=extraction_width,
        collapse_columns=collapse_columns,
        statistic=statistic,
        aperture=aperture,
        gaussian_fwhm=gaussian_fwhm,
        pixel_scale=pixel_scale,
        instrument=instrument,
        spatial_defaults=spatial_defaults,
        output_shape=output_shape,
        origin=origin,
        intended_fiducial=intended_position,
    )
    x = np.array([topology.locations[fiber_id].ifu_x for fiber_id in topology.fiber_ids])
    y = np.array([topology.locations[fiber_id].ifu_y for fiber_id in topology.fiber_ids])
    values = np.array([spatial.fiber_values[fiber_id] for fiber_id in topology.fiber_ids])
    measured = weighted_centroid(x, y, values)
    return PointingQuicklook(
        fiber_values=spatial.fiber_values,
        image=spatial.image,
        instrument=spatial.instrument,
        fiber_positions=spatial.fiber_positions,
        measured_centroid=measured,
        requested_position=intended_position,
        fiber_errors=spatial.fiber_errors,
        spatial_weight=spatial.spatial_weight,
        spatial_support=spatial.spatial_support,
        spatial_x_coordinates=spatial.spatial_x_coordinates,
        spatial_y_coordinates=spatial.spatial_y_coordinates,
        spatial_gaussian_fwhm_arcsec=spatial.spatial_gaussian_fwhm_arcsec,
        spatial_pixel_scale_arcsec=spatial.spatial_pixel_scale_arcsec,
        intended_fiducial=spatial.intended_fiducial,
        collapse_columns=spatial.collapse_columns,
        collapse_statistic=spatial.collapse_statistic,
        extraction_width=spatial.extraction_width,
        extracted_spectra=spatial.extracted_spectra,
        extraction_variance=spatial.extraction_variance,
        extraction_valid_fraction=spatial.extraction_valid_fraction,
        effective_aperture_width=spatial.effective_aperture_width,
        extraction_valid=spatial.extraction_valid,
    )


def _canonical_lrs2_channel_products(
    channel: str,
    amplifier_products: Mapping[str, SpatialQuicklook],
) -> tuple[str, dict[str, SpatialQuicklook]]:
    """Validate and order the two products belonging to ``channel``."""

    component = lrs2_component_for_channel(channel)
    expected_tokens = lrs2_amplifier_tokens_for_channel(component.name)
    expected_set = set(expected_tokens)
    normalized: dict[str, SpatialQuicklook] = {}

    for supplied_key, product in amplifier_products.items():
        key = str(supplied_key).strip().upper()
        candidates = [token for token in expected_tokens if key == token]
        # The full token is the normal interface.  Accepting the two-letter
        # suffix is useful for direct in-memory calls, but only within the
        # already selected channel so it cannot infer a cross-slot pairing.
        if not candidates:
            candidates = [token for token in expected_tokens if key == token[-2:]]
        if len(candidates) != 1:
            raise ValueError(
                f"Amplifier {supplied_key!r} does not belong uniquely to LRS2 "
                f"channel {component.name!r}; expected {expected_tokens}"
            )
        token = candidates[0]
        if token in normalized:
            raise ValueError(f"Duplicate LRS2 amplifier product for {token}")
        if not isinstance(product, SpatialQuicklook):
            raise TypeError("amplifier_products must contain SpatialQuicklook results")
        if product.instrument is not None and Instrument.from_value(product.instrument) is not Instrument.LRS2:
            raise ValueError(f"LRS2 channel product {token} has non-LRS2 instrument")
        normalized[token] = product

    missing = [token for token in expected_tokens if token not in normalized]
    if missing:
        raise ValueError(
            f"Cannot create complete LRS2 channel {component.name!r}; "
            f"missing amplifier product(s): {', '.join(missing)}"
        )
    if set(normalized) != expected_set:
        raise ValueError(
            f"LRS2 channel {component.name!r} received an unexpected amplifier pair"
        )
    return component.name, {token: normalized[token] for token in expected_tokens}


def _channel_product_arrays(
    amplifier_token: str,
    product: SpatialQuicklook,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return position/value/error arrays in the product's fiber order."""

    fiber_ids = tuple(product.fiber_values)
    if set(fiber_ids) != set(product.fiber_positions):
        raise ValueError(
            f"LRS2 amplifier product {amplifier_token} has mismatched fiber "
            "value and position identifiers"
        )
    if len(fiber_ids) != 140:
        raise ValueError(
            f"LRS2 amplifier product {amplifier_token} must contain 140 fibers; "
            f"found {len(fiber_ids)}"
        )
    positions = np.asarray(
        [product.fiber_positions[fiber_id] for fiber_id in fiber_ids], dtype=float
    )
    values = np.asarray(
        [product.fiber_values[fiber_id] for fiber_id in fiber_ids], dtype=float
    )
    if positions.shape != (140, 2) or values.shape != (140,):
        raise ValueError(
            f"LRS2 amplifier product {amplifier_token} has invalid fiber array shapes"
        )

    errors: np.ndarray | None = None
    if product.fiber_errors and set(product.fiber_errors) == set(fiber_ids):
        errors = np.asarray(
            [product.fiber_errors[fiber_id] for fiber_id in fiber_ids], dtype=float
        )
    return positions, values, errors


def combine_lrs2_channel_products(
    channel: str,
    amplifier_products: Mapping[str, SpatialQuicklook],
    *,
    gaussian_fwhm_arcsec: float | None = None,
    pixel_scale_arcsec: float | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> LRS2ChannelQuicklook:
    """Combine two independently reduced LRS2 amplifiers into one channel.

    Composition begins at the collapsed physical-fiber boundary.  The
    detector arrays, traces, extracted spectra, and detector variances remain
    available only on the two underlying :class:`SpatialQuicklook` products.
    """

    canonical_channel, products = _canonical_lrs2_channel_products(
        channel, amplifier_products
    )
    positions_and_values = [
        _channel_product_arrays(token, products[token])
        for token in products
    ]
    positions = np.concatenate(
        [item[0] for item in positions_and_values], axis=0
    )
    values = np.concatenate([item[1] for item in positions_and_values], axis=0)
    errors_by_amp = [item[2] for item in positions_and_values]
    errors = (
        np.concatenate([error for error in errors_by_amp if error is not None], axis=0)
        if all(error is not None for error in errors_by_amp)
        else None
    )
    if positions.shape != (280, 2) or values.shape != (280,):
        raise ValueError(
            f"A complete LRS2 channel must contain 280 fibers; found {values.size}"
        )

    defaults = spatial_defaults_for(Instrument.LRS2)
    resolved_fwhm = (
        float(defaults["gaussian_fwhm_arcsec"])
        if gaussian_fwhm_arcsec is None
        else float(gaussian_fwhm_arcsec)
    )
    resolved_pixel_scale = (
        float(defaults["pixel_scale_arcsec"])
        if pixel_scale_arcsec is None
        else float(pixel_scale_arcsec)
    )
    intended = tuple(defaults["intended_fiducial"])
    spatial = gaussian_splat(
        positions,
        values,
        errors,
        fwhm=resolved_fwhm,
        pixel_scale=resolved_pixel_scale,
        output_shape=output_shape,
        origin=origin,
    )

    pointing_products = [isinstance(product, PointingQuicklook) for product in products.values()]
    if any(pointing_products) and not all(pointing_products):
        raise TypeError(
            "LRS2 channel products must all be standard-star or all be flat products"
        )
    measured = (
        weighted_centroid(positions[:, 0], positions[:, 1], values)
        if all(pointing_products)
        else None
    )
    return LRS2ChannelQuicklook(
        channel=canonical_channel,
        amplifier_products=products,
        fiber_positions=positions,
        fiber_values=values,
        image=spatial.image,
        spatial_x_coordinates=spatial.x_coordinates,
        spatial_y_coordinates=spatial.y_coordinates,
        spatial_support=spatial.support,
        spatial_weight=spatial.weight,
        intended_fiducial=(float(intended[0]), float(intended[1])),
        measured_centroid=measured,
        spatial_gaussian_fwhm_arcsec=resolved_fwhm,
        spatial_pixel_scale_arcsec=resolved_pixel_scale,
        fiber_errors=errors,
        spatial_variance=spatial.variance,
    )


def combine_lrs2_channels(
    amplifier_products: Mapping[str, SpatialQuicklook],
    *,
    gaussian_fwhm_arcsec: float | None = None,
    pixel_scale_arcsec: float | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> dict[str, LRS2ChannelQuicklook]:
    """Combine a complete eight-amplifier LRS2 exposure into four channels.

    The function requires all eight authoritative amplifier tokens.  Missing
    data therefore remain explicit instead of being represented as a partial
    channel with an apparently complete result type.
    """

    expected_tokens = {
        token
        for channel in ("UV", "Orange", "Red", "Far-Red")
        for token in lrs2_amplifier_tokens_for_channel(channel)
    }
    canonical_products: dict[str, SpatialQuicklook] = {}
    for supplied_key, product in amplifier_products.items():
        token = str(supplied_key).strip().upper()
        if token in canonical_products:
            raise ValueError(f"Duplicate LRS2 amplifier product for {token}")
        canonical_products[token] = product
    supplied_tokens = set(canonical_products)
    missing = sorted(expected_tokens - supplied_tokens)
    unexpected = sorted(supplied_tokens - expected_tokens)
    if missing:
        raise ValueError(
            "Cannot create complete LRS2 channel set; missing amplifier product(s): "
            + ", ".join(missing)
        )
    if unexpected:
        raise ValueError(
            "Unexpected amplifier product(s) for LRS2 channel set: "
            + ", ".join(unexpected)
        )
    return {
        channel: combine_lrs2_channel_products(
            channel,
            {token: canonical_products[token] for token in lrs2_amplifier_tokens_for_channel(channel)},
            gaussian_fwhm_arcsec=gaussian_fwhm_arcsec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            output_shape=output_shape,
            origin=origin,
        )
        for channel in ("UV", "Orange", "Red", "Far-Red")
    }

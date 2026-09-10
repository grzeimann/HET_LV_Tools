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
from .instrument import Instrument


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

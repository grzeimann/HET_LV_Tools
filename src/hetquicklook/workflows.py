"""Operational quick-look workflows built from small array algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .algorithms.centroid import Centroid, weighted_centroid
from .algorithms.collapse import (
    collapse_extracted_spectra,
    collapse_fibers,
    spatial_image_from_fiber_values,
)
from .algorithms.extraction import extract_fractional_aperture
from .algorithms.spatial import gaussian_splat
from .fibers import FiberTopology


LRS2_STANDARD_FIDUCIAL = (0.0, 0.0)


@dataclass(frozen=True)
class SpatialQuicklook:
    """Collapsed fiber values and their IFU-plane image."""

    fiber_values: Mapping[str, float]
    image: np.ndarray
    fiber_errors: Mapping[str, float] = field(default_factory=dict)
    spatial_weight: np.ndarray | None = None
    spatial_support: np.ndarray | None = None
    extracted_spectra: np.ndarray | None = None
    extraction_variance: np.ndarray | None = None


@dataclass(frozen=True)
class PointingQuicklook(SpatialQuicklook):
    """Standard-star spatial quick look with an optional target fiducial."""

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


def _spatial_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = 200,
    statistic: str = "mean",
    aperture: int | float | None = None,
    gaussian_fwhm: float = 1.8,
    pixel_scale: float = 1.0,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> SpatialQuicklook:
    """Run the in-memory detector-to-fiber-to-spatial quick-look path."""

    image = np.asarray(detector, dtype=float)
    if image.ndim != 2:
        raise ValueError("detector must be a two-dimensional array")
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
        spatial = gaussian_splat(
            positions,
            values,
            None if errors_array is None else errors_array,
            fwhm=gaussian_fwhm,
            pixel_scale=pixel_scale,
            output_shape=output_shape,
            origin=origin,
        )
        return SpatialQuicklook(
            fiber_values={
                fiber_id: float(values[index])
                for index, fiber_id in enumerate(topology.fiber_ids)
            },
            image=spatial.image,
            fiber_errors=fiber_errors,
            spatial_weight=spatial.weight,
            spatial_support=spatial.support,
            extracted_spectra=extraction.get_array("spectrum"),
            extraction_variance=extraction.get_array("variance"),
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
    )


def run_ldls_flat_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = 200,
    statistic: str = "mean",
    aperture: int | float | None = None,
    gaussian_fwhm: float = 1.8,
    pixel_scale: float = 1.0,
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
    collapse_columns: int = 200,
    statistic: str = "mean",
    aperture: int | float | None = None,
    gaussian_fwhm: float = 1.8,
    pixel_scale: float = 1.0,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> PointingQuicklook:
    """Create a spatial product and retain the existing provisional centroid.

    The weighted centroid is kept for compatibility with the original in-memory
    workflow.  It is an explicit data product boundary and is not a final
    standard-star estimator or fiducial policy.
    """

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
        output_shape=output_shape,
        origin=origin,
    )
    x = np.array([topology.locations[fiber_id].ifu_x for fiber_id in topology.fiber_ids])
    y = np.array([topology.locations[fiber_id].ifu_y for fiber_id in topology.fiber_ids])
    values = np.array([spatial.fiber_values[fiber_id] for fiber_id in topology.fiber_ids])
    measured = weighted_centroid(x, y, values)
    return PointingQuicklook(
        fiber_values=spatial.fiber_values,
        image=spatial.image,
        measured_centroid=measured,
        requested_position=requested_position,
        fiber_errors=spatial.fiber_errors,
        spatial_weight=spatial.spatial_weight,
        spatial_support=spatial.spatial_support,
        extracted_spectra=spatial.extracted_spectra,
        extraction_variance=spatial.extraction_variance,
    )

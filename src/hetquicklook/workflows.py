"""Operational quick-look workflows built from small array algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .algorithms.centroid import Centroid, weighted_centroid
from .algorithms.collapse import collapse_fibers, spatial_image_from_fiber_values
from .fibers import FiberTopology


@dataclass(frozen=True)
class SpatialQuicklook:
    """Collapsed fiber values and their IFU-plane image."""

    fiber_values: Mapping[str, float]
    image: np.ndarray


@dataclass(frozen=True)
class PointingQuicklook(SpatialQuicklook):
    """Standard-star spatial quick look with an optional target fiducial."""

    measured_centroid: Centroid
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


def run_ldls_flat_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    aperture: int = 1,
    statistic: str = "mean",
) -> SpatialQuicklook:
    """Create a collapsed spatial product for an LDLS-flat detector image."""

    fiber_values = collapse_fibers(
        detector, topology, aperture=aperture, statistic=statistic
    )
    return SpatialQuicklook(
        fiber_values=fiber_values,
        image=spatial_image_from_fiber_values(fiber_values, topology),
    )


def run_standard_star_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    requested_position: tuple[float, float] | None = None,
    aperture: int = 1,
    statistic: str = "mean",
) -> PointingQuicklook:
    """Create a collapsed spatial product and source-location centroid."""

    fiber_values = collapse_fibers(
        detector, topology, aperture=aperture, statistic=statistic
    )
    x = np.array([topology.locations[fiber_id].ifu_x for fiber_id in topology.fiber_ids])
    y = np.array([topology.locations[fiber_id].ifu_y for fiber_id in topology.fiber_ids])
    values = np.array([fiber_values[fiber_id] for fiber_id in topology.fiber_ids])
    measured = weighted_centroid(x, y, values)
    return PointingQuicklook(
        fiber_values=fiber_values,
        image=spatial_image_from_fiber_values(fiber_values, topology),
        measured_centroid=measured,
        requested_position=requested_position,
    )


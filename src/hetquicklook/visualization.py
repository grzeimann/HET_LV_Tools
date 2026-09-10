"""Thin Matplotlib interfaces for quick-look results."""

from __future__ import annotations

from typing import Any

import numpy as np

from .fibers import FiberTopology
from .workflows import PointingQuicklook, SpatialQuicklook


def _image_extent(result: SpatialQuicklook) -> tuple[float, float, float, float] | None:
    """Return physical image bounds when the reconstruction supplied them."""

    x = result.spatial_x_coordinates
    y = result.spatial_y_coordinates
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return None
    x_step = float(np.median(np.diff(x))) if len(x) > 1 else 1.0
    y_step = float(np.median(np.diff(y))) if len(y) > 1 else 1.0
    return (
        float(x[0] - x_step / 2.0),
        float(x[-1] + x_step / 2.0),
        float(y[0] - y_step / 2.0),
        float(y[-1] + y_step / 2.0),
    )


def _fiber_positions(
    result: SpatialQuicklook, topology: FiberTopology
) -> list[tuple[float, float]]:
    """Use positions recorded on the product, with topology as compatibility fallback."""

    if result.fiber_positions:
        return list(result.fiber_positions.values())
    return [
        (location.ifu_x, location.ifu_y)
        for location in topology.locations.values()
    ]


def plot_spatial_quicklook(
    result: SpatialQuicklook,
    topology: FiberTopology,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot a spatial quick-look image and its fiber positions."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    extent = _image_extent(result)
    image_artist = ax.imshow(
        result.image,
        origin="lower",
        interpolation="nearest",
        extent=extent,
    )
    positions = _fiber_positions(result, topology)
    if extent is None:
        x_min = min(round(position[0]) for position in positions)
        y_min = min(round(position[1]) for position in positions)
        x = [position[0] - x_min for position in positions]
        y = [position[1] - y_min for position in positions]
    else:
        x = [position[0] for position in positions]
        y = [position[1] for position in positions]
    ax.scatter(x, y, facecolors="none", edgecolors="white", linewidths=0.5)
    if title:
        ax.set_title(title)
    ax.set_xlabel("IFU x")
    ax.set_ylabel("IFU y")
    return image_artist


def plot_pointing_quicklook(
    result: PointingQuicklook,
    topology: FiberTopology,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot a standard-star result with measured and requested positions."""

    artist = plot_spatial_quicklook(result, topology, ax=ax, title=title)
    axes = artist.axes
    if _image_extent(result) is None:
        positions = _fiber_positions(result, topology)
        x_min = min(round(position[0]) for position in positions)
        y_min = min(round(position[1]) for position in positions)
        measured_x = result.measured_centroid.x - x_min
        measured_y = result.measured_centroid.y - y_min
    else:
        measured_x = result.measured_centroid.x
        measured_y = result.measured_centroid.y
    axes.scatter(
        measured_x,
        measured_y,
        marker="x",
        color="red",
        label="measured",
    )
    if result.requested_position is not None:
        if _image_extent(result) is None:
            requested_x = result.requested_position[0] - x_min
            requested_y = result.requested_position[1] - y_min
        else:
            requested_x, requested_y = result.requested_position
        axes.scatter(
            requested_x,
            requested_y,
            marker="+",
            color="cyan",
            label="requested",
        )
        axes.legend()
    return artist

"""Thin Matplotlib interfaces for quick-look results."""

from __future__ import annotations

from typing import Any

from .fibers import FiberTopology
from .workflows import PointingQuicklook, SpatialQuicklook


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
    image_artist = ax.imshow(result.image, origin="lower", interpolation="nearest")
    x_min = min(round(location.ifu_x) for location in topology.locations.values())
    y_min = min(round(location.ifu_y) for location in topology.locations.values())
    x = [location.ifu_x - x_min for location in topology.locations.values()]
    y = [location.ifu_y - y_min for location in topology.locations.values()]
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
    x_min = min(round(location.ifu_x) for location in topology.locations.values())
    y_min = min(round(location.ifu_y) for location in topology.locations.values())
    axes.scatter(
        result.measured_centroid.x - x_min,
        result.measured_centroid.y - y_min,
        marker="x",
        color="red",
        label="measured",
    )
    if result.requested_position is not None:
        axes.scatter(
            result.requested_position[0] - x_min,
            result.requested_position[1] - y_min,
            marker="+",
            color="cyan",
            label="requested",
        )
        axes.legend()
    return artist

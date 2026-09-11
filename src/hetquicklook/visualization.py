"""Small Matplotlib helpers for inspecting quick-look evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .fibers import FiberTopology
from .topology import VirusTopologyLoader
from .workflows import LRS2ChannelQuicklook, PointingQuicklook, SpatialQuicklook


QuicklookResult = SpatialQuicklook | LRS2ChannelQuicklook


def finite_limits(
    values: np.ndarray,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> tuple[float | None, float | None]:
    """Return finite display limits, or ``(None, None)`` for flat data."""

    data = np.asarray(values, dtype=float)
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None, None
    lo, hi = np.percentile(finite, percentiles)
    if lo == hi:
        return None, None
    return float(lo), float(hi)


def grid_extent(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> list[float]:
    """Return Matplotlib pixel-edge bounds for physical coordinate centers."""

    x = np.asarray(x_coordinates, dtype=float).ravel()
    y = np.asarray(y_coordinates, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("spatial coordinate arrays must not be empty")
    x_step = float(np.median(np.diff(x))) if x.size > 1 else 1.0
    y_step = float(np.median(np.diff(y))) if y.size > 1 else 1.0
    return [
        float(x[0] - x_step / 2.0),
        float(x[-1] + x_step / 2.0),
        float(y[0] - y_step / 2.0),
        float(y[-1] + y_step / 2.0),
    ]


def _image_extent(
    result: QuicklookResult,
) -> tuple[float, float, float, float] | None:
    """Return physical image bounds when the reconstruction supplied them."""

    x = result.spatial_x_coordinates
    y = result.spatial_y_coordinates
    if x is None or y is None or len(x) == 0 or len(y) == 0:
        return None
    return tuple(grid_extent(x, y))  # type: ignore[return-value]


def fiber_position_array(
    result: QuicklookResult,
    topology: FiberTopology | None = None,
) -> np.ndarray:
    """Return product fiber positions as an ``(nfiber, 2)`` array."""

    positions: Any = getattr(result, "fiber_positions", None)
    if isinstance(positions, Mapping):
        positions = list(positions.values())
        if not positions and topology is not None:
            positions = [
                (location.ifu_x, location.ifu_y)
                for location in topology.locations.values()
            ]
    elif positions is None and topology is not None:
        positions = [
            (location.ifu_x, location.ifu_y)
            for location in topology.locations.values()
        ]
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("fiber_positions must contain (x, y) pairs")
    return positions


def fiber_value_array(result: QuicklookResult) -> np.ndarray:
    """Return collapsed fiber values in their product order."""

    values: Any = result.fiber_values
    if isinstance(values, Mapping):
        values = list(values.values())
    return np.asarray(values, dtype=float).ravel()


def _spatial_coordinates(result: QuicklookResult) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(result.image)
    x = (
        np.arange(image.shape[1], dtype=float)
        if result.spatial_x_coordinates is None
        else np.asarray(result.spatial_x_coordinates, dtype=float)
    )
    y = (
        np.arange(image.shape[0], dtype=float)
        if result.spatial_y_coordinates is None
        else np.asarray(result.spatial_y_coordinates, dtype=float)
    )
    return x, y


def _plot_spatial_result(
    result: QuicklookResult,
    *,
    topology: FiberTopology | None,
    ax: Any,
    title: str | None,
    percentiles: tuple[float, float],
    show_fibers: bool,
    show_fiducial: bool,
    show_centroid: bool,
    colorbar: bool,
) -> Any:
    image = np.asarray(result.image, dtype=float)
    vmin, vmax = finite_limits(image, percentiles)
    kwargs: dict[str, Any] = {
        "origin": "lower",
        "aspect": "equal",
        "interpolation": "nearest",
    }
    extent = _image_extent(result)
    if extent is not None:
        kwargs["extent"] = extent
    if vmin is not None:
        kwargs.update(vmin=vmin, vmax=vmax)
    artist = ax.imshow(image, **kwargs)
    if colorbar:
        ax.figure.colorbar(artist, ax=ax, label="Collapsed quick-look signal")

    positions = fiber_position_array(result, topology) if show_fibers else None
    plot_positions = positions
    offset = np.zeros(2, dtype=float)
    if positions is not None and positions.size:
        if extent is None:
            offset = np.array(
                [min(round(position[0]) for position in positions),
                 min(round(position[1]) for position in positions)],
                dtype=float,
            )
            plot_positions = positions - offset
        ax.scatter(
            plot_positions[:, 0],
            plot_positions[:, 1],
            facecolors="none",
            edgecolors="white",
            linewidths=0.5,
            label="Fibers",
        )

    fiducial = getattr(result, "intended_fiducial", None)
    if show_fiducial and fiducial is not None:
        point = np.asarray(fiducial, dtype=float) - offset
        ax.scatter(
            [point[0]], [point[1]], marker="x", s=100, linewidths=2, label="Intended"
        )

    measured = getattr(result, "measured_centroid", None)
    if show_centroid and measured is not None:
        point = np.array([measured.x, measured.y], dtype=float) - offset
        if np.all(np.isfinite(point)):
            ax.scatter(
                [point[0]], [point[1]], marker="+", s=120, linewidths=2, label="Measured"
            )

    if len(ax.get_legend_handles_labels()[0]):
        ax.legend(loc="best", fontsize="small")
    ax.set_xlabel("IFU x [arcsec]")
    ax.set_ylabel("IFU y [arcsec]")
    if title:
        ax.set_title(title)
    return artist


def plot_spatial_image(
    result: QuicklookResult,
    *,
    topology: FiberTopology | None = None,
    ax: Any = None,
    title: str | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    show_fibers: bool = True,
    show_fiducial: bool = True,
    show_centroid: bool = True,
    colorbar: bool = True,
) -> Any:
    """Render a spatial image for an amplifier or LRS2 channel result."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    _plot_spatial_result(
        result,
        topology=topology,
        ax=ax,
        title=title,
        percentiles=percentiles,
        show_fibers=show_fibers,
        show_fiducial=show_fiducial,
        show_centroid=show_centroid,
        colorbar=colorbar,
    )
    ax.figure.tight_layout()
    return ax.figure


def plot_lrs2_channels(
    channels: Mapping[str, LRS2ChannelQuicklook],
    *,
    title: str | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    show_fibers: bool = True,
    show_fiducial: bool = False,
    show_centroid: bool = False,
) -> Any:
    """Render the four-position LRS2 channel layout.

    The four axes are kept even when evidence is partial. Missing channels
    are labelled in their expected position, while the display stretch is
    calculated from the channels that are present.
    """

    import matplotlib.pyplot as plt

    order = ("UV", "Orange", "Red", "Far-Red")
    finite_chunks = [
        np.asarray(channels[channel].image, dtype=float).ravel()
        for channel in order
        if channel in channels
    ]
    finite_values = (
        np.concatenate(finite_chunks) if finite_chunks else np.array([], dtype=float)
    )
    vmin, vmax = finite_limits(finite_values, percentiles)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, channel_name in zip(axes.flat, order):
        channel = channels.get(channel_name)
        if channel is None:
            ax.set_title(f"{channel_name} (missing)")
            ax.text(
                0.5,
                0.5,
                "No complete channel evidence",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue

        image = np.asarray(channel.image, dtype=float)
        kwargs: dict[str, Any] = {
            "origin": "lower",
            "aspect": "equal",
            "interpolation": "nearest",
        }
        if vmin is not None:
            kwargs.update(vmin=vmin, vmax=vmax)
        extent = grid_extent(
            np.asarray(channel.spatial_x_coordinates, dtype=float),
            np.asarray(channel.spatial_y_coordinates, dtype=float),
        )
        artist = ax.imshow(image, extent=extent, **kwargs)
        fig.colorbar(artist, ax=ax, fraction=0.046, pad=0.04, label="Collapsed signal")

        if show_fibers:
            positions = np.asarray(channel.fiber_positions, dtype=float)
            if positions.size:
                ax.scatter(
                    positions[:, 0],
                    positions[:, 1],
                    facecolors="none",
                    edgecolors="white",
                    s=10,
                    linewidths=0.35,
                    label="Fibers",
                )
        if show_fiducial and channel.intended_fiducial is not None:
            fx, fy = channel.intended_fiducial
            ax.scatter([fx], [fy], marker="x", s=90, linewidths=1.8, label="Intended")
        if show_centroid and channel.measured_centroid is not None:
            cx, cy = channel.measured_centroid.x, channel.measured_centroid.y
            if np.all(np.isfinite([cx, cy])):
                ax.scatter(
                    [cx], [cy], marker="+", s=110, linewidths=1.8, label="Measured"
                )
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="best", fontsize="small")
        ax.set_xlabel("IFU x [arcsec]")
        ax.set_ylabel("IFU y [arcsec]")
        ax.set_title(channel_name)

    if title:
        fig.suptitle(title)
    return fig


def plot_virus_ifu_amplifiers(
    amplifier_products: Mapping[str, SpatialQuicklook],
    *,
    ifu_slot: str | None = None,
    title: str | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    show_fibers: bool = True,
    show_fiducial: bool = False,
    show_centroid: bool = False,
) -> Any:
    """Render available amplifier products for one VIRUS IFU.

    This is an evidence view for one IFU, not a whole-VIRUS spatial
    reconstruction. Each amplifier retains its own detector-derived image.
    """

    import matplotlib.pyplot as plt

    slot = None if ifu_slot is None else str(ifu_slot).strip().zfill(3)
    order = ("LL", "LU", "RL", "RU")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    for ax, amplifier in zip(axes.flat, order):
        token = amplifier if amplifier in amplifier_products else None
        if token is None and slot is not None:
            token = f"{slot}{amplifier}"
        product = amplifier_products.get(token) if token is not None else None
        if product is None:
            ax.set_title(f"{amplifier} (missing)")
            ax.text(
                0.5,
                0.5,
                "No amplifier evidence",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_axis_off()
            continue
        _plot_spatial_result(
            product,
            topology=None,
            ax=ax,
            title=amplifier,
            percentiles=percentiles,
            show_fibers=show_fibers,
            show_fiducial=show_fiducial,
            show_centroid=show_centroid,
            colorbar=True,
        )
    if title:
        fig.suptitle(title)
    return fig


def plot_virus_ifu_grid(
    ifu_products: Mapping[str, SpatialQuicklook],
    *,
    title: str | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
    show_fibers: bool = True,
    show_fiducial: bool = False,
    show_centroid: bool = False,
) -> Any:
    """Render VIRUS IFU images in the authoritative focal-plane layout."""

    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt

    if not ifu_products:
        raise ValueError("ifu_products must contain at least one IFU")
    focal_plane, _ = VirusTopologyLoader().resolve_fplane()
    products = {
        str(slot).strip().zfill(3): product
        for slot, product in ifu_products.items()
    }
    unknown = sorted(set(products) - set(focal_plane))
    if unknown:
        raise ValueError(
            "VIRUS IFU products have no focal-plane positions: "
            + ", ".join(unknown)
        )

    positions = {
        slot: (float(position[0]), float(position[1]))
        for slot, position in focal_plane.items()
        if slot != "000"
    }
    x_coordinates = sorted({position[0] for position in positions.values()})
    y_coordinates = sorted(
        {position[1] for position in positions.values()}, reverse=True
    )
    slot_by_position = {position: slot for slot, position in positions.items()}

    image_values = [
        np.asarray(product.image, dtype=float).ravel()
        for product in products.values()
    ]
    vmin, vmax = finite_limits(np.concatenate(image_values), percentiles)
    fig, axes = plt.subplots(
        len(y_coordinates),
        len(x_coordinates),
        figsize=(16, 16),
        squeeze=False,
    )
    fig.subplots_adjust(
        left=0.03,
        right=0.97,
        bottom=0.03,
        top=0.95,
        wspace=0.02,
        hspace=0.02,
    )
    artist = None
    for row_index, y_coordinate in enumerate(y_coordinates):
        for column_index, x_coordinate in enumerate(x_coordinates):
            ax = axes[row_index, column_index]
            slot = slot_by_position.get((x_coordinate, y_coordinate))
            ax.set_xticks([])
            ax.set_yticks([])
            if slot is None:
                ax.set_axis_off()
                continue
            ax.set_title(
                slot,
                fontsize="small",
                pad=2,
                color="white",
                path_effects=[
                    pe.Stroke(linewidth=2.5, foreground="black"),
                    pe.Normal(),
                ],
                bbox={"facecolor": "black", "edgecolor": "none", "pad": 1.5},
            )
            product = products.get(slot)
            if product is None:
                ax.text(
                    0.5,
                    0.5,
                    "unavailable",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize="x-small",
                )
                ax.set_facecolor("0.85")
                continue

            image = np.asarray(product.image, dtype=float)
            kwargs: dict[str, Any] = {
                "origin": "lower",
                "aspect": "equal",
                "interpolation": "nearest",
            }
            extent = _image_extent(product)
            if extent is not None:
                kwargs["extent"] = extent
            if vmin is not None:
                kwargs.update(vmin=vmin, vmax=vmax)
            artist = ax.imshow(image, **kwargs)

            if show_fibers:
                fiber_positions = fiber_position_array(product)
                if fiber_positions.size:
                    ax.scatter(
                        fiber_positions[:, 0],
                        fiber_positions[:, 1],
                        facecolors="none",
                        edgecolors="white",
                        s=3,
                        linewidths=0.25,
                    )
            if show_fiducial and product.intended_fiducial is not None:
                fx, fy = product.intended_fiducial
                ax.scatter([fx], [fy], marker="x", s=30, linewidths=1.0)
            if show_centroid and product.measured_centroid is not None:
                cx, cy = product.measured_centroid.x, product.measured_centroid.y
                if np.all(np.isfinite([cx, cy])):
                    ax.scatter([cx], [cy], marker="+", s=40, linewidths=1.0)

    if artist is not None and vmin is not None:
        fig.colorbar(
            artist,
            ax=axes.ravel().tolist(),
            fraction=0.02,
            pad=0.01,
            label="Collapsed quick-look signal",
        )
    if title:
        fig.suptitle(title)
    return fig


def plot_fiber_values(
    result: QuicklookResult,
    *,
    topology: FiberTopology | None = None,
    ax: Any = None,
    title: str | None = None,
    percentiles: tuple[float, float] = (2.0, 98.0),
) -> Any:
    """Render collapsed fiber values at their physical positions."""

    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    positions = fiber_position_array(result, topology)
    values = fiber_value_array(result)
    if positions.shape[0] != values.size:
        raise ValueError("fiber positions and values must have matching lengths")
    vmin, vmax = finite_limits(values, percentiles)
    kwargs: dict[str, Any] = {"s": 55}
    if vmin is not None:
        kwargs.update(vmin=vmin, vmax=vmax)
    artist = ax.scatter(positions[:, 0], positions[:, 1], c=values, **kwargs)
    ax.figure.colorbar(artist, ax=ax, label="Collapsed quick-look signal")
    ax.set_aspect("equal")
    ax.set_xlabel("IFU x [arcsec]")
    ax.set_ylabel("IFU y [arcsec]")
    ax.set_title(title or "Fiber-level evidence")
    ax.figure.tight_layout()
    return ax.figure


def plot_spatial_support(
    result: QuicklookResult,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Render Gaussian support evidence for a spatial product."""

    import matplotlib.pyplot as plt

    if result.spatial_support is None:
        raise ValueError("result does not contain spatial support")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    x, y = _spatial_coordinates(result)
    artist = ax.imshow(
        np.asarray(result.spatial_support),
        origin="lower",
        aspect="equal",
        extent=grid_extent(x, y),
    )
    ax.figure.colorbar(artist, ax=ax, label="Gaussian support")
    ax.set_xlabel("IFU x [arcsec]")
    ax.set_ylabel("IFU y [arcsec]")
    ax.set_title(title or "Spatial support")
    ax.figure.tight_layout()
    return ax.figure


def plot_spatial_quicklook(
    result: SpatialQuicklook,
    topology: FiberTopology,
    *,
    ax: Any = None,
    title: str | None = None,
) -> Any:
    """Plot an amplifier spatial result and its fiber positions."""

    if ax is None:
        ax = _new_axes()
    return _plot_spatial_result(
        result,
        topology=topology,
        ax=ax,
        title=title,
        percentiles=(2.0, 98.0),
        show_fibers=True,
        show_fiducial=False,
        show_centroid=False,
        colorbar=False,
    )


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
    extent = _image_extent(result)
    positions = fiber_position_array(result, topology)
    offset = np.zeros(2, dtype=float)
    if extent is None and positions.size:
        offset = np.array(
            [min(round(position[0]) for position in positions),
             min(round(position[1]) for position in positions)],
            dtype=float,
        )
    measured = np.array(
        [result.measured_centroid.x, result.measured_centroid.y], dtype=float
    ) - offset
    axes.scatter(
        measured[0],
        measured[1],
        marker="x",
        color="red",
        label="measured",
    )
    if result.requested_position is not None:
        requested = np.asarray(result.requested_position, dtype=float) - offset
        axes.scatter(
            requested[0],
            requested[1],
            marker="+",
            color="cyan",
            label="requested",
        )
        axes.legend()
    return artist


def _new_axes() -> Any:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    return ax

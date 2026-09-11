import matplotlib

matplotlib.use("Agg")

import numpy as np

from hetquicklook.visualization import (
    fiber_position_array,
    fiber_value_array,
    finite_limits,
    grid_extent,
    plot_lrs2_channels,
    plot_virus_ifu_grid,
    plot_fiber_values,
    plot_spatial_image,
    plot_spatial_support,
)
from hetquicklook.workflows import LRS2ChannelQuicklook, SpatialQuicklook


def _channel_result() -> LRS2ChannelQuicklook:
    positions = np.array([[0.0, 0.0], [0.4, 0.0]])
    return LRS2ChannelQuicklook(
        channel="UV",
        amplifier_products={},
        fiber_positions=positions,
        fiber_values=np.array([1.0, 2.0]),
        image=np.ones((2, 2)),
        spatial_x_coordinates=np.array([-0.2, 0.2]),
        spatial_y_coordinates=np.array([-0.2, 0.2]),
        spatial_support=np.ones((2, 2), dtype=bool),
        spatial_weight=np.ones((2, 2)),
        intended_fiducial=(0.0, 0.0),
        measured_centroid=None,
        spatial_gaussian_fwhm_arcsec=1.2,
        spatial_pixel_scale_arcsec=0.4,
    )


def test_visualization_adapters_accept_mapping_and_array_products() -> None:
    channel = _channel_result()
    np.testing.assert_array_equal(fiber_position_array(channel), channel.fiber_positions)
    np.testing.assert_array_equal(fiber_value_array(channel), channel.fiber_values)

    amplifier = SpatialQuicklook(
        fiber_values={"a": 3.0},
        fiber_positions={"a": (1.0, 2.0)},
        image=np.ones((1, 1)),
    )
    np.testing.assert_array_equal(fiber_position_array(amplifier), [[1.0, 2.0]])
    np.testing.assert_array_equal(fiber_value_array(amplifier), [3.0])


def test_visualization_helpers_render_channel_and_support_evidence() -> None:
    channel = _channel_result()
    image_figure = plot_spatial_image(channel, show_fibers=True)
    values_figure = plot_fiber_values(channel)
    support_figure = plot_spatial_support(channel)
    assert len(image_figure.axes) >= 1
    assert len(values_figure.axes) >= 1
    assert len(support_figure.axes) >= 1


def test_display_helpers_handle_finite_limits_and_physical_grid_edges() -> None:
    assert finite_limits(np.array([np.nan, 1.0, 3.0])) == (1.04, 2.96)
    assert grid_extent(np.array([-0.2, 0.2]), np.array([-0.2, 0.2])) == [
        -0.4, 0.4, -0.4, 0.4
    ]


def test_lrs2_channel_layout_keeps_missing_panels_visible() -> None:
    channel = _channel_result()
    channels = {
        name: channel
        for name in ("UV", "Orange", "Red", "Far-Red")
    }
    channels.pop("Orange")

    figure = plot_lrs2_channels(channels, show_fibers=False)

    assert len(figure.axes) >= 4
    assert any("Orange (missing)" in axis.get_title() for axis in figure.axes)


def test_virus_ifu_grid_uses_focal_plane_slots() -> None:
    products = {
        slot: SpatialQuicklook(
            fiber_values={"fiber": 1.0},
            fiber_positions={"fiber": (0.0, 0.0)},
            image=np.ones((2, 2)),
            spatial_x_coordinates=np.array([-0.5, 0.5]),
            spatial_y_coordinates=np.array([-0.5, 0.5]),
        )
        for slot in ("074", "075")
    }

    figure = plot_virus_ifu_grid(products, show_fibers=False)

    image_axes = [axis for axis in figure.axes if axis.get_title() in {"074", "075"}]
    assert len(image_axes) == 2
    assert len(figure.axes) >= 100
    title = image_axes[0].title
    assert title.get_color() == "white"
    assert len(title.get_path_effects()) == 2
    assert title.get_bbox_patch().get_facecolor()[:3] == (0.0, 0.0, 0.0)

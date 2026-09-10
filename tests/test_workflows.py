import numpy as np
import pytest

from hetquicklook.fibers import FiberTopology
from hetquicklook.workflows import (
    LRS2ChannelQuicklook,
    LRS2_SPATIAL_DEFAULTS,
    PointingQuicklook,
    SpatialQuicklook,
    VIRUS_SPATIAL_DEFAULTS,
    combine_lrs2_channel_products,
    combine_lrs2_channels,
    run_ldls_flat_quicklook,
    run_standard_star_quicklook,
)
from hetquicklook.algorithms.centroid import Centroid


def test_standard_star_workflow_returns_pointing_offset() -> None:
    topology = FiberTopology.from_arrays(
        "amp-a",
        ("left", "center", "right"),
        np.array([[1], [2], [3]]),
        np.array([[1], [1], [1]]),
        np.array([0, 1, 2]),
        np.array([0, 0, 0]),
    )
    detector = np.zeros((3, 4), dtype=float)
    detector[1, 2] = 10
    detector[1, 3] = 10

    result = run_standard_star_quicklook(
        detector, topology, requested_position=(1.0, 0.0)
    )

    assert result.measured_centroid.x == 1.5
    assert result.offset == (0.5, 0.0)


def test_ldls_workflow_produces_spatial_image() -> None:
    topology = FiberTopology.from_arrays(
        "amp-a",
        ("a", "b"),
        np.array([[0], [1]]),
        np.array([[0], [0]]),
        np.array([4, 5]),
        np.array([6, 6]),
    )
    result = run_ldls_flat_quicklook(np.array([[2.0, 3.0]]), topology)

    assert result.image.shape == (1, 2)
    assert result.image.tolist() == [[2.0, 3.0]]


def _dense_topology() -> FiberTopology:
    detector_columns = 220
    fiber_ids = ("f0", "f1", "f2")
    detector_x = np.tile(np.arange(detector_columns, dtype=float), (3, 1))
    detector_y = np.repeat(np.array([[10.0], [15.0], [20.0]]), detector_columns, axis=1)
    return FiberTopology.from_arrays(
        "amp-a",
        fiber_ids,
        detector_x,
        detector_y,
        np.array([-1.0, 0.0, 1.0]),
        np.zeros(3),
    )


def test_workflow_selects_instrument_spatial_defaults_and_pads_bounds() -> None:
    detector = np.ones((30, 220), dtype=float)
    virus = run_ldls_flat_quicklook(
        detector, _dense_topology(), instrument="virus"
    )
    lrs2 = run_ldls_flat_quicklook(
        detector, _dense_topology(), instrument="lrs2"
    )

    assert VIRUS_SPATIAL_DEFAULTS["gaussian_fwhm_arcsec"] == 1.5
    assert VIRUS_SPATIAL_DEFAULTS["pixel_scale_arcsec"] == 1.0
    assert LRS2_SPATIAL_DEFAULTS["gaussian_fwhm_arcsec"] == 1.2
    assert LRS2_SPATIAL_DEFAULTS["pixel_scale_arcsec"] == 0.4
    assert virus.spatial_gaussian_fwhm_arcsec == 1.5
    assert virus.spatial_pixel_scale_arcsec == 1.0
    assert lrs2.spatial_gaussian_fwhm_arcsec == 1.2
    assert lrs2.spatial_pixel_scale_arcsec == 0.4
    assert np.diff(virus.spatial_x_coordinates).tolist() == [1.0] * (
        virus.spatial_x_coordinates.size - 1
    )
    np.testing.assert_allclose(np.diff(lrs2.spatial_x_coordinates), 0.4)
    assert not virus.spatial_support[:, 0].any()
    assert not virus.spatial_support[:, -1].any()
    assert not lrs2.spatial_support[:, 0].any()
    assert not lrs2.spatial_support[:, -1].any()


def test_workflow_spatial_overrides_are_passed_to_generic_splat() -> None:
    result = run_ldls_flat_quicklook(
        np.ones((30, 220), dtype=float),
        _dense_topology(),
        instrument="lrs2",
        gaussian_fwhm=2.0,
        pixel_scale=0.5,
    )

    assert result.spatial_gaussian_fwhm_arcsec == 2.0
    assert result.spatial_pixel_scale_arcsec == 0.5
    np.testing.assert_allclose(np.diff(result.spatial_x_coordinates), 0.5)


def test_standard_workflow_exposes_both_fiducials_without_rejection() -> None:
    detector = np.ones((30, 220), dtype=float)
    virus = run_standard_star_quicklook(
        detector, _dense_topology(), instrument="virus"
    )
    lrs2 = run_standard_star_quicklook(
        detector, _dense_topology(), instrument="lrs2"
    )

    assert virus.intended_fiducial == (0.0, 0.0)
    assert virus.requested_position == (0.0, 0.0)
    assert lrs2.intended_fiducial == (0.0, 0.0)
    assert lrs2.requested_position == (0.0, 0.0)
    assert virus.offset is not None
    assert lrs2.offset is not None


def _lrs2_amplifier_product(
    token: str,
    *,
    value: float = 1.0,
    x_position: float = 0.0,
    pointing: bool = False,
) -> SpatialQuicklook:
    fiber_ids = tuple(f"{token}-{index:03d}" for index in range(140))
    positions = {
        fiber_id: (x_position, float(index % 14) * 0.1)
        for index, fiber_id in enumerate(fiber_ids)
    }
    values = {fiber_id: value for fiber_id in fiber_ids}
    product = SpatialQuicklook(
        fiber_values=values,
        image=np.zeros((3, 3)),
        instrument="lrs2",
        fiber_positions=positions,
        intended_fiducial=(0.0, 0.0),
    )
    if not pointing:
        return product
    return PointingQuicklook(
        fiber_values=product.fiber_values,
        image=product.image,
        instrument=product.instrument,
        fiber_positions=product.fiber_positions,
        intended_fiducial=product.intended_fiducial,
        measured_centroid=Centroid(99.0, 99.0, 1.0, 1),
        requested_position=(0.0, 0.0),
    )


def test_lrs2_channel_combination_uses_authoritative_pairs_and_280_fibers() -> None:
    products = {
        "056LL": _lrs2_amplifier_product("056LL", value=1.0),
        "056LU": _lrs2_amplifier_product("056LU", value=2.0),
    }

    result = combine_lrs2_channel_products("UV", products)

    assert isinstance(result, LRS2ChannelQuicklook)
    assert result.channel == "UV"
    assert tuple(result.amplifier_products) == ("056LL", "056LU")
    assert result.fiber_positions.shape == (280, 2)
    assert result.fiber_values.shape == (280,)
    assert result.fiber_values[:140].tolist() == [1.0] * 140
    assert result.fiber_values[140:].tolist() == [2.0] * 140
    assert result.spatial_gaussian_fwhm_arcsec == 1.2
    assert result.spatial_pixel_scale_arcsec == 0.4
    assert result.intended_fiducial == (0.0, 0.0)


def test_all_lrs2_channels_combine_into_four_280_fiber_products() -> None:
    tokens = (
        "056LL", "056LU", "056RL", "056RU",
        "066LL", "066LU", "066RL", "066RU",
    )
    products = {token: _lrs2_amplifier_product(token) for token in tokens}

    channels = combine_lrs2_channels(products)

    assert tuple(channels) == ("UV", "Orange", "Red", "Far-Red")
    assert all(result.fiber_positions.shape == (280, 2) for result in channels.values())
    assert all(result.fiber_values.shape == (280,) for result in channels.values())


def test_lrs2_channel_combination_preserves_position_value_correspondence() -> None:
    first = _lrs2_amplifier_product("056LL", value=10.0, x_position=-2.0)
    second = _lrs2_amplifier_product("056LU", value=20.0, x_position=3.0)

    result = combine_lrs2_channel_products(
        "UV", {"056LL": first, "056LU": second}
    )

    np.testing.assert_allclose(result.fiber_positions[:140, 0], -2.0)
    np.testing.assert_allclose(result.fiber_positions[140:, 0], 3.0)
    np.testing.assert_allclose(result.fiber_values[:140], 10.0)
    np.testing.assert_allclose(result.fiber_values[140:], 20.0)


def test_lrs2_channel_centroid_is_recomputed_from_all_fibers() -> None:
    first = _lrs2_amplifier_product(
        "056LL", value=1.0, x_position=-1.0, pointing=True
    )
    second = _lrs2_amplifier_product(
        "056LU", value=1.0, x_position=3.0, pointing=True
    )

    result = combine_lrs2_channel_products(
        "UV", {"056LL": first, "056LU": second}
    )

    assert result.measured_centroid is not None
    assert result.measured_centroid.x == 1.0
    assert result.measured_centroid.y == pytest.approx(0.65)
    assert result.offset == pytest.approx((1.0, 0.65))


def test_lrs2_channel_spatial_overrides_are_explicit() -> None:
    products = {
        "056LL": _lrs2_amplifier_product("056LL"),
        "056LU": _lrs2_amplifier_product("056LU"),
    }

    result = combine_lrs2_channel_products(
        "UV",
        products,
        gaussian_fwhm_arcsec=2.0,
        pixel_scale_arcsec=0.5,
    )

    assert result.spatial_gaussian_fwhm_arcsec == 2.0
    assert result.spatial_pixel_scale_arcsec == 0.5
    np.testing.assert_allclose(np.diff(result.spatial_x_coordinates), 0.5)


def test_lrs2_channel_does_not_accept_missing_or_wrong_amplifier() -> None:
    product = _lrs2_amplifier_product("056LL")
    with pytest.raises(ValueError, match="missing amplifier"):
        combine_lrs2_channel_products("UV", {"056LL": product})
    with pytest.raises(ValueError, match="does not belong"):
        combine_lrs2_channel_products(
            "UV",
            {"056LL": product, "056RL": _lrs2_amplifier_product("056RL")},
        )


def test_lrs2_channel_set_does_not_silently_create_partial_products() -> None:
    products = {
        "056LL": _lrs2_amplifier_product("056LL"),
        "056LU": _lrs2_amplifier_product("056LU"),
    }

    with pytest.raises(ValueError, match="missing amplifier"):
        combine_lrs2_channels(products)

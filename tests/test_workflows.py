import numpy as np

from hetquicklook.fibers import FiberTopology
from hetquicklook.workflows import (
    LRS2_SPATIAL_DEFAULTS,
    VIRUS_SPATIAL_DEFAULTS,
    run_ldls_flat_quicklook,
    run_standard_star_quicklook,
)


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

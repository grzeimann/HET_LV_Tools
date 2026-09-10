import numpy as np

from hetquicklook.fibers import FiberTopology
from hetquicklook.workflows import (
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

import numpy as np

from hetquicklook.algorithms.centroid import centroid_2d
from hetquicklook.algorithms.collapse import (
    collapse_fiber_signal,
    collapse_fibers,
    spatial_image_from_fiber_values,
)
from hetquicklook.algorithms.extraction import extract_trace
from hetquicklook.fibers import FiberTopology


def _topology() -> FiberTopology:
    return FiberTopology.from_arrays(
        "amp-a",
        ("f0", "f1", "f2"),
        np.array([[1, 1], [2, 2], [3, 3]]),
        np.array([[1, 2], [1, 2], [1, 2]]),
        np.array([0, 1, 2]),
        np.array([0, 0, 0]),
    )


def test_extract_and_collapse_ignore_nonfinite_values() -> None:
    detector = np.arange(25, dtype=float).reshape(5, 5)
    detector[1, 1] = np.nan
    trace = _topology().traces["f0"]

    extracted = extract_trace(detector, trace)

    assert np.isnan(extracted[0])
    assert collapse_fiber_signal(extracted) == 11.0


def test_collapse_and_spatial_mapping() -> None:
    detector = np.ones((5, 5), dtype=float)
    detector[1:3, 2] = 5.0
    topology = _topology()

    values = collapse_fibers(detector, topology)
    image = spatial_image_from_fiber_values(values, topology)

    assert values == {"f0": 1.0, "f1": 5.0, "f2": 1.0}
    assert image.shape == (1, 3)
    assert image.tolist() == [[1.0, 5.0, 1.0]]


def test_centroid_2d_uses_array_coordinates() -> None:
    image = np.zeros((3, 4))
    image[1, 2] = 10
    centroid = centroid_2d(image)

    assert (centroid.x, centroid.y, centroid.n_used) == (2.0, 1.0, 1)


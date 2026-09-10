import numpy as np
import pytest

from hetquicklook.fibers import FiberTopology


def test_fiber_topology_from_arrays() -> None:
    topology = FiberTopology.from_arrays(
        "amp-a",
        ("f0", "f1"),
        np.array([[1, 1], [3, 3]]),
        np.array([[1, 2], [1, 2]]),
        np.array([0, 1]),
        np.array([0, 0]),
    )

    assert topology.fiber_ids == ("f0", "f1")
    assert topology.traces["f1"].detector_x.tolist() == [3.0, 3.0]


def test_fiber_topology_requires_matching_trace_and_location_ids() -> None:
    with pytest.raises(ValueError):
        FiberTopology("amp-a", {}, {"f0": object()})


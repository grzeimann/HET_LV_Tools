"""Amplifier-level fiber traces and IFU-plane positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class FiberTrace:
    """Detector samples belonging to one fiber trace.

    ``detector_x`` and ``detector_y`` are one-dimensional arrays of equal
    length in detector pixel coordinates. The arrays need not describe a
    straight trace.
    """

    fiber_id: str
    detector_x: np.ndarray
    detector_y: np.ndarray

    def __post_init__(self) -> None:
        x = np.asarray(self.detector_x)
        y = np.asarray(self.detector_y)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError("Fiber trace coordinates must be equal-length 1-D arrays")
        if x.size == 0:
            raise ValueError("Fiber trace must contain at least one detector sample")
        object.__setattr__(self, "detector_x", x.astype(float, copy=False))
        object.__setattr__(self, "detector_y", y.astype(float, copy=False))


@dataclass(frozen=True)
class FiberLocation:
    """Physical IFU-plane position for one fiber.

    Coordinates are in the caller's IFU-plane coordinate system. This initial
    layer does not assume a sky-coordinate transform or a particular unit.
    """

    fiber_id: str
    ifu_x: float
    ifu_y: float


@dataclass(frozen=True)
class FiberTopology:
    """Fiber traces and IFU locations for one amplifier."""

    amplifier: str
    traces: Mapping[str, FiberTrace]
    locations: Mapping[str, FiberLocation]

    def __post_init__(self) -> None:
        trace_ids = set(self.traces)
        location_ids = set(self.locations)
        if trace_ids != location_ids:
            raise ValueError("Every fiber must have both a trace and an IFU location")
        for fiber_id, trace in self.traces.items():
            if trace.fiber_id != fiber_id:
                raise ValueError(f"Trace key {fiber_id!r} does not match its fiber ID")
        for fiber_id, location in self.locations.items():
            if location.fiber_id != fiber_id:
                raise ValueError(f"Location key {fiber_id!r} does not match its fiber ID")

    @property
    def fiber_ids(self) -> tuple[str, ...]:
        """Return fiber IDs in deterministic order."""

        return tuple(sorted(self.traces))

    @classmethod
    def from_arrays(
        cls,
        amplifier: str,
        fiber_ids: list[str] | tuple[str, ...],
        detector_x: np.ndarray,
        detector_y: np.ndarray,
        ifu_x: np.ndarray,
        ifu_y: np.ndarray,
    ) -> "FiberTopology":
        """Build a topology from rectangular arrays.

        Args:
            detector_x: Array with shape ``(n_fibers, n_samples)``.
            detector_y: Array with shape ``(n_fibers, n_samples)``.
            ifu_x: Array with shape ``(n_fibers,)``.
            ifu_y: Array with shape ``(n_fibers,)``.
        """

        identifiers = tuple(fiber_ids)
        x = np.asarray(detector_x)
        y = np.asarray(detector_y)
        px = np.asarray(ifu_x)
        py = np.asarray(ifu_y)
        expected = (len(identifiers),)
        if x.ndim != 2 or y.shape != x.shape or px.shape != expected or py.shape != expected:
            raise ValueError("Fiber arrays have incompatible shapes")
        traces = {
            fiber_id: FiberTrace(fiber_id, x[index], y[index])
            for index, fiber_id in enumerate(identifiers)
        }
        locations = {
            fiber_id: FiberLocation(fiber_id, float(px[index]), float(py[index]))
            for index, fiber_id in enumerate(identifiers)
        }
        return cls(amplifier=amplifier, traces=traces, locations=locations)


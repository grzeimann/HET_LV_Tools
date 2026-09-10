"""Small result containers shared by the numerical quick-look algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AlgorithmResult:
    """Named numerical arrays and scalar evidence from one algorithm.

    The ``get_array`` method mirrors the supplied VIRUSFlow result boundary,
    while named attributes on the specialized results below make common
    quick-look use concise.
    """

    kind: str
    version: str
    arrays: Mapping[str, np.ndarray]
    scalars: Mapping[str, Any]
    metadata: Mapping[str, Any] | None = None

    def get_array(self, name: str) -> np.ndarray:
        """Return a named output array."""

        value = self.arrays[name]
        if value is None:
            raise KeyError(name)
        return value

    @property
    def meta(self) -> Mapping[str, Any] | None:
        """Compatibility alias for result metadata."""

        return self.metadata


@dataclass(frozen=True)
class SpatialReconstructionResult:
    """Image, support, and optional propagated variance from a Gaussian splat."""

    image: np.ndarray
    weight: np.ndarray
    support: np.ndarray
    variance: np.ndarray | None
    x_coordinates: np.ndarray
    y_coordinates: np.ndarray
    contribution_count: np.ndarray

    @property
    def error(self) -> np.ndarray | None:
        """Return propagated one-sigma error when errors were supplied."""

        if self.variance is None:
            return None
        return np.sqrt(self.variance)

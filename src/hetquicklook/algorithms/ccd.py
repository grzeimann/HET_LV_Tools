"""Compatibility name for the shared CCD preparation algorithms.

The implementation lives in :mod:`hetquicklook.algorithms.detector`; this
module keeps the supplied reference module's ``ccd`` import boundary without
creating a second detector implementation.
"""

from .detector import (
    DEFAULT_GAIN,
    DEFAULT_READ_NOISE,
    DETECTOR_REDUCTION_VERSION,
    orient_amplifier_image,
    prepare_detector,
    reduce_amplifier_array,
)

__all__ = [
    "DEFAULT_GAIN",
    "DEFAULT_READ_NOISE",
    "DETECTOR_REDUCTION_VERSION",
    "orient_amplifier_image",
    "prepare_detector",
    "reduce_amplifier_array",
]

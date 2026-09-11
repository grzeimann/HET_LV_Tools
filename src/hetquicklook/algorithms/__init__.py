"""Small numerical operations used by quick-look workflows."""

from .centroid import Centroid, centroid_2d, weighted_centroid
from .collapse import (
    DEFAULT_COLLAPSE_COLUMNS,
    DEFAULT_COLLAPSE_STATISTIC,
    collapse_fiber_signal,
    collapse_fibers,
    collapse_extracted_spectra,
    collapse_fiber_spectra,
    collapse_spectra,
    central_column_bounds,
    select_central_columns,
    spatial_image_from_fiber_values,
)
from .detector import (
    DEFAULT_GAIN,
    DEFAULT_READ_NOISE,
    orient_amplifier_image,
    prepare_detector,
    reduce_amplifier_array,
)
from .extraction import (
    EXTRACTION_VERSION,
    extract_fractional_aperture,
    extract_trace,
    fractional_aperture_geometry,
)
from .results import AlgorithmResult, SpatialReconstructionResult
from .spatial import (
    DEFAULT_GAUSSIAN_FWHM_ARCSEC,
    DEFAULT_PIXEL_SCALE_ARCSEC,
    gaussian_splat,
    reconstruct_spatial_image,
)
from .trace import build_trace_map, fit_fiber_traces, robust_polyfit_predict

__all__ = [
    "Centroid",
    "AlgorithmResult",
    "SpatialReconstructionResult",
    "DEFAULT_COLLAPSE_COLUMNS",
    "DEFAULT_COLLAPSE_STATISTIC",
    "DEFAULT_GAIN",
    "DEFAULT_READ_NOISE",
    "DEFAULT_GAUSSIAN_FWHM_ARCSEC",
    "DEFAULT_PIXEL_SCALE_ARCSEC",
    "EXTRACTION_VERSION",
    "centroid_2d",
    "collapse_fiber_signal",
    "collapse_fibers",
    "collapse_extracted_spectra",
    "collapse_fiber_spectra",
    "collapse_spectra",
    "central_column_bounds",
    "select_central_columns",
    "extract_fractional_aperture",
    "extract_trace",
    "fractional_aperture_geometry",
    "fit_fiber_traces",
    "build_trace_map",
    "gaussian_splat",
    "reconstruct_spatial_image",
    "orient_amplifier_image",
    "prepare_detector",
    "reduce_amplifier_array",
    "robust_polyfit_predict",
    "spatial_image_from_fiber_values",
    "weighted_centroid",
]

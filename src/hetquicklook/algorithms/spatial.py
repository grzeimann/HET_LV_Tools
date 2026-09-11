"""Wavelength-independent Gaussian-splat spatial reconstruction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .results import SpatialReconstructionResult


# These values keep direct low-level array calls backwards compatible. The
# quick-look workflows resolve instrument-specific values before calling this
# instrument-agnostic routine.
DEFAULT_GAUSSIAN_FWHM_ARCSEC = 1.8
DEFAULT_PIXEL_SCALE_ARCSEC = 1.0
DEFAULT_SUPPORT_SIGMA = 2.0


def _coordinate_grid(
    minimum: float,
    maximum: float,
    pixel_scale: float,
) -> np.ndarray:
    """Create an inclusive grid aligned to a lower coordinate bound."""

    lower = np.floor(minimum / pixel_scale) * pixel_scale
    upper = np.ceil(maximum / pixel_scale) * pixel_scale
    count = int(np.rint((upper - lower) / pixel_scale)) + 1
    return lower + np.arange(max(1, count), dtype=float) * pixel_scale


def _requested_grid(
    positions: np.ndarray,
    *,
    pixel_scale: float,
    output_shape: tuple[int, int] | None,
    origin: tuple[float, float] | None,
    x_bounds: Sequence[float] | None,
    y_bounds: Sequence[float] | None,
    padding: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resolve output coordinate arrays from explicit bounds or positions.

    Automatically inferred bounds include the requested padding around the
    authoritative fiber-coordinate extent. Explicit bounds remain an
    intentional override and are used as supplied.
    """

    finite_positions = positions[np.all(np.isfinite(positions), axis=1)]
    if output_shape is not None:
        if len(output_shape) != 2 or any(int(value) <= 0 for value in output_shape):
            raise ValueError("output_shape must contain two positive integers")
        if origin is None:
            origin = (0.0, 0.0)
        x = origin[0] + np.arange(int(output_shape[1]), dtype=float) * pixel_scale
        y = origin[1] + np.arange(int(output_shape[0]), dtype=float) * pixel_scale
        return x, y

    if x_bounds is None:
        if finite_positions.size == 0:
            raise ValueError("positions must contain finite coordinates when no grid is supplied")
        x = _coordinate_grid(
            float(np.min(finite_positions[:, 0])) - padding,
            float(np.max(finite_positions[:, 0])) + padding,
            pixel_scale,
        )
    else:
        if len(x_bounds) != 2 or not np.all(np.isfinite(x_bounds)):
            raise ValueError("x_bounds must contain two finite coordinates")
        x = _coordinate_grid(float(min(x_bounds)), float(max(x_bounds)), pixel_scale)
    if y_bounds is None:
        if finite_positions.size == 0:
            raise ValueError("positions must contain finite coordinates when no grid is supplied")
        y = _coordinate_grid(
            float(np.min(finite_positions[:, 1])) - padding,
            float(np.max(finite_positions[:, 1])) + padding,
            pixel_scale,
        )
    else:
        if len(y_bounds) != 2 or not np.all(np.isfinite(y_bounds)):
            raise ValueError("y_bounds must contain two finite coordinates")
        y = _coordinate_grid(float(min(y_bounds)), float(max(y_bounds)), pixel_scale)
    return x, y


def gaussian_splat(
    fiber_positions: np.ndarray,
    fiber_values: np.ndarray,
    errors: np.ndarray | None = None,
    *,
    fwhm: float = DEFAULT_GAUSSIAN_FWHM_ARCSEC,
    pixel_scale: float = DEFAULT_PIXEL_SCALE_ARCSEC,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
    x_bounds: Sequence[float] | None = None,
    y_bounds: Sequence[float] | None = None,
    support_sigma: float = DEFAULT_SUPPORT_SIGMA,
    grid_padding: float | None = None,
    fill_value: float = float("nan"),
) -> SpatialReconstructionResult:
    """Reconstruct an IFU-plane image by locally splatting Gaussian fibers.

    Args:
        fiber_positions: Physical ``(x, y)`` positions with shape ``(nfiber, 2)``.
        fiber_values: One scalar value per fiber.
        errors: Optional one-sigma error per fiber. Formal image variance is
            returned only at pixels whose contributing fibers all have finite
            errors.
        fwhm: Gaussian FWHM in the coordinate units of the supplied positions.
        pixel_scale: Output coordinate units per image pixel.
        output_shape/origin: Optional explicit ``(ny, nx)`` grid and its lower
            coordinate origin. Alternatively, bounds can be supplied. With no
            grid arguments, bounds are inferred from finite fiber positions.
        support_sigma: Radius in Gaussian sigma used to identify image support.
        grid_padding: Optional coordinate padding used only when inferring the
            grid from fiber positions. When omitted, the Gaussian support
            radius is used for backwards compatibility. This is independent
            of ``support_sigma`` so the output extent can be reduced without
            changing the Gaussian reconstruction kernel.

    Returns:
        A normalized weighted image, accumulated Gaussian weight, explicit
        support mask, contribution count, and optional propagated variance.
    """

    positions = np.asarray(fiber_positions, dtype=float)
    values = np.asarray(fiber_values, dtype=float).ravel()
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("fiber_positions must have shape (nfiber, 2)")
    if values.shape != (positions.shape[0],):
        raise ValueError("fiber_values must contain one value per position")
    if errors is not None:
        error_values = np.asarray(errors, dtype=float).ravel()
        if error_values.shape != values.shape:
            raise ValueError("errors must contain one value per position")
    else:
        error_values = None
    if not np.isfinite(fwhm) or fwhm <= 0.0:
        raise ValueError("fwhm must be finite and positive")
    if not np.isfinite(pixel_scale) or pixel_scale <= 0.0:
        raise ValueError("pixel_scale must be finite and positive")
    if not np.isfinite(support_sigma) or support_sigma <= 0.0:
        raise ValueError("support_sigma must be finite and positive")

    sigma_coordinate = float(fwhm) / 2.35
    sigma_pixels = sigma_coordinate / float(pixel_scale)
    support_radius = float(support_sigma) * sigma_coordinate
    if grid_padding is None:
        resolved_grid_padding = support_radius
    else:
        if not np.isfinite(grid_padding) or grid_padding < 0.0:
            raise ValueError("grid_padding must be finite and non-negative")
        resolved_grid_padding = float(grid_padding)
    x_coordinates, y_coordinates = _requested_grid(
        positions,
        pixel_scale=float(pixel_scale),
        output_shape=output_shape,
        origin=origin,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
        padding=resolved_grid_padding,
    )
    ny, nx = y_coordinates.size, x_coordinates.size
    x_grid, y_grid = np.meshgrid(x_coordinates, y_coordinates)
    flux_sum = np.zeros((ny, nx), dtype=float)
    weight_sum = np.zeros((ny, nx), dtype=float)
    contribution_count = np.zeros((ny, nx), dtype=np.int32)
    variance_numerator = np.zeros((ny, nx), dtype=float)
    error_weight_sum = np.zeros((ny, nx), dtype=float)
    kernel_radius = max(
        1, int(np.ceil(max(4.0, float(support_sigma)) * sigma_pixels))
    )

    for index, ((x_position, y_position), value) in enumerate(zip(positions, values)):
        if not np.isfinite(value) or not np.isfinite(x_position) or not np.isfinite(y_position):
            continue
        x_center = int(np.floor((x_position - x_coordinates[0]) / pixel_scale))
        y_center = int(np.floor((y_position - y_coordinates[0]) / pixel_scale))
        x_low = max(0, x_center - kernel_radius)
        x_high = min(nx, x_center + kernel_radius + 1)
        y_low = max(0, y_center - kernel_radius)
        y_high = min(ny, y_center + kernel_radius + 1)
        if x_low >= x_high or y_low >= y_high:
            continue
        local_x = x_grid[y_low:y_high, x_low:x_high]
        local_y = y_grid[y_low:y_high, x_low:x_high]
        dx = local_x - x_position
        dy = local_y - y_position
        local_support = np.square(dx) + np.square(dy) <= support_radius**2
        local_weights = np.exp(
            -0.5 * (np.square(dx) + np.square(dy)) / sigma_coordinate**2
        )
        local_weights = np.where(local_support, local_weights, 0.0)
        flux_sum[y_low:y_high, x_low:x_high] += local_weights * value
        weight_sum[y_low:y_high, x_low:x_high] += local_weights
        contribution_count[y_low:y_high, x_low:x_high] += local_support.astype(np.int32)
        if error_values is not None and np.isfinite(error_values[index]) and error_values[index] >= 0.0:
            variance_numerator[y_low:y_high, x_low:x_high] += (
                np.square(local_weights) * np.square(error_values[index])
            )
            error_weight_sum[y_low:y_high, x_low:x_high] += local_weights

    support = weight_sum > 0.0
    image = np.full((ny, nx), fill_value, dtype=float)
    np.divide(flux_sum, weight_sum, out=image, where=support)
    weight = np.where(support, weight_sum, 0.0)
    variance: np.ndarray | None = None
    if error_values is not None:
        variance = np.full((ny, nx), np.nan, dtype=float)
        complete = support & np.isclose(
            error_weight_sum, weight_sum, rtol=1.0e-5, atol=1.0e-7
        )
        np.divide(
            variance_numerator,
            np.square(weight_sum),
            out=variance,
            where=complete,
        )
    return SpatialReconstructionResult(
        image=image,
        weight=weight,
        support=support,
        variance=variance,
        x_coordinates=x_coordinates,
        y_coordinates=y_coordinates,
        contribution_count=contribution_count,
    )


reconstruct_spatial_image = gaussian_splat

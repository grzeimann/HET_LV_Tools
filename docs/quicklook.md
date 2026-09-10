# Quick-look workflows

The numerical workflows operate on a detector image and an amplifier-level
fiber topology. A dense topology contains a detector trace for each detector
column and its physical IFU-plane position. The shared path uses a 5-pixel
fractional aperture, selects the central 200 detector columns, and reconstructs
the IFU image with a Gaussian splat.

```python
from hetquicklook.workflows import run_standard_star_quicklook

result = run_standard_star_quicklook(
    detector,
    topology,
    requested_position=(12.0, 9.0),
)
print(result.measured_centroid)
print(result.offset)
```

`run_ldls_flat_quicklook` extracts and collapses each trace, then places the
fiber values on an IFU-plane image. Use `detector_variance`, `trace_map`,
`extraction_width`, `collapse_columns`, `gaussian_fwhm`, and `pixel_scale` to
make the numerical inputs explicit. The current `mean` collapse statistic is
configurable and remains pending scientific confirmation.

`run_standard_star_quicklook` returns the same spatial product and retains the
existing positive-signal weighted centroid as a compatibility product. That
centroid is not the final scientific estimator; later pointing work can
consume `result.image`, `result.spatial_support`, or `result.fiber_values`.

The workflows do not read files or select observations. Those responsibilities
belong to discovery and higher-level operational code, keeping the numerical
contracts easy to test with arrays.

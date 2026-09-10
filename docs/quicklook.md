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
    instrument="lrs2",
)
print(result.measured_centroid)
print(result.offset)
print(result.intended_fiducial)
```

`run_ldls_flat_quicklook` extracts and collapses each trace, then places the
fiber values on an IFU-plane image. Use `detector_variance`, `trace_map`,
`extraction_width`, `collapse_columns`, `gaussian_fwhm`, and `pixel_scale` to
override the workflow inputs. The defaults are a 5-pixel fractional aperture,
a central 200-column finite-value median, and instrument-selected spatial
parameters:

| Instrument | Gaussian FWHM | Pixel scale |
| --- | ---: | ---: |
| VIRUS | 1.5 arcsec | 1.0 arcsec/pixel |
| LRS2 | 1.2 arcsec | 0.4 arcsec/pixel |

When bounds are omitted, the spatial algorithm derives them from the physical
fiber-coordinate extent and pads the extent by the configured Gaussian support
radius. The resulting `spatial_x_coordinates` and `spatial_y_coordinates`
keep the image grid tied to the physical IFU coordinates.

`run_standard_star_quicklook` records the intended `(0, 0)` position in the
selected VIRUS or LRS2 IFU coordinate system and returns the existing
positive-signal weighted centroid as a measured location. Both positions are
evidence for the observer. No automatic quality decision follows from their
separation. Later pointing work can consume `result.image`,
`result.spatial_support`, `result.spatial_weight`, `result.fiber_values`, and
the extraction evidence arrays. The result also records the selected
instrument, physical fiber positions, collapse settings, spatial grid, and
spatial settings used to make the image.

The workflows do not read files or select observations. Those responsibilities
belong to discovery and higher-level operational code, keeping the numerical
contracts easy to test with arrays.

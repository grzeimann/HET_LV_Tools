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

## Workflow composition and plotting

`build_amplifier_topology` is the explicit workflow boundary for one prepared
detector and one resolved physical identity. It loads the dated trace
reference, fits the shared dense trace map, attaches the authoritative fiber
positions, and returns trace and resource provenance. It does not discover
archives or load raw files.

`run_lrs2_channel_quicklooks` is the thin archive-facing composition helper
for an already discovered `Exposure`; it loads its raw members on demand. It
runs the same independent amplifier path
for the eight expected LRS2 amplifier tokens, retains each
`LRS2AmplifierQuicklookEvidence` record, and returns an `LRS2QuicklookSet`
containing both the amplifier products and the four complete channel
products. It requires all eight requested frame members and reports missing
members explicitly.

The reusable Matplotlib functions in `hetquicklook.visualization` include
`plot_spatial_image`, `plot_fiber_values`, and `plot_spatial_support`. They
accept either an amplifier `SpatialQuicklook` or an `LRS2ChannelQuicklook`.
The notebook keeps its four-panel layout as an exploratory presentation
choice while using these package helpers for the individual evidence views.

## LRS2 channel products

An LRS2 amplifier remains the detector-processing unit. The existing workflow
returns one `SpatialQuicklook` per amplifier, retaining its detector, trace,
extraction, variance, and collapse evidence. An LRS2 channel is the spatial
quick-look unit formed from two independently processed amplifier products:

| Channel | Amplifiers | Fibers |
| --- | --- | ---: |
| UV | `056LL` + `056LU` | 280 |
| Orange | `056RL` + `056RU` | 280 |
| Red | `066LL` + `066LU` | 280 |
| Far-Red | `066RL` + `066RU` | 280 |

`combine_lrs2_channel_products` performs this composition only after each
amplifier has reached its collapsed fiber values and authoritative physical
IFU positions. It concatenates those position/value arrays and calls the same
instrument-agnostic Gaussian-splat algorithm once for the complete channel.
The two amplifier products remain attached to the resulting
`LRS2ChannelQuicklook`, so detector-coordinate evidence is not lost or merged.

`combine_lrs2_channels` applies the authoritative pairs and returns the four
channel products for a complete eight-amplifier LRS2 exposure. Both helpers
require the expected data; missing amplifiers raise a clear error rather than
being represented as a complete 280-fiber channel. Standard-star channel
centroids are recomputed from all 280 physical fibers using the existing
weighted-centroid measurement and remain evidence without an automatic quality
decision.

The resulting hierarchy is:

```text
shared amplifier numerical pipeline
        -> SpatialQuicklook per amplifier
        -> LRS2 topology composition
        -> LRS2ChannelQuicklook per channel
        -> four-channel operator presentation
```

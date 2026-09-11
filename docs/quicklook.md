# Interactive quick start

The high-level API is the normal entry point for an interactive notebook. It
keeps archive discovery, exposure bookkeeping, detector preparation, trace
resolution, topology construction, and instrument-specific presentation
behind a small set of orchestration objects.

```python
from hetquicklook import QuicklookSite

ql = QuicklookSite(
    raw_roots={
        "lrs2": "~/data/LRS2",
        "virus": "~/data/VIRUS",
    },
)

LRS2Night = ql.night("20260512", instrument="lrs2")
LRS2Night

# Later in the night, rediscover new archives in place.
LRS2Night.update()

exposure = LRS2Night[25]  # choose another row from the displayed table as needed
product = exposure.quicklook()
product.plot()
```

Evaluating `LRS2Night` in Jupyter displays one compact row per encoded exposure,
including exposures from multiple observation archives. Rows use zero-based
Python indexing, so `LRS2Night[19]` selects the row labelled `19`. The selected
`QuicklookExposure` retains its `observation`, `raw_exposure`, `metadata`, and
`classification` attributes.

Automatic dispatch uses the existing deterministic classification. Recognized
LDLS or Qth flats use the flat workflow, catalog-matched standard stars use the
standard-star workflow, and all other frame types use the target workflow for
spatial inspection. An expert may use `exposure.quicklook(kind="flat")`,
`kind="standard"`, or `kind="target"` when an explicit override is appropriate.

VIRUS processing is sequential. For diagnostics, the quick-look call accepts
`timing=True` and `memory_check=True`. The resulting per-IFU diagnostics are
available at `product.ifus["074"].diagnostics`.

Archive-backed quick looks use the prepared detector's central 200-column
window for flat, standard-star, and target exposures. Flats are traced from
their own local window; standards and targets use the cached local trace from
a suitable classified flat. Extraction is collapsed immediately, so the
default product does not retain raw detector arrays or extracted spectra. Use
`exposure.quicklook(detailed_evidence=True)` when detailed per-amplifier
detector and extraction evidence is needed.

For LRS2, successful `product.channels` contains the complete UV, Orange, Red,
Far-Red channel set. Missing required amplifier files or processing failures
raise a contextual `QuicklookError`; no partial scientific product is returned.
The product retains `product.raw_result` and `product.amplifier_evidence`.
A channel is created only from its two required 140-fiber amplifier products.

For VIRUS, `product.ifus` groups amplifier evidence by physical IFU slot and
composes one approximately 50-arcsecond IFU-plane image from the available
fibers. A present amplifier that fails quick-look reduction is available in
`product.ifus["095"].unavailable_amplifiers` with its failure reason; missing
raw amplifier files remain hard errors. With no `ifu` argument, `product.plot()` renders the
available IFU images in the authoritative 10-by-10 focal-plane layout, with
each VIRUS slot labelled in its physical position. The positions occupied by
other instruments (`054`, `055`, `056`, `064`, `065`, and `066`) are left
blank. Pass `ifu="074"` to render one IFU directly. The individual amplifier
views remain available through
`product.ifus["074"].plot_amplifiers()` when the quick look was run with
`detailed_evidence=True`.

All `product.plot()` variants accept `cmap`, `vmin`, and `vmax` for display
control. Explicit `vmin` or `vmax` values override the corresponding percentile
limit; omitted limits continue to use the `percentiles` argument.

The high-level objects are orchestration and presentation wrappers. The
lower-level numerical workflows remain useful when an expert needs direct
control over a prepared detector array and `FiberTopology`; their richer
in-memory evidence options remain available for direct inspection.

`QuicklookSite` resolves the repository's dated `Fiber_Locations` tree from
the installed package location, so the notebook working directory does not
affect normal trace lookup. An explicit `trace_root` may still be supplied for
an alternate external trace deployment. Calling `LRS2Night.update()` rediscovers
the same date and instrument through that site, reloads the observation
metadata, and refreshes the table and exposure wrappers in place. Its return
value is the same night object, so evaluating `LRS2Night.update()` in a
notebook renders the refreshed table immediately.

# Low-level quick-look workflows

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
| VIRUS | 1.8 arcsec | 1.0 arcsec/pixel |
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
runs the same independent amplifier path for the eight expected LRS2 amplifier
tokens, retains each `LRS2AmplifierQuicklookEvidence` record, and returns an
`LRS2QuicklookSet` containing both the amplifier products and complete channel
products. Missing required members or processing failures raise a contextual
`QuicklookError` before a result is returned.

The reusable Matplotlib functions in `hetquicklook.visualization` include
`plot_spatial_image`, `plot_fiber_values`, `plot_spatial_support`,
`plot_lrs2_channels`, and the per-IFU VIRUS amplifier view. The high-level
product uses `plot_lrs2_channels` for the four-panel LRS2 layout and preserves
the lower-level helpers for individual evidence views.

## LRS2 channel products

An LRS2 amplifier remains the detector-processing unit. The existing workflow
returns one compact `SpatialQuicklook` per amplifier, retaining collapsed
fiber values, physical positions, and local trace provenance. An LRS2 channel
is the spatial quick-look unit formed from two independently processed
amplifier products:

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

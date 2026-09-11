# HET_LV_Tools: Quick-look tools for HET VIRUS and LRS2

`HET_LV_Tools` provides lightweight quick-look tools for operational assessment
of the Hobby–Eberly Telescope (HET) VIRUS and LRS2 instruments. The Python
package is named `hetquicklook`.

The project is aimed at daily RA OPS checks using LDLS and Qth flats,
standard-star observations, and other science or calibration frames. It turns
the detector and fiber information already present in HET data into inspectable
spatial and pointing evidence without reproducing the full VIRUS or LRS2
reduction pipelines.

- Getting started: [Installation](docs/installation.md) · [Quick-look workflows](docs/quicklook.md)
- Data and instrument information: [Data layout](docs/data_layout.md) · [VIRUS](docs/virus.md) · [LRS2](docs/lrs2.md)
- Interactive exploration: [Observer quickstart notebook](notebooks/hetquicklook_presentation_playground.ipynb)

## Install (conda or pip)

```bash
# Conda environment with runtime and development dependencies
conda env create -f environment.yml
conda activate hetquicklook

# Or use a virtual environment
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Quickstart (CLI)

The CLI currently provides archive discovery and member inventory. It accepts
direct date/instrument archives and the nested VIRUS Corral date-tar layout.

```bash
# Show help and verify installation
hetquicklook -h

# List VIRUS observations for one night
hetquicklook discover /path/to/data --instrument virus --date 20260910

# Inspect literal archive members and parsed frame identities as JSON
hetquicklook discover /path/to/data --instrument virus --date 20260910 \
  --json --inventory
```

The same discovery layer is available from Python:

```python
from hetquicklook import QuicklookConfig, discover_observations, load_observation

observations = discover_observations(
    QuicklookConfig("/data/het"), instrument="virus", date="20260910"
)
observation = load_observation(observations[0])
print(observation.exposure_ids)
```

## Interactive quickstart

For routine notebook use, configure the raw roots once, then choose a night,
select an exposure from its compact HTML inventory, and display the instrument
quick look. In a source or editable installation, dated trace resources are
resolved from this repository by default; pass `trace_root=...` to
`QuicklookSite` when using an external trace deployment:

```python
from hetquicklook import QuicklookSite

ql = QuicklookSite(
    raw_roots={"lrs2": "~/data/LRS2", "virus": "~/data/VIRUS"},
)
LRS2Night = ql.night("20260512", instrument="lrs2")
LRS2Night
LRS2_product = LRS2Night[25].quicklook()  # choose a row from the displayed table
LRS2_figure = LRS2_product.plot(cmap="coolwarm")

VIRUSNight = ql.night("20260609", instrument="virus")
VIRUSNight
VIRUS_product = VIRUSNight[11].quicklook()  # choose a row from the displayed table
VIRUS_figure = VIRUS_product.plot(cmap="coolwarm")

# Run this again later to refresh the same table and night object.
LRS2Night.update()
```

The selected exposure retains its observation, classification, and
archive-member provenance. The resulting product retains collapsed fiber and
spatial results, amplifier-level evidence, and complete LRS2 channel products.
Use `exposure.quicklook(detailed_evidence=True)` when detector arrays and
extracted spectra are needed for detailed inspection. See the
[interactive quick-start guide](docs/quicklook.md) for evidence retention,
instrument-specific products, and advanced inspection.

For the detector-to-fiber workflow and result objects, see the [quick-look
workflow guide](docs/quicklook.md) and the notebook.

## Data and configuration

Discovery expects a configurable root with date and instrument directories:

```text
ROOT/
  20260910/
    virus/
      observation.tar
    lrs2/
      observation.tar
```

VIRUS Corral date archives are also supported:

```text
ROOT/
  20260910.tar
    virus/
      virus0000001.tar
        <FITS members>
```

The package inventories what is present, including malformed members, partial
amplifier sets, nested archive provenance, and archives containing multiple
exposure IDs. It does not infer completeness during discovery. The discovery
layer recognizes `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, and `.tar.xz` archives.

Static VIRUS and LRS2 fiber-position resources are packaged with
`hetquicklook`. Dated detector traces are supplied through a trace root with
the following structure:

```text
TRACE_ROOT/
  Fiber_Locations/
    YYYYMMDD/
      fiber_loc_<SPECID>_<IFUSLOT>_<IFUID>_<AMP>.txt
```

See [installation](docs/installation.md) for the topology loader and
[data-layout documentation](docs/data_layout.md) for the supported archive
forms.

## Current scientific capabilities

The current package and notebook provide:

* exposure-level FITS metadata, explicit header disagreements, filename
  identities, and deterministic classification of LDLS or Qth flats and
  catalog-matched standard stars, with all other frame types dispatched to the
  general target spatial quick-look path;
* explicit VIRUS and LRS2 instrument topology, packaged IFU-plane positions,
  and dated fiber-trace resolution from `Fiber_Locations`;
* detector preparation with overscan subtraction, trimming, amplifier
  orientation, gain handling, and detector variance/read-noise evidence;
* continuum-flat trace fitting, fractional-aperture fiber extraction, and
  central-column collapse into one value per fiber;
* wavelength-independent Gaussian-splat reconstruction of IFU spatial images,
  with physical coordinates, weights, support masks, propagated errors, and
  contribution counts retained for review;
* standard-star weighted centroids and measured-minus-intended offsets using
  the `(0, 0)` IFU fiducial; and
* complete LRS2 quick looks that process eight amplifier products independently
  and compose them into the four UV, Orange, Red, and Far-Red channel products
  while retaining amplifier-level evidence; and
* high-level site, night, exposure, and product wrappers with compact and
  refreshable exposure inventories, automatic flat/standard/target dispatch,
  per-IFU and full VIRUS focal-plane plots, complete LRS2 channel plots, and
  display controls such as `cmap`, `vmin`, and `vmax`.

The default quick-look settings are:

| Setting | VIRUS | LRS2 |
| --- | ---: | ---: |
| Fractional extraction width | 5 detector pixels | 5 detector pixels |
| Collapse window/statistic | central 200 columns, median | central 200 columns, median |
| Gaussian-splat FWHM | 1.8 arcsec | 1.2 arcsec |
| Spatial-grid padding | 1.5 arcsec | 0.3 arcsec |
| Output pixel scale | 1.0 arcsec/pixel | 0.4 arcsec/pixel |
| Standard-star fiducial | `(0, 0)` in selected IFU | `(0, 0)` in LRS2 IFU |

`grid_padding_arcsec` pads the inferred image bounds around the physical fiber
coordinates. It changes the output extent, not the fiber positions or the
Gaussian reconstruction kernel. Override it per exposure, for example with
`exposure.quicklook(grid_padding_arcsec=2.0)`.

These products are intended to help an observer inspect obscuration,
contamination, illumination structure, blocked or dim fibers, trace behavior,
spatial support, and standard-star pointing. The results supply evidence for
review; they do not issue automatic warnings, rejection decisions, or quality
grades.

## Notebook and presentation work

The [observer quickstart notebook](notebooks/hetquicklook_presentation_playground.ipynb)
is written for an observer who wants a small number of repeatable notebook
steps. It opens example `VIRUSNight` and `LRS2Night` inventories, shows example
standard-star, flat, and long-science selections, uses the `coolwarm` colormap,
and demonstrates refreshing a night with `.update()`. The package's lower-level
workflow and detailed evidence options are documented in
[quicklook.md](docs/quicklook.md).

The notebook is deliberately exploratory. Display stretches and figure layout
can change without changing the stored scientific results.

## Current limitations

The current scope stops at wavelength-independent quick-look evidence. It does
not yet provide wavelength calibration, sky modeling, differential atmospheric
refraction correction, production spectrophotometric calibration, seeing
estimation, or automated quality policy. Operational thresholds and further
validation on representative observations remain future work.

## Testing and development

Run the test suite with:

```bash
python -m pytest
```

The [implementation workspace](docs/implementation_workspace.md) records the
current implementation boundary, scientific contracts, and remaining
integration questions. [Architectural_Design.md](Architectural_Design.md)
describes the separation between discovery, metadata, topology, algorithms,
workflows, and presentation.

## License

HET_LV_Tools is released under the MIT License. See [LICENSE](LICENSE).

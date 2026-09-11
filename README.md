# HET_LV_Tools: Quick-look tools for HET VIRUS and LRS2

`HET_LV_Tools` provides lightweight quick-look tools for operational assessment
of the Hobby–Eberly Telescope (HET) VIRUS and LRS2 instruments. The Python
package is named `hetquicklook`.

The project is aimed at daily RA OPS checks using LDLS flats and standard-star
observations. It turns the detector and fiber information already present in
HET data into inspectable spatial and pointing evidence without reproducing the
full VIRUS or LRS2 reduction pipelines.

- Getting started: [Installation](docs/installation.md) · [Quick-look workflows](docs/quicklook.md)
- Data and instrument information: [Data layout](docs/data_layout.md) · [VIRUS](docs/virus.md) · [LRS2](docs/lrs2.md)
- Interactive exploration: [Presentation playground](notebooks/hetquicklook_presentation_playground.ipynb)

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
      observation.tar.gz
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
exposure IDs. It does not infer completeness during discovery.

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
  identities, and deterministic classification of LDLS flats and standard
  stars;
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
  while retaining amplifier-level evidence.

The default quick-look settings are:

| Setting | VIRUS | LRS2 |
| --- | ---: | ---: |
| Fractional extraction width | 5 detector pixels | 5 detector pixels |
| Collapse window/statistic | central 200 columns, median | central 200 columns, median |
| Gaussian-splat FWHM | 1.5 arcsec | 1.2 arcsec |
| Output pixel scale | 1.0 arcsec/pixel | 0.4 arcsec/pixel |
| Standard-star fiducial | `(0, 0)` in selected IFU | `(0, 0)` in LRS2 IFU |

These products are intended to help an observer inspect obscuration,
contamination, illumination structure, blocked or dim fibers, trace behavior,
spatial support, and standard-star pointing. The results supply evidence for
review; they do not issue automatic warnings, rejection decisions, or quality
grades.

## Notebook and presentation work

The [presentation playground](notebooks/hetquicklook_presentation_playground.ipynb)
walks through the current archive-to-result path on real VIRUS or LRS2 data:
discovery, exposure metadata, one-amplifier inspection, topology and trace
provenance, quick-look results, and LRS2 channel composition. Its presentation
views include reconstructed spatial images, fiber-level values, standard-star
fiducials and centroids, spatial support, and lower-level trace or extraction
evidence when a result needs investigation.

The notebook is deliberately exploratory. Display stretches and figure layout
can change without changing the stored scientific results.

## Near-term goals

The next development work is to:

* connect archive selection, metadata, topology, quick-look execution, plotting,
  and output into a concise end-to-end operational workflow;
* use representative LRS2 flat and standard-star observations, followed by a
  VIRUS IFU, to settle the operator-facing presentation;
* promote useful notebook views into reusable package visualizations, including
  a clear four-channel LRS2 view and a VIRUS IFU or whole-focal-plane overview;
* evaluate centroid and display choices on real observations before defining
  evidence-based operational thresholds.

The current scope stops at wavelength-independent quick-look evidence. It does
not yet provide wavelength calibration, sky modeling, differential atmospheric
refraction correction, production spectrophotometric calibration, seeing
estimation, or automated quality policy.

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

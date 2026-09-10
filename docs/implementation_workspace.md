# Implementation workspace

This document records the implementation boundary established after comparing
the package skeleton with the supplied VIRUSFlow raw-I/O references and
representative local archives.

## Current state

The repository now has a complete small data-access and representation layer:

| Area | Current implementation | Boundary |
| --- | --- | --- |
| Configuration and discovery | [`config.py`](../src/hetquicklook/config.py) and [`discovery.py`](../src/hetquicklook/discovery.py) discover direct date/instrument archives and nested Corral date-tar VIRUS observations. | Discovery reports archive evidence; it does not decide scientific suitability or completeness. |
| Member inventory | `inventory_members()` records every regular tar member, literal member path, basename, size, nested archive member, parsed identity, and parse error. | No detector array is loaded during inventory. |
| Filename identity | `parse_member_identity()` follows the supplied VIRUSFlow convention: basename fields 0, 1, and 2 are exposure ID, amplifier token, and frame type; the amplifier token is split into a three-character IFU slot and suffix. | Unrecognized names remain in the inventory with `identity=None`. |
| Observation representation | [`observation.py`](../src/hetquicklook/observation.py) provides `Observation` plus one `Exposure` for each encoded exposure ID. Each exposure contains only its actual frames, normalized metadata, classification, header errors, and physical amplifier identities. | Archive-level metadata does not borrow the first header when multiple exposure IDs are present. No expected amplifier set is required. |
| FITS metadata | [`metadata.py`](../src/hetquicklook/metadata.py) normalizes demonstrated VIRUS exposure and scientific-header fields from all readable headers belonging to one exposure. It retains per-field disagreements and member provenance. | Missing values remain `None`; disagreements are observable and the package does not choose a scientific winner. |
| Raw loading | [`raw.py`](../src/hetquicklook/raw.py) loads direct or nested FITS members through the primary HDU and returns detector data, header, identity, and archive/member provenance. | Loading is on demand and has no VIRUSFlow cache, timing, index, or database registry. |
| Instrument interpretation | [`instrument.py`](../src/hetquicklook/instrument.py) retains the high-level hierarchies, encodes the fixed LRS2 component identities, and combines raw tokens with supported VIRUS/LRS2 header identity fields. | Generic raw parsing remains independent of LRS2 interpretation; incomplete VIRUS addresses remain incomplete rather than receiving defaults. |
| Classification | [`classification.py`](../src/hetquicklook/classification.py) provides right-parsed `OBJECT` intent, deterministic VIRUS/LRS2 flat classification, and the static HET/Hydra standard-star catalog with explicit historical aliases. | Classification is exact after parsing and alias normalization; it does not use broad historical substring matching. |
| Topology resources | [`topology.py`](../src/hetquicklook/topology.py) loads packaged VIRUS/LRS2 static resources and resolves dated VIRUS/LRS2 traces from an external `trace_root`. | Static resources are package data; dated `Fiber_Locations/` remains independently configured calibration state. |
| CLI | [`cli.py`](../src/hetquicklook/cli.py) supports `discover --json --inventory` for literal member and identity inspection. | Scientific selection and quick-look execution remain future commands. |
| Numerical algorithms and workflows | Shared detector preparation, continuum-flat trace fitting, fractional extraction, central-column collapse, and Gaussian-splat spatial reconstruction now operate on loaded arrays and authoritative fiber topology. | They remain in-memory array operations; archive discovery and scientific selection stay outside the algorithms. |

## Established raw-data contract

The supplied VIRUSFlow code and the explicit project information support these
behaviors:

* Direct archives occur below `ROOT/YYYYMMDD/INSTRUMENT/`.
* A Corral date archive can contain `virus/virusNNNNNNN.tar`; the physical
  provenance is the date tar plus the inner observation-tar member plus the
  FITS member.
* Member identity is parsed from the basename, independent of an internal
  directory prefix.
* The minimum recognized basename is equivalent to
  `<exposure_id>_<amp_token>_<frame_type>.fits` with an amplifier token of at
  least five characters. The implementation preserves the reference parser's
  field behavior rather than validating unprovided naming rules.
* The primary HDU supplies the detector array and relevant raw header. The
  loader copies the array before closing the FITS stream.
* An archive can contain multiple exposure IDs. The observation object exposes
  those groups without treating one archive as one exposure. Header metadata
  and classifications are attached to the exposure containing the header.
* The filesystem is evidence: partial amplifier sets, malformed members, and
  missing optional header values remain representable.

Representative local inspection provided the following checks:

* `~/data/LRS2/20260512/lrs2/lrs20000001.tar` contains 8 `twi` FITS members,
  all for exposure `20260512T015436.4`; the primary array is `(1032, 2128)`.
* `~/data/VIRUS/20260609/virus/virus0000001.tar` contains 300 `twi` FITS
  members, all for exposure `20260609T022141.1`; the primary array is
  `(1032, 1064)`.
* LRS2 archive `lrs20000032.tar` contains 24 `drk` members across 3 exposure
  IDs, and VIRUS archive `virus0000025.tar` contains 900 `drk` members across
  3 exposure IDs. This confirms that grouping is required at the member level.
* The representative LRS2 tokens include `056LL`, `056LU`, `056RL`, `056RU`,
  `066LL`, `066LU`, `066RL`, and `066RU`. The supplied map interprets these as
  UV/Orange for slot 056 and Red/Far-Red for slot 066.
* Real LRS2 flat exposures include `OBJECT=ldls_long_B` and `OBJECT=Qth_R`;
  the former applies to slot 056 and the latter to slot 066. Real VIRUS flat
  groups use `OBJECT=ldls_long` and `frame_type=flt`.

The representative commands can be run from a source checkout with
`PYTHONPATH=src` when the package has not been installed:

```bash
PYTHONPATH=src python -m hetquicklook.cli discover ~/data/LRS2 \
  --instrument lrs2 --date 20260512 --json --inventory
PYTHONPATH=src python -m hetquicklook.cli discover ~/data/VIRUS \
  --instrument virus --date 20260609 --json --inventory
```

## Public data-access boundary

The intended sequence is:

```python
from hetquicklook import QuicklookConfig, discover_observations, load_observation

discovered = discover_observations(
    QuicklookConfig("/data/het"), instrument="virus", date="20260609"
)[0]
observation = load_observation(discovered)

observation.frames                  # literal FITS evidence
observation.exposures               # exposure-level metadata and classification
observation.group_by_exposure()     # only exposure IDs present in names
frame = observation.frames_for(frame_type="twi")[0]
loaded = observation.load_frame(frame)
loaded.data                          # primary-HDU detector array
loaded.provenance                    # archive, nested member, FITS member
observation.exposure_for(exposure_id).metadata
observation.exposure_for(exposure_id).classification
```

`ObservationMetadata` remains an archive summary and records archive/member
identity. `ExposureMetadata` records header-derived values at the exposure
level, while `ExposureClassification` records the deterministic quick-look
interpretation separately. `PhysicalAmplifierIdentity` keeps raw filename
identity, fixed LRS2 identity, and header-derived VIRUS fields distinct.

The topology boundary is explicit:

* Packaged VIRUS geometry uses `resources/virus/IFUcen_HETDEX.txt`, with the
  configured `IFUcen_HETDEX_reverse_R.txt` alternate for IFUID `004`, and the
  supplied amplifier slices/corrections/order.
* VIRUS traces use dated `Fiber_Locations/<date>/fiber_loc_<SPECID>_<IFUSLOT>_<IFUID>_<AMP>.txt` resources selected by nearest date.
* Packaged VIRUS focal-plane coordinates use `resources/virus/fplaneall.txt`
  and expose IFUSLOT to `(x, y)` values.
* Packaged LRS2 geometry uses the channel-specific `resources/lrs2/` files.
  Their five-line headers, six-column rows, coordinate columns 1 and 2,
  280-fiber shape, amplifier slices, and reversal are represented by the
  loader.
* The same dated trace resolver handles VIRUS and LRS2 identities. The
  external root is the directory containing `Fiber_Locations/`.

## Established quick-look numerical path

The next layer now follows the supplied VIRUSFlow/Panacea numerical contracts:

```text
raw primary-HDU amplifier image
    -> robust row-wise overscan subtraction and trim
    -> established amplifier orientation and gain
    -> detector error/variance
    -> continuum-flat trace fit from the dated fiber_loc reference
    -> dense per-fiber detector trace map
    -> 5-pixel fractional top-hat extraction
    -> fiber x detector-column spectra
    -> central 200 detector columns
    -> one scalar per fiber
    -> Gaussian-splat IFU-plane image
```

The detector, trace, and extraction implementations are shared by VIRUS and
LRS2. Their fiber counts come from the input arrays: VIRUS commonly has 112
traces per amplifier and LRS2 commonly has 140. Instrument names do not select
different numerical routines. The documented `504/018/RU` trace exception is
retained.

Detector preparation uses the established `32 * nx / 1064` overscan scaling,
row-wise robust subtraction, overscan trimming, the `LU`/`RL` both-axis flip,
the `AMPNAME=LR` or `UL` column flip, gain multiplication, and the fallback
values `GAIN=0.85` and `RDNOISE=3.0`. It does not add bias, dark, flat,
cosmic-ray, scattered-light, or bad-column corrections.

Trace fitting uses 40 detector-column chunks, median profiles, broad
cross-dispersion background removal, Gaussian smoothing, local maxima,
three-point parabolic subpixel localization, reference offsets for configured
dead fibers, and an independent robust polynomial of degree at most 4 per
fiber. Trace sample positions, sample columns, residuals, residual RMS values,
valid sample counts, and reference-interpolated state remain available as QA
arrays.

Extraction uses an exact continuous top-hat aperture with default width 5.0
detector pixels. Flux uses the fractional overlap weights and diagonal
variance uses their squares. The initial quick-look workflow uses
`pixel_mask=None` unless a caller supplies one.

Collapse selects the central 200 detector columns by default, with the width
configurable and centered on the actual extracted-spectrum width. The current
`mean` statistic is retained as the explicit default from the existing
quick-look collapse routine. Confirmation of the final statistic remains open.

Spatial reconstruction uses a local Gaussian splat over authoritative physical
fiber `(x, y)` positions. The default FWHM is 1.8 arcsec, inherited from
`make_mosaic_cube_from_fit.py`, and output pixel scale and bounds are explicit
parameters. Pixels without Gaussian support remain `NaN` with a false support
mask and zero accumulated weight. The reconstruction has no wavelength,
multiple-shot, sky, ADR, WCS, or M101-specific quality requirements.

The provisional LRS2 standard-star intended fiducial is `(0, 0)` in the LRS2
IFU/focal-plane coordinate system. It is retained as documented configuration
knowledge and awaits explicit verification; it is not silently applied to
VIRUS. The spatial image and fiber-level values are the product boundary for a
future measured-centroid estimator.

## Standard-star catalog

The static default catalog is the 45-name HET/Hydra set in
`hetquicklook.classification.STANDARD_STAR_NAMES`. Classification parses
`TARGET_IFUSLOT_TRACK` from the right, extracts the target, normalizes only
explicit historical aliases, and performs exact canonical membership. The
historical Panacea spellings retained as aliases are `HZ_44`, `HZ_21`, `HZ_4`,
`FEIGE_34`, `FEIGE_110`, `GRW+70_5824`, `BD+26+2606`, `BD_+17_4708`, and
`BD_+26_2606`. The underscored spellings are not members of the canonical
catalog.

## Questions that remain

The following questions remain relevant to the next scientific layer:

| Topic | Question | Why it matters |
| --- | --- | --- |
| Collapse statistic | Which explicit statistic should reduce the selected central 200 detector columns to one scalar? | The current `mean` behavior is retained as the default pending scientific confirmation. |
| Pointing estimator | Which measured-centroid estimator should consume the spatial image or fiber values? | The existing weighted centroid remains a compatibility helper and has no final scientific authority. |
| Pointing fiducials | Is the LRS2 intended fiducial `(0, 0)` correct, and what is the corresponding VIRUS fiducial? | The LRS2 value is provisional and VIRUS is unresolved. |
| Spatial defaults | What final pixel scale and image bounds should be operational defaults? | The algorithm accepts explicit values and currently follows the 1 arcsec/pixel script default when none is supplied. |
| Quality and operations | Which bad-pixel, saturation, cosmic-ray, and partial-component conditions should be reported or reject a scientific result? | Detector, trace, extraction, and spatial support evidence is preserved; thresholds remain deferred. |

## Tests and completion criteria for this pass

The synthetic suite covers direct and nested archives, deterministic inventory,
basename parsing, malformed names, primary-header metadata, detector loading,
provenance, partial observations, multi-exposure metadata, disagreement
visibility, flat and exact standard boundaries, fixed LRS2 identities, VIRUS
identity construction, packaged static topology resources, real dated
VIRUS/LRS2 trace resolution, topology loader row/order contracts, detector
preparation, shared 112/140-fiber trace fitting and extraction, central-column
collapse, and single-shot Gaussian reconstruction. The normal suite does not
depend on `~/data`.

This layer is complete when a loaded VIRUS or LRS2 amplifier can follow the
shared detector-to-fiber quick-look path through a spatial image while keeping
instrument identity, dated trace calibration, and authoritative IFU positions
separate. Full archive selection, wavelength calibration, sky modeling, DAR,
production spectrophotometric calibration, and final quality/rejection policy
remain outside this boundary.

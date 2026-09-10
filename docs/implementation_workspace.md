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
| Classification | [`classification.py`](../src/hetquicklook/classification.py) provides right-parsed `OBJECT` intent, deterministic VIRUS/LRS2 flat classification, and an explicit unavailable/loaded standard-star catalog boundary. | No runtime catalog lookup and no historical Panacea standard list are used. |
| Topology resources | [`topology.py`](../src/hetquicklook/topology.py) loads packaged VIRUS/LRS2 static resources and resolves dated VIRUS/LRS2 traces from an external `trace_root`. | Static resources are package data; dated `Fiber_Locations/` remains independently configured calibration state. |
| CLI | [`cli.py`](../src/hetquicklook/cli.py) supports `discover --json --inventory` for literal member and identity inspection. | Scientific selection and quick-look execution remain future commands. |
| Numerical algorithms and workflows | The existing array-based algorithms and in-memory workflows remain unchanged. | They still require detector arrays and authoritative fiber topology as inputs. |

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

## Questions that remain for the next layer

The following questions remain relevant to the next scientific layer:

| Topic | Question | Why it matters |
| --- | --- | --- |
| Standards | What canonical packaged standard-star catalog should be used? | `OBJECT` parsing is implemented, but classification remains explicitly unknown without this catalog. |
| Detector reduction | What minimal overscan/trim/gain/noise treatment is required before trace sampling? | Raw primary-HDU arrays and supported orientation facts are available; scientific detector preparation is not yet selected. |
| Extraction | What aperture, trace sampling, weighting, background, and quality rules reproduce the operational signal? | The current extraction code remains a synthetic placeholder. |
| Collapse | Which detector samples and normalization define an LDLS or standard-star collapsed value? | This determines the diagnostic quantity and units. |
| Spatial products | How should physical IFU coordinates, gaps, overlaps, missing fibers, and orientation appear in an image? | The existing spatial array operation is not yet a pipeline-backed product. |
| Pointing | What requested-coordinate convention, centroid estimator, and fiducial rules define the operational offset? | A measured centroid is not meaningful until it shares the requested coordinate system. |
| Quality and operations | Which bad-pixel, saturation, cosmic-ray, and partial-component conditions should be reported or reject a scientific result? | Raw loading deliberately preserves evidence without making scientific validity decisions. |

## Tests and completion criteria for this pass

The synthetic suite covers direct and nested archives, deterministic inventory,
basename parsing, malformed names, primary-header metadata, detector loading,
provenance, partial observations, multi-exposure metadata, disagreement
visibility, flat and standard boundaries, fixed LRS2 identities, VIRUS identity
construction, packaged static topology resources, real dated VIRUS/LRS2 trace
resolution, and topology loader row/order contracts. The normal suite does not
depend on `~/data`.

This pass is complete when a real VIRUS or LRS2 archive can be discovered,
inventoried, inspected by encoded identities, represented as an observation,
and loaded on demand with exposure metadata, deterministic classification,
instrument identity, and archive/member provenance. Fiber extraction, collapse,
spatial reconstruction, centroiding, pointing offsets, and quick-look plots
remain outside this boundary.

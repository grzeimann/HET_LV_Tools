# HET_LV_Tools

## Background

`HET_LV_Tools` is intended to provide lightweight quick-look tools for operational assessment of HET instrumentation, initially focused on VIRUS and LRS2.

The immediate motivation is to support daily RA OPS checks using existing HET data products and existing knowledge from the mature VIRUS and LRS2 reduction pipelines.

The first requested applications are:

1. Generate collapsed spatial images from recent VIRUS and LRS2 LDLS-flat observations to look for obscuration, contamination, or other spatial throughput anomalies.
2. Generate collapsed spatial images from recent VIRUS and LRS2 standard-star observations to assess pointing, including fiducials for intended and measured source locations.

The project is not intended to reproduce the full reduction pipelines. The required instrument knowledge already exists. The goal is to expose the limited subset of that knowledge needed for fast operational diagnostics.

---

# Design Goals

The initial system should be:

* small;
* easy to understand;
* easy to test;
* usable both at TACC and locally at HET;
* based on the data that are actually present;
* explicit about instrument and fiber topology;
* composed primarily of small algorithms with documented input/output contracts;
* and capable of supporting simple higher-level operational workflows.

The initial implementation should avoid unnecessary abstraction and should not attempt to solve problems that are not yet required by the operational use cases.

---

# Data Philosophy

The filesystem is treated as evidence of what data exist.

For VIRUS and LRS2, the existing Remedy workflow discovers observation tar files beneath a structure of the form

```text
ROOT/
    DATE/
        INSTRUMENT/
            <observation tar files>
```

The current TACC implementation uses a configurable root followed by date and instrument, with VIRUS and LRS2 observations discovered from instrument-specific tar filenames.

The same software should be usable at TACC and on the mountain by changing the root directory rather than changing the underlying discovery logic.

The discovery system should report what is present.

It should not require that an expected set of files be complete before representing or processing an observation.

A missing file therefore represents absence of data, rather than automatically representing an invalid observation.

---

# Metadata

The system should maintain structured metadata describing the observations and files that were actually discovered.

Existing operational code already extracts useful observation-level metadata directly from FITS headers, including quantities such as:

* object name;
* observing program;
* exposure times;
* requested coordinates;
* and observation date/time.

The exact metadata schema for `HET_LV_Tools` remains to be defined.

The metadata layer should describe the available data and provide enough information for higher-level code to select appropriate observations and operations.

---

# Instrument Topology

The system must encode the known physical organization of VIRUS and LRS2.

This information is already known through the existing instrument reduction pipelines.

At a high level, instrument topology describes how detector-level components combine into larger physical structures.

For example:

```text
VIRUS
amplifier -> IFU -> instrument

LRS2
amplifier -> channel -> spectrograph -> instrument
```

The detailed mappings should be derived from existing pipeline knowledge rather than independently reconstructed.

---

# Fiber Topology

Fiber topology is distinct from instrument topology.

At the amplifier level, fiber topology describes:

1. the detector trace of each fiber; and
2. the mapping between that fiber and its physical location in the IFU plane.

For the initial quick-look applications, wavelength information is not required.

The required relationship is therefore approximately:

```text
detector signal
      |
      v
fiber trace
      |
      v
fiber
      |
      v
physical IFU-plane position
```

Astrometric information may later extend this physical IFU-plane mapping into sky coordinates where required.

The detailed representation of fiber topology remains to be defined.

---

# Architectural Principles

## Separation of Responsibilities

The package should keep several responsibilities separate:

```text
filesystem discovery
        |
        v
metadata representation

instrument topology
        |
        +----------------+
                         |
fiber topology ----------+
                         |
                         v
                    algorithms
                         |
                         v
                     workflows
                         |
                         v
              visualization / CLI
```

Each layer should remain relatively independent.

### Discovery

Discovery determines what data are present.

It should know about:

* root paths;
* observing dates;
* instruments;
* filesystem organization;
* archive or tar-file organization.

It should not contain scientific analysis logic.

### Metadata

Metadata represents what was discovered.

It should provide structured information that can be queried by higher-level code.

### Instrument topology

Instrument topology describes how detector components relate to physical instrument components.

### Fiber topology

Fiber topology describes the detector traces and physical IFU-plane positions of fibers.

### Algorithms

Algorithms should perform relatively small, well-defined numerical operations.

They should preferably operate on supplied arrays and mappings rather than performing hidden filesystem discovery or observation selection.

### Workflows

Workflows provide the necessary cohesion between otherwise independent algorithms.

A workflow may know:

* what type of observation is being processed;
* which data are available;
* which topology information is needed;
* which sequence of algorithms should be run;
* and what quick-look product should be returned.

A useful distinction is:

> Algorithms know how to perform operations. Workflows know why and when those operations are combined.

### Visualization and CLI

The visualization and command-line layers should remain thin interfaces over the underlying package functionality.

They should not independently implement scientific algorithms.

---

# Coding Conventions

The project should use:

* PEP 8 compliant Python;
* Google-style docstrings;
* type annotations for public interfaces;
* clearly documented function input/output contracts;
* explicit array shapes where important;
* explicit units where relevant;
* explicit coordinate conventions where relevant;
* and predictable handling of invalid or missing numerical values.

Low-level numerical functions should generally return results rather than relying on hidden side effects.

Classes should be introduced where they represent meaningful entities or persistent structured state. Simple numerical operations should generally remain functions.

---

# Package Structure

The Git repository is:

```text
HET_LV_Tools
```

The Python package will use:

```text
hetquicklook
```

The initial repository structure is:

```text
HET_LV_Tools/
├── pyproject.toml
├── environment.yml
├── README.md
├── LICENSE
│
├── docs/
│   ├── installation.md
│   ├── data_layout.md
│   ├── virus.md
│   ├── lrs2.md
│   └── quicklook.md
│
├── src/
│   └── hetquicklook/
│       ├── __init__.py
│       ├── config.py
│       ├── discovery.py
│       ├── metadata.py
│       ├── instrument.py
│       ├── fibers.py
│       │
│       ├── algorithms/
│       │   ├── __init__.py
│       │   ├── extraction.py
│       │   ├── collapse.py
│       │   └── centroid.py
│       │
│       ├── visualization.py
│       ├── workflows.py
│       └── cli.py
│
└── tests/
    ├── test_discovery.py
    ├── test_metadata.py
    ├── test_instrument.py
    ├── test_fibers.py
    ├── test_collapse.py
    └── test_workflows.py
```

This structure should be considered an initial skeleton rather than a fixed final organization.

---

# Module Responsibilities

## `config.py`

Contains configuration needed by multiple parts of the package.

One known requirement is configuration of the filesystem root so that the same code can operate at TACC and on the mountain.

Site-specific paths should not be embedded throughout the package.

---

## `discovery.py`

Responsible for identifying observations and files present in the filesystem.

This module should reflect the actual data found rather than determining whether an expected dataset is complete.

---

## `metadata.py`

Provides structured representations of discovered observation and file metadata.

The exact data model remains to be defined.

---

## `instrument.py`

Contains the known physical topology of VIRUS and LRS2.

The detailed implementation and source of these mappings remain to be defined from the existing pipeline code.

---

## `fibers.py`

Contains the amplifier-level fiber topology required by the quick-look algorithms.

This includes:

* fiber traces on the detector;
* association of detector traces with fibers;
* physical IFU-plane positions of those fibers.

The detailed representation remains to be defined from existing pipeline knowledge.

---

## `algorithms/`

Contains small numerical algorithms with explicit input/output contracts.

The exact set of algorithms should grow from concrete requirements rather than being designed comprehensively in advance.

The initial anticipated categories are:

### `extraction.py`

Operations required to obtain fiber-level signal from detector data.

Exact algorithms remain to be defined.

### `collapse.py`

Operations associated with collapsing the appropriate signal into fiber-level or spatial measurements.

Exact algorithms remain to be defined.

### `centroid.py`

Operations associated with measuring the spatial location of a source.

Exact algorithms remain to be defined.

---

## `workflows.py`

Combines discovery, metadata, topology, and algorithms into operational quick-look procedures.

The first workflows are expected to correspond to:

* LDLS-flat spatial quick looks;
* standard-star pointing quick looks.

The detailed sequences should be defined only after the underlying data requirements and algorithms are specified.

---

## `visualization.py`

Contains visualization of quick-look results.

The detailed visual products remain to be defined.

---

## `cli.py`

Provides a simple operational interface to the package.

The CLI should call package functionality rather than contain independent scientific or filesystem logic.

The desired operational workflow should eventually consist of a relatively small number of clear calls, conceptually following:

```text
configure
    |
discover observations
    |
construct metadata
    |
select data
    |
run quick-look workflow
    |
visualize result
```

The exact public API should be established as the underlying modules are implemented.

---

# Installation

The project should support both pip-based and Conda-based installation.

`pyproject.toml` should provide the authoritative Python package definition.

For example:

```bash
pip install .
```

A Conda `environment.yml` should provide a convenient supported environment while installing the same underlying Python package.

The runtime dependency set should initially remain small.

Known expected dependencies are:

* NumPy;
* Astropy;
* Matplotlib.

Additional dependencies should be introduced only when required.

---

# Documentation

User-facing documentation should live under `docs/`.

The initial documentation structure should cover:

```text
installation.md
data_layout.md
virus.md
lrs2.md
quicklook.md
```

Documentation should distinguish between:

* filesystem organization;
* instrument topology;
* fiber topology;
* algorithms;
* and operational workflows.

Existing pipeline knowledge should be referenced or adapted rather than unnecessarily recreated.

---

# Testing

Testing should be part of the repository from the beginning.

The primary testing areas are expected to be:

* filesystem discovery;
* metadata extraction;
* instrument topology;
* fiber topology;
* numerical algorithms;
* higher-level workflows.

Because mature VIRUS and LRS2 pipelines and real HET observations already exist, representative real datasets can ultimately provide strong regression tests.

The exact test fixtures and reference datasets remain to be selected.

---

# Initial Dependency Direction

The desired internal dependency direction is approximately:

```text
config
  |
  v
discovery
  |
  v
metadata

instrument ----+
               |
fibers --------+----> algorithms
               |
metadata ------+
               |
               v
           workflows
               |
               v
      visualization / CLI
```

Lower-level modules should not depend upon higher-level workflows or interfaces.

In particular:

* numerical algorithms should not perform filesystem discovery;
* fiber topology should not depend upon the CLI;
* discovery should not know how quick-look diagnostics are constructed;
* visualization should consume results rather than implement the underlying scientific operation.

---

# Initial Scope

The initial implementation should establish the architecture and enough functionality to support the first operational quick-look applications.

Known initial requirements are:

* VIRUS support;
* LRS2 support;
* configurable TACC/mountain root directory;
* filesystem discovery;
* observation metadata;
* instrument topology;
* amplifier-level fiber topology;
* basic numerical algorithms required by the quick-look workflows;
* quick-look visualization;
* simple Python interfaces;
* simple command-line interfaces;
* user documentation;
* tests.

The following areas remain intentionally undefined until additional requirements or existing pipeline implementations are reviewed:

* exact metadata schema;
* exact VIRUS topology representation;
* exact LRS2 topology representation;
* exact fiber-topology representation;
* extraction algorithm;
* collapse algorithm;
* spatial reconstruction method;
* centroid algorithm;
* quick-look product data structures;
* fiducial conventions;
* detailed CLI syntax;
* future browser or web visualization architecture.

These should be filled in from known instrument behavior and existing implementation experience rather than inferred in advance.

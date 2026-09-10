# HET_LV_Tools

`HET_LV_Tools` provides small quick-look tools for operational assessment of
the HET VIRUS and LRS2 instruments. The Python package is named
`hetquicklook`.

The first implementation provides:

* discovery of observation archives under a configurable root;
* structured observation metadata, including common FITS header fields;
* explicit high-level VIRUS and LRS2 topology descriptions;
* generic amplifier-level fiber traces and IFU-plane positions;
* array-based extraction, collapse, and centroid algorithms; and
* in-memory LDLS-flat and standard-star quick-look workflows.

Install the package in an environment with:

```bash
pip install .
```

The repository's architectural scope and instrument assumptions are described
in [Architectural_Design.md](Architectural_Design.md). User documentation is
in [docs/](docs/installation.md). The implementation status, remaining tasks,
and open questions are tracked in
[`docs/implementation_workspace.md`](docs/implementation_workspace.md).

The discovery command can be used before any reduction-specific topology is
configured:

```bash
hetquicklook discover /path/to/data --instrument virus --date 20260910
```

The package deliberately accepts detector arrays and fiber topology as inputs
to its numerical workflows. Pipeline-specific trace and component mappings can
therefore be added from authoritative VIRUS and LRS2 reduction-pipeline data
without changing the algorithms.

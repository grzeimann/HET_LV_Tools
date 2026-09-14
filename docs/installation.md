# Installation

The supported Python installation uses the package definition in
`pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The same runtime dependencies can be installed with Conda:

```bash
conda env create -f environment.yml
conda activate hetquicklook
```

Run the tests with:

```bash
python -m pytest
```

The static VIRUS and LRS2 topology files, along with the dated
`Fiber_Locations` trace tree used by the high-level quick-look API, are
included as package data under `hetquicklook/resources/`. Their lookup is
independent of the caller's working directory. An alternate dated trace tree
can still be supplied explicitly to the topology loader:

```python
from hetquicklook import VirusTopologyLoader

topology = VirusTopologyLoader(trace_root="/path/to/configuration")
```

Here `/path/to/configuration/Fiber_Locations/` is the dated trace tree.

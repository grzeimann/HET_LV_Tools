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

The static VIRUS and LRS2 topology files are included as package data under
`hetquicklook/resources/`. Their lookup uses `importlib.resources`, so callers
can import the package from any working directory. Dated trace references stay
outside the package and are supplied through the topology loader:

```python
from hetquicklook import VirusTopologyLoader

topology = VirusTopologyLoader(trace_root="/path/to/configuration")
```

Here `/path/to/configuration/Fiber_Locations/` is the dated trace tree.

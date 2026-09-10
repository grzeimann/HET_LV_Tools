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


# Data layout

Discovery expects a configurable root containing date directories and
instrument directories:

```text
ROOT/
  20260910/
    virus/
      observation.tar
    lrs2/
      observation.tar
```

The HET mountain VDAS directory layout is also supported:

```text
ROOT/
  20260910/
    virus/
      virus0000001/
        exp01/
          virus/
            <FITS files>
    lrs2/
      lrs20000001/
        exp01/
          lrs2/
            <FITS files>
```

Each `virusNNNNNNN` or `lrs2NNNNNNN` directory is treated as one observation.
FITS files below it are inventoried using paths relative to that directory and
grouped by the exposure IDs encoded in their basenames.

Date directory names may use `YYYYMMDD`, `YYYY-MM-DD`, or `YYYY_MM_DD`.
Archives with `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, and `.tar.xz` suffixes are
reported. Discovery does not require companion files or a complete expected
observation set.

The VIRUS Corral form is also recognized when the configured root contains a
date tar:

```text
ROOT/
  20260910.tar
    virus/
      virus0000001.tar
        <FITS members>
```

The discovered observation retains both the date-tar path and the inner
observation-tar member. Use `load_observation()` to inventory literal members,
parse their basename identities, and load a selected primary-HDU detector
array on demand.

```python
from hetquicklook import QuicklookConfig, discover_observations

observations = discover_observations(
    QuicklookConfig("/data/het"), instrument="virus", date="20260910"
)
```

Use `hetquicklook discover ROOT --json` to inspect the records from a shell.

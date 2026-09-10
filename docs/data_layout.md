# Data layout

Discovery expects a configurable root containing date directories and
instrument directories:

```text
ROOT/
  20260910/
    virus/
      observation.tar
    lrs2/
      observation.tar.gz
```

Date directory names may use `YYYYMMDD`, `YYYY-MM-DD`, or `YYYY_MM_DD`.
Archives with `.tar`, `.tar.gz`, `.tgz`, `.tar.bz2`, and `.tar.xz` suffixes are
reported. Discovery does not require companion files or a complete expected
observation set.

```python
from hetquicklook import QuicklookConfig, discover_observations

observations = discover_observations(
    QuicklookConfig("/data/het"), instrument="virus", date="20260910"
)
```

Use `hetquicklook discover ROOT --json` to inspect the records from a shell.


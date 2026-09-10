# Quick-look workflows

The numerical workflows operate on a detector image and an amplifier-level
fiber topology. A topology contains a detector trace for each fiber and its
physical IFU-plane position.

```python
from hetquicklook.workflows import run_standard_star_quicklook

result = run_standard_star_quicklook(
    detector,
    topology,
    requested_position=(12.0, 9.0),
)
print(result.measured_centroid)
print(result.offset)
```

`run_ldls_flat_quicklook` extracts and collapses each trace, then places the
fiber values on an IFU-plane image. `run_standard_star_quicklook` performs the
same operation and computes a positive-signal weighted centroid, with an
optional requested-position fiducial.

The workflows do not read files or select observations. Those responsibilities
belong to discovery and higher-level operational code, keeping the numerical
contracts easy to test with arrays.


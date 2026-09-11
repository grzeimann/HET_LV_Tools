# VIRUS

The high-level VIRUS component hierarchy used by the package is:

```text
amplifier -> IFU -> instrument
```

The static VIRUS topology resources are packaged with `hetquicklook`:

```text
hetquicklook/resources/virus/
    IFUcen_HETDEX.txt
    IFUcen_HETDEX_reverse_R.txt
    fplaneall.txt
```

`VirusTopologyLoader` applies the established IFU-specific corrections,
amplifier slices, extracted-spectrum reversal, and focal-plane parsing. Dated
detector traces remain external calibration state:

```python
from hetquicklook import VirusTopologyLoader

loader = VirusTopologyLoader(trace_root="/path/to/configuration")
positions, position_reference = loader.fiber_positions("043", "LL")
traces, trace_reference = loader.resolve_trace_reference(identity, "20230116")
```

The fiber-position loader uses the source IFU x/y columns. The four amplifier
slices are combined only after detector extraction and collapse, when their
448 physical fiber values are reconstructed into one IFU-plane quick-look
image.

The trace root contains `Fiber_Locations/<YYYYMMDD>/`. The same resolver also
supports LRS2 physical identities because those trace files use the same
address convention.

# VIRUS

The high-level VIRUS component hierarchy used by the package is:

```text
amplifier -> IFU -> instrument
```

The package does not invent the detailed amplifier-to-IFU mapping. Detector
traces and IFU-plane positions are supplied through
`hetquicklook.fibers.FiberTopology`, using mappings derived from the
authoritative VIRUS reduction-pipeline configuration.


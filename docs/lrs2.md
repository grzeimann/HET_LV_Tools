# LRS2

The high-level LRS2 component hierarchy used by the package is:

```text
amplifier -> channel -> spectrograph -> instrument
```

The fixed LRS2 identities are:

```text
LRS2-B: SPECID 503, IFUSLOT 056, IFUID 7001
LRS2-R: SPECID 502, IFUSLOT 066, IFUID 7002
```

Static channel maps are packaged under `hetquicklook/resources/lrs2/`:

```text
LRS2_B_UV_mapping.txt
LRS2_B_OR_mapping.txt
LRS2_R_NR_mapping.txt
LRS2_R_FR_mapping.txt
```

`LRS2FiberPositionLoader` reads the established five-line-header format and
returns 140 positions per amplifier using the supplied channel/amp slices and
reversal. LRS2 dated traces use the shared external
`Fiber_Locations/<YYYYMMDD>/fiber_loc_<SPECID>_<IFUSLOT>_<IFUID>_<AMP>.txt`
resolver.

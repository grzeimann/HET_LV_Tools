"""High-level instrument topology descriptions.

The topology here describes component relationships only. Detailed amplifier
names and mappings should be supplied from the relevant reduction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Mapping

if TYPE_CHECKING:
    from .discovery import RawFrameIdentity


class Instrument(str, Enum):
    """Supported HET instruments."""

    VIRUS = "virus"
    LRS2 = "lrs2"

    @classmethod
    def from_value(cls, value: "Instrument | str") -> "Instrument":
        """Normalize an instrument name.

        Raises:
            ValueError: If ``value`` is not a supported instrument.
        """

        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {"virus": cls.VIRUS, "lrs2": cls.LRS2}
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(f"Unsupported instrument: {value!r}") from error


@dataclass(frozen=True)
class InstrumentTopology:
    """Describe the physical hierarchy of one instrument.

    ``levels`` are ordered from detector-level component to instrument-level
    component. This is intentionally separate from fiber traces and spatial
    positions, which belong to :mod:`hetquicklook.fibers`.
    """

    instrument: Instrument
    levels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument", Instrument.from_value(self.instrument))
        if not self.levels:
            raise ValueError("Instrument topology must contain at least one level")
        if len(set(self.levels)) != len(self.levels):
            raise ValueError("Instrument topology levels must be unique")
        if self.levels[-1] != "instrument":
            raise ValueError("The last topology level must be 'instrument'")


_TOPOLOGIES = {
    Instrument.VIRUS: InstrumentTopology(
        instrument=Instrument.VIRUS,
        levels=("amplifier", "ifu", "instrument"),
    ),
    Instrument.LRS2: InstrumentTopology(
        instrument=Instrument.LRS2,
        levels=("amplifier", "channel", "spectrograph", "instrument"),
    ),
}


def topology_for(instrument: Instrument | str) -> InstrumentTopology:
    """Return the high-level topology for ``instrument``."""

    return _TOPOLOGIES[Instrument.from_value(instrument)]


_LRS2_CHANNELS = {
    ("056", "LL"): "UV",
    ("056", "LU"): "UV",
    ("056", "RL"): "Orange",
    ("056", "RU"): "Orange",
    ("066", "LL"): "Red",
    ("066", "LU"): "Red",
    ("066", "RL"): "Far-Red",
    ("066", "RU"): "Far-Red",
}


@dataclass(frozen=True)
class LRS2InstrumentComponent:
    """Static identity and amplifier membership for one LRS2 channel."""

    name: str
    resource_key: str
    specid: str
    ifu_slot: str
    ifuid: str
    amplifiers: tuple[str, ...]


_LRS2_COMPONENTS = (
    LRS2InstrumentComponent("UV", "UV", "503", "056", "7001", ("LL", "LU")),
    LRS2InstrumentComponent(
        "Orange", "OR", "503", "056", "7001", ("RL", "RU")
    ),
    LRS2InstrumentComponent("Red", "NR", "502", "066", "7002", ("LL", "LU")),
    LRS2InstrumentComponent(
        "Far-Red", "FR", "502", "066", "7002", ("RL", "RU")
    ),
)


@dataclass(frozen=True)
class PhysicalAmplifierIdentity:
    """Physical address known for one raw amplifier frame.

    VIRUS identities are complete only when the FITS header supplies the
    IFUID, SPECID, and controller. LRS2 supplies its static IFUID/SPECID from
    the fixed instrument topology; controller remains header-derived when
    present.
    """

    instrument: Instrument
    ifu_slot: str
    amplifier: str
    ifuid: str | None = None
    specid: str | None = None
    controller: str | None = None

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (self.ifu_slot, self.amplifier, self.ifuid, self.specid, self.controller)
        )

    @property
    def key(self) -> str | None:
        """Return the VIRUSFlow-style address when all parts are known."""

        if not self.complete:
            return None
        return "+".join(
            (self.ifu_slot, self.ifuid or "", self.specid or "", self.amplifier, self.controller or "")
        )


def lrs2_component_for(
    ifu_slot: str, amplifier: str
) -> LRS2InstrumentComponent | None:
    """Return the fixed LRS2 component for one slot/amplifier pair."""

    slot = str(ifu_slot).strip().zfill(3)
    amp = str(amplifier).strip().upper()
    for component in _LRS2_COMPONENTS:
        if component.ifu_slot == slot and amp in component.amplifiers:
            return component
    return None


def lrs2_channel_for(ifu_slot: str, amplifier: str) -> str | None:
    """Return the supplied LRS2 slot/amplifier channel mapping.

    This is deliberately a small interpretation layer over the generic raw
    identity. Unknown tokens return ``None`` rather than being classified by
    inference.
    """

    component = lrs2_component_for(ifu_slot, amplifier)
    return component.name if component is not None else None


def lrs2_channel_for_identity(identity: RawFrameIdentity) -> str | None:
    """Interpret an encoded raw identity using the known LRS2 topology."""

    return lrs2_channel_for(identity.ifu_slot, identity.amplifier)


def physical_identity_for(
    instrument: Instrument | str,
    identity: RawFrameIdentity,
    header: Mapping[str, Any] | None = None,
) -> PhysicalAmplifierIdentity:
    """Combine raw filename identity with supported physical metadata."""

    parsed_instrument = Instrument.from_value(instrument)
    header = header or {}
    if parsed_instrument is Instrument.LRS2:
        component = lrs2_component_for(identity.ifu_slot, identity.amplifier)
        return PhysicalAmplifierIdentity(
            instrument=parsed_instrument,
            ifu_slot=identity.ifu_slot,
            amplifier=identity.amplifier,
            ifuid=component.ifuid if component else None,
            specid=component.specid if component else None,
            controller=_header_text(header, "CONTID", "CONTROLLER"),
        )
    return PhysicalAmplifierIdentity(
        instrument=parsed_instrument,
        ifu_slot=identity.ifu_slot,
        amplifier=identity.amplifier,
        ifuid=_header_text(header, "IFUID"),
        specid=_header_text(header, "SPECID"),
        controller=_header_text(header, "CONTID", "CONTROLLER"),
    )


def _header_text(header: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = header.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return None

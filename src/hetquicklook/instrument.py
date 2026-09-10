"""High-level instrument topology descriptions.

The topology here describes component relationships only. Detailed amplifier
names and mappings should be supplied from the relevant reduction pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

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


def lrs2_channel_for(ifu_slot: str, amplifier: str) -> str | None:
    """Return the supplied LRS2 slot/amplifier channel mapping.

    This is deliberately a small interpretation layer over the generic raw
    identity. Unknown tokens return ``None`` rather than being classified by
    inference.
    """

    return _LRS2_CHANNELS.get((str(ifu_slot), str(amplifier).upper()))


def lrs2_channel_for_identity(identity: RawFrameIdentity) -> str | None:
    """Interpret an encoded raw identity using the known LRS2 topology."""

    return lrs2_channel_for(identity.ifu_slot, identity.amplifier)

"""Configuration shared by discovery and higher-level workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .instrument import Instrument


@dataclass(frozen=True)
class QuicklookConfig:
    """Filesystem configuration for HET quick-look data.

    Args:
        root: Directory containing date directories. The path does not need to
            exist when the configuration is created; discovery reports an
            empty result for a missing root.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).expanduser())

    def instrument_directory(self, date: str, instrument: Instrument | str) -> Path:
        """Return the directory for one date and instrument."""

        parsed_instrument = Instrument.from_value(instrument)
        return self.root / str(date) / parsed_instrument.value


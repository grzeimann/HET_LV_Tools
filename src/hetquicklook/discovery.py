"""Filesystem discovery for observation archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path
import re
from typing import Iterator

from .config import QuicklookConfig
from .instrument import Instrument


_ARCHIVE_SUFFIXES = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")
_DATE_PATTERNS = (re.compile(r"^(\d{8})$"), re.compile(r"^(\d{4})[-_](\d{2})[-_](\d{2})$"))


@dataclass(frozen=True)
class DiscoveredObservation:
    """One observation archive found on disk.

    The object records only what discovery can establish without opening the
    archive. Missing companion files do not invalidate this observation.
    """

    archive_path: Path
    date: calendar_date
    instrument: Instrument

    def __post_init__(self) -> None:
        object.__setattr__(self, "instrument", Instrument.from_value(self.instrument))

    @property
    def observation_id(self) -> str:
        """Return a stable identifier derived from the archive filename."""

        name = self.archive_path.name
        for suffix in _ARCHIVE_SUFFIXES:
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return self.archive_path.stem


def _parse_date(value: str) -> calendar_date | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.fullmatch(value)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 1:
            return calendar_date(
                int(groups[0][:4]), int(groups[0][4:6]), int(groups[0][6:])
            )
        return calendar_date(int(groups[0]), int(groups[1]), int(groups[2]))
    return None


def _date_directories(root: Path) -> Iterator[tuple[Path, calendar_date]]:
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if path.is_dir():
            parsed = _parse_date(path.name)
            if parsed is not None:
                yield path, parsed


def _is_archive(path: Path) -> bool:
    return any(path.name.lower().endswith(suffix) for suffix in _ARCHIVE_SUFFIXES)


def discover_observations(
    config: QuicklookConfig,
    *,
    instrument: Instrument | str | None = None,
    date: calendar_date | str | None = None,
) -> tuple[DiscoveredObservation, ...]:
    """Discover observation archives below a configured root.

    Args:
        config: Filesystem configuration.
        instrument: Optional instrument filter.
        date: Optional date filter. Accepted string forms are ``YYYYMMDD`` and
            ``YYYY-MM-DD``.

    Returns:
        Observations sorted by observing date and archive path. Every returned
        object corresponds to an archive that exists; no completeness check is
        performed.

    Raises:
        ValueError: If ``date`` cannot be parsed.
    """

    selected_instrument = Instrument.from_value(instrument) if instrument else None
    selected_date: date | None
    if date is None or isinstance(date, calendar_date):
        selected_date = date
    else:
        selected_date = _parse_date(str(date))
        if selected_date is None:
            raise ValueError(f"Unsupported date format: {date!r}")

    observations: list[DiscoveredObservation] = []
    for date_dir, observed_date in _date_directories(config.root):
        if selected_date is not None and observed_date != selected_date:
            continue
        if selected_instrument:
            instrument_dirs = [
                path
                for path in sorted(date_dir.iterdir())
                if path.is_dir()
                and path.name.lower() == selected_instrument.value
            ]
        else:
            instrument_dirs = [
                path for path in sorted(date_dir.iterdir()) if path.is_dir()
            ]
        for instrument_dir in instrument_dirs:
            if not instrument_dir.is_dir():
                continue
            try:
                observed_instrument = Instrument.from_value(instrument_dir.name)
            except ValueError:
                continue
            for archive_path in sorted(instrument_dir.iterdir()):
                if archive_path.is_file() and _is_archive(archive_path):
                    observations.append(
                        DiscoveredObservation(
                            archive_path=archive_path,
                            date=observed_date,
                            instrument=observed_instrument,
                        )
                    )
    return tuple(observations)

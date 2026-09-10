"""Structured metadata for discovered observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import io
from pathlib import Path
import tarfile
from typing import Any, Mapping

from .discovery import DiscoveredObservation
from .instrument import Instrument


def _header_value(header: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in header and header[key] not in (None, ""):
            return header[key]
    return None


def _float_header_value(header: Mapping[str, Any], *keys: str) -> float | None:
    value = _header_value(header, *keys)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class ObservationMetadata:
    """Metadata known for one discovered observation.

    Values absent from the source archive are represented by ``None``. The
    ``files`` tuple contains only archive members that were actually found.
    """

    observation_id: str
    archive_path: Path
    date: date
    instrument: Instrument
    files: tuple[str, ...] = ()
    object_name: str | None = None
    program: str | None = None
    exposure_time_s: float | None = None
    requested_ra_deg: float | None = None
    requested_dec_deg: float | None = None
    observation_time: datetime | str | None = None
    header_values: Mapping[str, Any] = field(default_factory=dict)


def metadata_from_header(
    observation: DiscoveredObservation,
    header: Mapping[str, Any],
    *,
    files: tuple[str, ...] = (),
) -> ObservationMetadata:
    """Create observation metadata from a FITS-like header mapping.

    The aliases cover common HET operational header spellings while preserving
    the original header values in ``header_values`` for later refinement.
    """

    date_obs = _header_value(header, "DATE-OBS", "DATEOBS", "UTSTART", "DATE")
    try:
        observation_time: datetime | str | None = (
            datetime.fromisoformat(str(date_obs).replace("Z", "+00:00"))
            if date_obs is not None
            else None
        )
    except ValueError:
        observation_time = str(date_obs) if date_obs is not None else None

    return ObservationMetadata(
        observation_id=observation.observation_id,
        archive_path=observation.archive_path,
        date=observation.date,
        instrument=observation.instrument,
        files=tuple(files),
        object_name=_header_value(header, "OBJECT", "OBJNAME", "TARGET"),
        program=_header_value(header, "PROGID", "PROGRAM", "OBS_PROG", "PROGRAMID"),
        exposure_time_s=_float_header_value(
            header, "EXPTIME", "EXPOSURE", "EXPOSURE_TIME"
        ),
        requested_ra_deg=_float_header_value(
            header, "REQRA", "RA", "TELRA", "CAT-RA"
        ),
        requested_dec_deg=_float_header_value(
            header, "REQDEC", "DEC", "TELDEC", "CAT-DEC"
        ),
        observation_time=observation_time,
        header_values=dict(header),
    )


def metadata_from_fits_header(
    observation: DiscoveredObservation,
    header: Mapping[str, Any],
    *,
    files: tuple[str, ...] = (),
) -> ObservationMetadata:
    """Backward-compatible descriptive alias for :func:`metadata_from_header`."""

    return metadata_from_header(observation, header, files=files)


def metadata_from_archive(observation: DiscoveredObservation) -> ObservationMetadata:
    """Read the first FITS header in an observation archive.

    Archive members are listed even when no FITS member is present. This keeps
    absence of a header discoverable without turning it into a completeness
    failure.
    """

    try:
        from astropy.io import fits
    except ImportError as error:  # pragma: no cover - dependency is declared
        raise RuntimeError("Astropy is required to read FITS metadata") from error

    with tarfile.open(observation.archive_path, mode="r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        names = tuple(member.name for member in members)
        for member in members:
            lower_name = member.name.lower()
            if not lower_name.endswith((".fits", ".fits.gz", ".fit", ".fit.gz")):
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            with fits.open(io.BytesIO(extracted.read()), memmap=False) as hdul:
                return metadata_from_header(
                    observation, hdul[0].header, files=names
                )
    return metadata_from_header(observation, {}, files=names)

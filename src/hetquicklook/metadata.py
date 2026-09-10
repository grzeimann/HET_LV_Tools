"""Structured metadata for discovered observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import math
from pathlib import Path
from collections.abc import Callable, Iterable
from typing import Any, Mapping

from .classification import ObjectIntent, parse_object_intent
from .discovery import ArchiveMember, DiscoveredObservation, inventory_members
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


def _text_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _finite_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _datetime_value(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, datetime.min.time())
    elif value is None:
        return None
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


@dataclass(frozen=True)
class MetadataDisagreement:
    """One exposure-level field with multiple normalized header values."""

    field: str
    values: tuple[Any, ...]
    members: tuple[str, ...]


@dataclass(frozen=True)
class ExposureMetadata:
    """Normalized metadata and interpreted intent for one exposure ID."""

    exposure_id: str
    frame_types: tuple[str, ...]
    frame_class: str | None
    header_values: Mapping[str, Any] = field(default_factory=dict)
    header_members: tuple[str, ...] = ()
    disagreements: tuple[MetadataDisagreement, ...] = ()
    observation_time: datetime | None = None
    airmass: float | None = None
    ambient_temperature: float | None = None
    humidity: float | None = None
    pressure: float | None = None
    exposure_time_s: float | None = None
    planned_exposure_time_s: float | None = None
    program_id: str | None = None
    object_name: str | None = None
    qobject: str | None = None
    qra: str | None = None
    qdec: str | None = None
    qprog: str | None = None
    requested_ra_deg: float | None = None
    requested_dec_deg: float | None = None
    object_target: str | None = None
    requested_target: str | None = None
    requested_target_source: str | None = None
    requested_ifuslot: str | None = None
    het_track: str | None = None
    observing_mode: str | None = None
    virus_primary: bool | None = None
    q_metadata_expected: bool = False
    q_metadata_complete: bool | None = None
    object_qobject_consistent: bool | None = None
    scientific_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def frame_type(self) -> str | None:
        """Return the sole frame type when the exposure is homogeneous."""

        return self.frame_types[0] if len(self.frame_types) == 1 else None

    @property
    def metadata_disagreements(self) -> tuple[MetadataDisagreement, ...]:
        """Descriptive alias for :attr:`disagreements`."""

        return self.disagreements


_EXPOSURE_FIELDS: dict[str, tuple[tuple[str, ...], Callable[[Any], Any]]] = {
    "observation_time": (("DATE", "DATE-OBS", "DATEOBS", "UTSTART"), _datetime_value),
    "airmass": (("AIRMASS",), _finite_float),
    "ambient_temperature": (("AMBTEMP", "AMBIENT", "TAMBIENT", "TEMPAMB", "OUTTEMP"), _finite_float),
    "humidity": (("HUMIDITY",), _finite_float),
    "pressure": (("PRESSURE",), _finite_float),
    "exposure_time_s": (("EXPTIME",), _finite_float),
    "planned_exposure_time_s": (("PEXPTIME",), _finite_float),
    "program_id": (("QPROG",), _text_value),
    "object_name": (("OBJECT",), _text_value),
    "qobject": (("QOBJECT",), _text_value),
    "qra": (("QRA",), _text_value),
    "qdec": (("QDEC",), _text_value),
    "qprog": (("QPROG",), _text_value),
    "rho_start": (("RHO_STRT",), _finite_float),
    "theta_start": (("THE_STRT",), _finite_float),
    "phi_start": (("PHI_STRT",), _finite_float),
    "x_start": (("X_STRT",), _finite_float),
    "y_start": (("Y_STRT",), _finite_float),
}


def _first_header_value(header: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = header.get(key)
        if value not in (None, ""):
            return value
    return None


def _field_value_and_disagreement(
    field: str,
    headers: tuple[tuple[str, Mapping[str, Any]], ...],
) -> tuple[Any, MetadataDisagreement | None]:
    keys, normalizer = _EXPOSURE_FIELDS[field]
    entries: list[tuple[str, Any]] = []
    for member_name, header in headers:
        value = normalizer(_first_header_value(header, keys))
        if value is not None:
            entries.append((member_name, value))
    if not entries:
        return None, None
    values: list[Any] = []
    members: list[str] = []
    for member_name, value in entries:
        if value not in values:
            values.append(value)
        members.append(member_name)
    disagreement = (
        MetadataDisagreement(field, tuple(values), tuple(members))
        if len(values) > 1
        else None
    )
    return entries[0][1], disagreement


def exposure_metadata_from_headers(
    exposure_id: str,
    frame_types: Iterable[str],
    headers: Mapping[str, Mapping[str, Any]],
) -> ExposureMetadata:
    """Build exposure metadata from all successfully read member headers.

    Values are normalized from the first header containing each field. If
    later headers disagree after normalization, the disagreement is retained
    in the result rather than being overwritten.
    """

    normalized_frame_types = tuple(sorted({str(value).strip() for value in frame_types}))
    frame_classes = {
        "science" if value.casefold() == "sci" else "calibration"
        for value in normalized_frame_types
    }
    frame_class = next(iter(frame_classes)) if len(frame_classes) == 1 else (
        "mixed" if frame_classes else None
    )
    header_items = tuple(headers.items())
    values: dict[str, Any] = {}
    disagreements: list[MetadataDisagreement] = []
    for field in _EXPOSURE_FIELDS:
        value, disagreement = _field_value_and_disagreement(field, header_items)
        values[field] = value
        if disagreement is not None:
            disagreements.append(disagreement)

    object_intent: ObjectIntent = parse_object_intent(values["object_name"])
    qobject = values["qobject"]
    requested_target = qobject or object_intent.target
    requested_target_source = "QOBJECT" if qobject else (
        "OBJECT_prefix" if object_intent.target else None
    )
    q_metadata_expected = frame_class == "science"
    q_metadata_complete = (
        all(values[field] is not None for field in ("qobject", "qra", "qdec", "qprog"))
        if q_metadata_expected
        else None
    )
    object_qobject_consistent = (
        object_intent.target == qobject
        if object_intent.target is not None and qobject is not None
        else None
    )
    if frame_class == "calibration":
        observing_mode = "calibration"
        virus_primary = None
    elif object_intent.is_parallel:
        observing_mode = "parallel"
        virus_primary = False
    else:
        observing_mode = "primary"
        virus_primary = True

    scientific_metadata = {
        "observation_time": values["observation_time"],
        "airmass": values["airmass"],
        "ambient_temperature": values["ambient_temperature"],
        "humidity": values["humidity"],
        "pressure": values["pressure"],
        "program_id": values["program_id"],
        "object": values["object_name"],
        "rho_start": values["rho_start"],
        "theta_start": values["theta_start"],
        "phi_start": values["phi_start"],
        "x_start": values["x_start"],
        "y_start": values["y_start"],
    }
    first_header = dict(headers[next(iter(headers))]) if headers else {}
    return ExposureMetadata(
        exposure_id=str(exposure_id),
        frame_types=normalized_frame_types,
        frame_class=frame_class,
        header_values=first_header,
        header_members=tuple(headers),
        disagreements=tuple(disagreements),
        observation_time=values["observation_time"],
        airmass=values["airmass"],
        ambient_temperature=values["ambient_temperature"],
        humidity=values["humidity"],
        pressure=values["pressure"],
        exposure_time_s=values["exposure_time_s"],
        planned_exposure_time_s=values["planned_exposure_time_s"],
        program_id=values["program_id"],
        object_name=values["object_name"],
        qobject=qobject,
        qra=values["qra"],
        qdec=values["qdec"],
        qprog=values["qprog"],
        requested_ra_deg=_finite_float(values["qra"]),
        requested_dec_deg=_finite_float(values["qdec"]),
        object_target=object_intent.target,
        requested_target=requested_target,
        requested_target_source=requested_target_source,
        requested_ifuslot=object_intent.requested_ifuslot,
        het_track=object_intent.track,
        observing_mode=observing_mode,
        virus_primary=virus_primary,
        q_metadata_expected=q_metadata_expected,
        q_metadata_complete=q_metadata_complete,
        object_qobject_consistent=object_qobject_consistent,
        scientific_metadata=scientific_metadata,
    )
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
    archive_members: tuple[ArchiveMember, ...] = ()
    exposure_ids: tuple[str, ...] = ()


def metadata_from_header(
    observation: DiscoveredObservation,
    header: Mapping[str, Any],
    *,
    files: tuple[str, ...] = (),
    archive_members: tuple[ArchiveMember, ...] = (),
    exposure_ids: tuple[str, ...] = (),
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
        archive_members=tuple(archive_members),
        exposure_ids=tuple(exposure_ids),
    )


def metadata_from_fits_header(
    observation: DiscoveredObservation,
    header: Mapping[str, Any],
    *,
    files: tuple[str, ...] = (),
    archive_members: tuple[ArchiveMember, ...] = (),
    exposure_ids: tuple[str, ...] = (),
) -> ObservationMetadata:
    """Backward-compatible descriptive alias for :func:`metadata_from_header`."""

    return metadata_from_header(
        observation,
        header,
        files=files,
        archive_members=archive_members,
        exposure_ids=exposure_ids,
    )


def metadata_from_members(
    observation: DiscoveredObservation,
    members: tuple[ArchiveMember, ...],
) -> ObservationMetadata:
    """Build an archive summary without borrowing a header across exposures.

    This compatibility helper retains a header only when the inventory
    contains exactly one parsed exposure. Multi-exposure callers should use
    :func:`hetquicklook.load_observation` to obtain per-exposure metadata.
    """

    from .raw import RawFrameLoader

    header: Mapping[str, Any] = {}
    exposure_ids = tuple(
        sorted(
            {
                member.identity.exposure_id
                for member in members
                if member.identity is not None
            }
        )
    )
    if len(exposure_ids) == 1:
        first_fits = next(
            (
                member
                for member in members
                if member.is_fits
                and member.identity is not None
                and member.identity.exposure_id == exposure_ids[0]
            ),
            None,
        )
        if first_fits is not None:
            header = RawFrameLoader().read_header(first_fits)
    return metadata_from_header(
        observation,
        header,
        files=tuple(member.member_name for member in members),
        archive_members=members,
        exposure_ids=exposure_ids,
    )


def metadata_from_archive(observation: DiscoveredObservation) -> ObservationMetadata:
    """Build an archive summary from an inventoried observation archive.

    Archive members are listed even when no FITS member is present. This keeps
    absence of a header discoverable without turning it into a completeness
    failure. A multi-exposure archive does not borrow one exposure's header.
    """

    return metadata_from_members(observation, inventory_members(observation))

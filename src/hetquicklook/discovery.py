"""Filesystem discovery for observation archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as calendar_date
from pathlib import Path
from pathlib import PurePosixPath
import re
import tarfile
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
    outer_tar_member: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "archive_path", Path(self.archive_path))
        object.__setattr__(self, "instrument", Instrument.from_value(self.instrument))

    @property
    def observation_id(self) -> str:
        """Return a stable identifier derived from the archive filename."""

        name = (
            PurePosixPath(self.outer_tar_member).name
            if self.outer_tar_member is not None
            else self.archive_path.name
        )
        for suffix in _ARCHIVE_SUFFIXES:
            if name.lower().endswith(suffix):
                return name[: -len(suffix)]
        return self.archive_path.stem

    @property
    def storage_backend(self) -> str:
        """Return the small storage classification needed to load members."""

        return "date_tar" if self.outer_tar_member is not None else "tar"


@dataclass(frozen=True)
class RawFrameIdentity:
    """Identity encoded by a VIRUS or LRS2 FITS basename.

    The field meanings follow VIRUSFlow's parser. ``amplifier_token`` is kept
    as the original token, while ``ifu_slot`` and ``amplifier`` expose its
    established three-character-plus-suffix decomposition.
    """

    exposure_id: str
    amplifier_token: str
    frame_type: str

    @property
    def ifu_slot(self) -> str:
        """Return the first three characters of the amplifier token."""

        return self.amplifier_token[:3]

    @property
    def amplifier(self) -> str:
        """Return the amplifier suffix of the amplifier token."""

        return self.amplifier_token[3:]

    # These aliases use the names in the VIRUSFlow database code and make the
    # raw identity convenient to use when comparing pipeline records.
    @property
    def amp_token(self) -> str:
        """Alias for :attr:`amplifier_token`."""

        return self.amplifier_token

    @property
    def ifuslot(self) -> str:
        """Alias for :attr:`ifu_slot`."""

        return self.ifu_slot

    @property
    def amp(self) -> str:
        """Alias for :attr:`amplifier`."""

        return self.amplifier


@dataclass(frozen=True)
class ArchiveMember:
    """One literal regular file member found in an observation archive.

    ``archive_path`` is the physical outer tar path. For a nested Corral
    layout, ``outer_tar_member`` identifies the inner observation tar and
    ``member_name`` identifies the FITS member within it. ``identity`` is
    ``None`` for non-FITS or malformed names; the member remains in the
    inventory so that the filesystem evidence is not hidden.
    """

    archive_path: Path
    member_name: str
    size: int
    outer_tar_member: str | None = None
    identity: RawFrameIdentity | None = None
    parse_error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "archive_path", Path(self.archive_path))
        object.__setattr__(self, "size", int(self.size))

    @property
    def member_basename(self) -> str:
        """Return the basename used for identity parsing."""

        return PurePosixPath(self.member_name).name

    @property
    def tar_member(self) -> str:
        """Alias matching the raw pipeline's member terminology."""

        return self.member_name

    @property
    def outer_member(self) -> str | None:
        """Alias for the nested observation-tar member, when present."""

        return self.outer_tar_member

    @property
    def is_fits(self) -> bool:
        """Whether this member has a FITS filename suffix."""

        return self.member_basename.lower().endswith(
            (".fits", ".fits.gz", ".fit", ".fit.gz")
        )


def _identity_parse(member_name: str) -> tuple[RawFrameIdentity | None, str | None]:
    """Parse one basename using VIRUSFlow's established minimum contract."""

    basename = PurePosixPath(member_name).name
    if not basename.lower().endswith(".fits"):
        return None, "member is not a .fits file"
    parts = basename[:-5].split("_")
    if len(parts) < 3:
        return None, "FITS basename has fewer than three underscore fields"
    amplifier_token = parts[1]
    if len(amplifier_token) < 5:
        return None, "amplifier token has fewer than five characters"
    return (
        RawFrameIdentity(
            exposure_id=parts[0],
            amplifier_token=amplifier_token,
            frame_type=parts[2],
        ),
        None,
    )


def parse_member_identity(member_name: str) -> RawFrameIdentity | None:
    """Parse a FITS member basename into its encoded raw-frame identity.

    The parser follows VIRUSFlow's ``_parse_virus_member_name`` behavior and
    returns ``None`` for an unrecognized name. Use :func:`inventory_members`
    when the parse error itself must remain observable.
    """

    identity, _ = _identity_parse(member_name)
    return identity


def _archive_member(
    archive_path: Path,
    member: tarfile.TarInfo,
    *,
    outer_tar_member: str | None,
) -> ArchiveMember:
    identity, parse_error = _identity_parse(member.name)
    return ArchiveMember(
        archive_path=archive_path,
        member_name=member.name,
        size=member.size,
        outer_tar_member=outer_tar_member,
        identity=identity,
        parse_error=parse_error,
    )


def inventory_members(observation: DiscoveredObservation) -> tuple[ArchiveMember, ...]:
    """Inventory literal regular members without loading FITS detector data.

    Direct archives are opened once and nested Corral archives are opened at
    both tar levels. Results are sorted by literal member names for stable
    inspection and testing.
    """

    archive_path = observation.archive_path
    with tarfile.open(archive_path, mode="r:*") as outer:
        if observation.outer_tar_member is None:
            members = [
                _archive_member(archive_path, member, outer_tar_member=None)
                for member in outer.getmembers()
                if member.isfile()
            ]
        else:
            nested = outer.getmember(observation.outer_tar_member)
            stream = outer.extractfile(nested)
            if stream is None:
                raise FileNotFoundError(
                    f"Cannot extract {observation.outer_tar_member} from {archive_path}"
                )
            with tarfile.open(fileobj=stream, mode="r:*") as inner:
                members = [
                    _archive_member(
                        archive_path,
                        member,
                        outer_tar_member=observation.outer_tar_member,
                    )
                    for member in inner.getmembers()
                    if member.isfile()
                ]
    return tuple(sorted(members, key=lambda item: item.member_name))


# A descriptive alias keeps call sites readable when the caller already knows
# that the input is one archive rather than a complete observation object.
inventory_archive = inventory_members


def _parse_date(value: str) -> calendar_date | None:
    for pattern in _DATE_PATTERNS:
        match = pattern.fullmatch(value)
        if not match:
            continue
        groups = match.groups()
        try:
            if len(groups) == 1:
                return calendar_date(
                    int(groups[0][:4]), int(groups[0][4:6]), int(groups[0][6:])
                )
            return calendar_date(int(groups[0]), int(groups[1]), int(groups[2]))
        except ValueError:
            return None
    return None


def _date_directories(root: Path) -> Iterator[tuple[Path, calendar_date]]:
    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if path.is_dir():
            parsed = _parse_date(path.name)
            if parsed is not None:
                yield path, parsed


def _date_archives(root: Path) -> Iterator[tuple[Path, calendar_date]]:
    """Yield date-tar containers from the Corral layout."""

    if not root.is_dir():
        return
    for path in sorted(root.iterdir()):
        if not path.is_file() or not path.name.lower().endswith(".tar"):
            continue
        parsed = _parse_date(path.name[:-4])
        if parsed is not None:
            yield path, parsed


def _is_nested_virus_archive(name: str) -> bool:
    """Match the nested VIRUS tar convention used by VIRUSFlow."""

    parts = PurePosixPath(name).parts
    return (
        any(part.lower() == "virus" for part in parts[:-1])
        and parts[-1].lower().startswith("virus")
        and parts[-1].lower().endswith(".tar")
    )


def _discover_nested_observations(
    archive_path: Path,
    observed_date: calendar_date,
) -> Iterator[DiscoveredObservation]:
    """Discover inner VIRUS observation tars in one date tar."""

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
    except (OSError, tarfile.TarError):
        return
    for member in sorted(members, key=lambda item: item.name):
        if member.isfile() and _is_nested_virus_archive(member.name):
            yield DiscoveredObservation(
                archive_path=archive_path,
                date=observed_date,
                instrument=Instrument.VIRUS,
                outer_tar_member=member.name,
            )


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
    for date_archive, observed_date in _date_archives(config.root):
        if selected_date is not None and observed_date != selected_date:
            continue
        if selected_instrument not in (None, Instrument.VIRUS):
            continue
        observations.extend(_discover_nested_observations(date_archive, observed_date))
    observations.sort(
        key=lambda item: (
            item.date,
            str(item.archive_path),
            item.outer_tar_member or "",
        )
    )
    return tuple(observations)

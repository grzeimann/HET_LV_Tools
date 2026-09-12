"""Representation of one discovered observation and its raw frames."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from .classification import ExposureClassification, StandardStarCatalog, classify_exposure
from .discovery import ArchiveMember, DiscoveredObservation, inventory_members
from .instrument import PhysicalAmplifierIdentity, physical_identity_for
from .metadata import (
    ExposureMetadata,
    ObservationMetadata,
    exposure_metadata_from_headers,
    metadata_from_header,
)
from .raw import RawFrameData, RawFrameLoader


@dataclass(frozen=True)
class Exposure:
    """One encoded exposure and only the raw frames found for it."""

    exposure_id: str
    frames: tuple[ArchiveMember, ...]
    metadata: ExposureMetadata
    classification: ExposureClassification
    physical_identities: Mapping[str, PhysicalAmplifierIdentity]
    header_errors: tuple[tuple[str, str], ...] = ()

    @property
    def frame_types(self) -> tuple[str, ...]:
        return self.metadata.frame_types

    @property
    def amplifier_tokens(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    member.identity.amplifier_token
                    for member in self.frames
                    if member.identity is not None
                }
            )
        )

    def identity_for(self, member: ArchiveMember) -> PhysicalAmplifierIdentity:
        """Return the physical identity associated with an inventoried frame."""

        try:
            return self.physical_identities[member.member_name]
        except KeyError as error:
            raise ValueError("member does not belong to this exposure") from error

    def load_frame(
        self,
        member: ArchiveMember,
        *,
        loader: RawFrameLoader | None = None,
    ) -> RawFrameData:
        if member not in self.frames:
            raise ValueError("member does not belong to this exposure")
        return (loader or RawFrameLoader()).load(member)

    def load_frames(self, *, loader: RawFrameLoader | None = None) -> tuple[RawFrameData, ...]:
        """Load all detector frames belonging to this exposure on demand."""

        active_loader = loader or RawFrameLoader()
        return tuple(self.load_frame(member, loader=active_loader) for member in self.frames)


@dataclass(frozen=True)
class Observation:
    """The source evidence and basic metadata for one observation.

    ``members`` contains every regular file in an archive or observation
    directory. ``frames`` filters that inventory to FITS-looking members,
    including malformed FITS names so their presence remains inspectable.
    Grouping methods use only members with a parsed
    :class:`~hetquicklook.discovery.RawFrameIdentity`.
    """

    discovered: DiscoveredObservation
    members: tuple[ArchiveMember, ...]
    metadata: ObservationMetadata
    exposures: tuple[Exposure, ...] = ()

    @property
    def observation_id(self) -> str:
        """Return the discovery-level observation identifier."""

        return self.discovered.observation_id

    @property
    def archive_path(self) -> Path:
        """Return the physical outer archive path."""

        return self.discovered.archive_path

    @property
    def frames(self) -> tuple[ArchiveMember, ...]:
        """Return all inventoried FITS members, whether parsed or malformed."""

        return tuple(member for member in self.members if member.is_fits)

    @property
    def parsed_frames(self) -> tuple[ArchiveMember, ...]:
        """Return FITS members with a recognized raw-frame identity."""

        return tuple(member for member in self.frames if member.identity is not None)

    @property
    def malformed_frames(self) -> tuple[ArchiveMember, ...]:
        """Return FITS members whose basenames did not satisfy the parser."""

        return tuple(member for member in self.frames if member.identity is None)

    @property
    def exposure_ids(self) -> tuple[str, ...]:
        """Return exposure IDs actually encoded by the inventoried members."""

        return tuple(exposure.exposure_id for exposure in self.exposures)

    def exposure_for(self, exposure_id: str) -> Exposure:
        """Return one exposure by its encoded identifier."""

        for exposure in self.exposures:
            if exposure.exposure_id == exposure_id:
                return exposure
        raise KeyError(exposure_id)

    def frames_for(
        self,
        *,
        exposure_id: str | None = None,
        amplifier_token: str | None = None,
        frame_type: str | None = None,
    ) -> tuple[ArchiveMember, ...]:
        """Select parsed frames by identity fields actually encoded in names."""

        selected: list[ArchiveMember] = []
        for member in self.parsed_frames:
            identity = member.identity
            assert identity is not None
            if exposure_id is not None and identity.exposure_id != exposure_id:
                continue
            if (
                amplifier_token is not None
                and identity.amplifier_token != amplifier_token
            ):
                continue
            if frame_type is not None and identity.frame_type != frame_type:
                continue
            selected.append(member)
        return tuple(selected)

    def group_by_exposure(self) -> dict[str, tuple[ArchiveMember, ...]]:
        """Group parsed frames by their encoded exposure identifier."""

        return self._group_by(lambda member: member.identity.exposure_id)  # type: ignore[union-attr]

    def group_by_amplifier(self) -> dict[str, tuple[ArchiveMember, ...]]:
        """Group parsed frames by their encoded amplifier token."""

        return self._group_by(lambda member: member.identity.amplifier_token)  # type: ignore[union-attr]

    def group_by_frame_type(self) -> dict[str, tuple[ArchiveMember, ...]]:
        """Group parsed frames by their encoded frame type."""

        return self._group_by(lambda member: member.identity.frame_type)  # type: ignore[union-attr]

    def _group_by(
        self,
        key_function: Callable[[ArchiveMember], str],
    ) -> dict[str, tuple[ArchiveMember, ...]]:
        groups: dict[str, list[ArchiveMember]] = {}
        for member in self.parsed_frames:
            identity = member.identity
            if identity is None:
                continue
            key = key_function(member)
            groups.setdefault(key, []).append(member)
        return {
            key: tuple(groups[key])
            for key in sorted(groups)
        }

    def load_frame(
        self,
        member: ArchiveMember,
        *,
        loader: RawFrameLoader | None = None,
    ) -> RawFrameData:
        """Load one inventoried detector frame and preserve its provenance."""

        if member not in self.members:
            raise ValueError("member does not belong to this observation")
        return (loader or RawFrameLoader()).load(member)

    def load_frames(
        self,
        members: Iterable[ArchiveMember] | None = None,
        *,
        exposure_id: str | None = None,
        amplifier_token: str | None = None,
        frame_type: str | None = None,
        loader: RawFrameLoader | None = None,
    ) -> tuple[RawFrameData, ...]:
        """Load selected detector frames from this observation.

        If ``members`` is omitted, the identity filters select parsed frames.
        Arrays are loaded only by this method or :meth:`load_frame`.
        """

        selected = (
            tuple(members)
            if members is not None
            else self.frames_for(
                exposure_id=exposure_id,
                amplifier_token=amplifier_token,
                frame_type=frame_type,
            )
        )
        active_loader = loader or RawFrameLoader()
        return tuple(self.load_frame(member, loader=active_loader) for member in selected)

    @classmethod
    def from_discovered(
        cls,
        discovered: DiscoveredObservation,
        *,
        standard_catalog: StandardStarCatalog | None = None,
    ) -> "Observation":
        """Inventory one archive and build exposure-level header metadata."""

        members = inventory_members(discovered)
        parsed_frames = tuple(member for member in members if member.identity is not None)
        header_results = RawFrameLoader().read_headers(parsed_frames)
        headers_by_member = {
            result.member.member_name: result.header
            for result in header_results
            if result.header is not None
        }
        groups: dict[str, list[ArchiveMember]] = {}
        for member in parsed_frames:
            identity = member.identity
            assert identity is not None
            groups.setdefault(identity.exposure_id, []).append(member)

        exposures: list[Exposure] = []
        for exposure_id in sorted(groups):
            frames = tuple(groups[exposure_id])
            frame_types = tuple(
                member.identity.frame_type
                for member in frames
                if member.identity is not None
            )
            exposure_headers = {
                member.member_name: headers_by_member[member.member_name]
                for member in frames
                if member.member_name in headers_by_member
            }
            exposure_metadata = exposure_metadata_from_headers(
                exposure_id,
                frame_types,
                exposure_headers,
            )
            slots = tuple(
                member.identity.ifu_slot
                for member in frames
                if member.identity is not None
            )
            classification = classify_exposure(
                discovered.instrument,
                frame_types=frame_types,
                object_name=exposure_metadata.object_name,
                ifu_slots=slots,
                standard_catalog=standard_catalog,
            )
            identities = {
                member.member_name: physical_identity_for(
                    discovered.instrument,
                    member.identity,
                    headers_by_member.get(member.member_name),
                )
                for member in frames
                if member.identity is not None
            }
            errors = tuple(
                (result.member.member_name, result.error or "header read failed")
                for result in header_results
                if result.error is not None and result.member in frames
            )
            exposures.append(
                Exposure(
                    exposure_id=exposure_id,
                    frames=frames,
                    metadata=exposure_metadata,
                    classification=classification,
                    physical_identities=identities,
                    header_errors=errors,
                )
            )

        summary_header: Mapping[str, object] = {}
        if len(exposures) == 1 and exposures[0].metadata.header_values:
            summary_header = exposures[0].metadata.header_values
        metadata = metadata_from_header(
            discovered,
            summary_header,
            files=tuple(member.member_name for member in members),
            archive_members=members,
            exposure_ids=tuple(exposure.exposure_id for exposure in exposures),
        )
        return cls(
            discovered=discovered,
            members=members,
            metadata=metadata,
            exposures=tuple(exposures),
        )


def load_observation(
    discovered: DiscoveredObservation,
    *,
    standard_catalog: StandardStarCatalog | None = None,
) -> Observation:
    """Build an :class:`Observation` from one discovery result."""

    return Observation.from_discovered(discovered, standard_catalog=standard_catalog)

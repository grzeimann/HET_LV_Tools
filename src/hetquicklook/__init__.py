"""Lightweight quick-look tools for HET instrumentation."""

from .config import QuicklookConfig
from .discovery import (
    ArchiveMember,
    DiscoveredObservation,
    RawFrameIdentity,
    discover_observations,
    inventory_members,
    parse_member_identity,
)
from .instrument import (
    Instrument,
    InstrumentTopology,
    lrs2_channel_for,
    lrs2_channel_for_identity,
    topology_for,
)
from .metadata import (
    ObservationMetadata,
    metadata_from_archive,
    metadata_from_header,
    metadata_from_members,
)
from .observation import Observation, load_observation
from .raw import RawFrameData, RawFrameLoader, load_frame

__all__ = [
    "DiscoveredObservation",
    "ArchiveMember",
    "Instrument",
    "InstrumentTopology",
    "ObservationMetadata",
    "Observation",
    "QuicklookConfig",
    "RawFrameData",
    "RawFrameIdentity",
    "RawFrameLoader",
    "discover_observations",
    "inventory_members",
    "load_frame",
    "load_observation",
    "lrs2_channel_for",
    "lrs2_channel_for_identity",
    "metadata_from_archive",
    "metadata_from_header",
    "metadata_from_members",
    "parse_member_identity",
    "topology_for",
]

__version__ = "0.1.0"

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
    LRS2InstrumentComponent,
    PhysicalAmplifierIdentity,
    InstrumentTopology,
    lrs2_component_for,
    lrs2_channel_for,
    lrs2_channel_for_identity,
    physical_identity_for,
    topology_for,
)
from .metadata import (
    ExposureMetadata,
    MetadataDisagreement,
    ObservationMetadata,
    exposure_metadata_from_headers,
    metadata_from_archive,
    metadata_from_header,
    metadata_from_members,
)
from .observation import Exposure, Observation, load_observation
from .raw import HeaderReadResult, RawFrameData, RawFrameLoader, load_frame
from .classification import (
    ExposureClassification,
    ObjectIntent,
    StandardStarCatalog,
    classify_exposure,
    parse_object_intent,
)
from .topology import (
    ConfigurationResourceError,
    LRS2FiberPositionLoader,
    TopologyReference,
    VirusTopologyLoader,
    virus_orientation,
)

__all__ = [
    "DiscoveredObservation",
    "ArchiveMember",
    "Instrument",
    "LRS2InstrumentComponent",
    "PhysicalAmplifierIdentity",
    "InstrumentTopology",
    "lrs2_component_for",
    "ObservationMetadata",
    "ExposureMetadata",
    "MetadataDisagreement",
    "Observation",
    "Exposure",
    "QuicklookConfig",
    "RawFrameData",
    "HeaderReadResult",
    "RawFrameIdentity",
    "RawFrameLoader",
    "discover_observations",
    "inventory_members",
    "load_frame",
    "load_observation",
    "lrs2_channel_for",
    "lrs2_channel_for_identity",
    "metadata_from_archive",
    "exposure_metadata_from_headers",
    "metadata_from_header",
    "metadata_from_members",
    "parse_member_identity",
    "topology_for",
    "physical_identity_for",
    "ExposureClassification",
    "ObjectIntent",
    "StandardStarCatalog",
    "classify_exposure",
    "parse_object_intent",
    "ConfigurationResourceError",
    "LRS2FiberPositionLoader",
    "TopologyReference",
    "VirusTopologyLoader",
    "virus_orientation",
]

__version__ = "0.1.0"

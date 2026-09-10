"""Lightweight quick-look tools for HET instrumentation."""

from .config import QuicklookConfig
from .discovery import DiscoveredObservation, discover_observations
from .instrument import Instrument, InstrumentTopology, topology_for
from .metadata import ObservationMetadata, metadata_from_archive, metadata_from_header

__all__ = [
    "DiscoveredObservation",
    "Instrument",
    "InstrumentTopology",
    "ObservationMetadata",
    "QuicklookConfig",
    "discover_observations",
    "metadata_from_archive",
    "metadata_from_header",
    "topology_for",
]

__version__ = "0.1.0"

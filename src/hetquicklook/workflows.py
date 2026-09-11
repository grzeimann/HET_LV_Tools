"""Operational quick-look workflows built from small array algorithms."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
from threading import Lock
from typing import TYPE_CHECKING, Mapping
import sys

try:
    import resource
except ImportError:  # pragma: no cover - resource is Unix-specific
    resource = None

import numpy as np

from .algorithms.centroid import Centroid, weighted_centroid
from .algorithms.collapse import (
    DEFAULT_COLLAPSE_COLUMNS,
    DEFAULT_COLLAPSE_STATISTIC,
    central_column_bounds,
    collapse_extracted_spectra,
    collapse_fibers,
    spatial_image_from_fiber_values,
)
from .algorithms.extraction import extract_fractional_aperture
from .algorithms.detector import reduce_amplifier_array
from .algorithms.results import AlgorithmResult
from .algorithms.spatial import gaussian_splat
from .algorithms.trace import fit_fiber_traces
from .fibers import FiberTopology
from .instrument import (
    Instrument,
    PhysicalAmplifierIdentity,
    lrs2_amplifier_tokens_for_channel,
    lrs2_channel_for,
    lrs2_component_for_channel,
)
from .topology import LRS2FiberPositionLoader, TopologyReference, VirusTopologyLoader

if TYPE_CHECKING:
    from .discovery import ArchiveMember
    from .observation import Exposure
    from .raw import RawFrameData, RawFrameLoader


VIRUS_SPATIAL_DEFAULTS = {
    "gaussian_fwhm_arcsec": 1.5,
    "pixel_scale_arcsec": 1.0,
    "grid_padding_arcsec": 1.1,
    "intended_fiducial": (0.0, 0.0),
}
LRS2_SPATIAL_DEFAULTS = {
    "gaussian_fwhm_arcsec": 1.2,
    "pixel_scale_arcsec": 0.4,
    "grid_padding_arcsec": 0.3,
    "intended_fiducial": (0.0, 0.0),
}
INSTRUMENT_SPATIAL_DEFAULTS = {
    Instrument.VIRUS: VIRUS_SPATIAL_DEFAULTS,
    Instrument.LRS2: LRS2_SPATIAL_DEFAULTS,
}
LRS2_STANDARD_FIDUCIAL = LRS2_SPATIAL_DEFAULTS["intended_fiducial"]
VIRUS_STANDARD_FIDUCIAL = VIRUS_SPATIAL_DEFAULTS["intended_fiducial"]


class QuicklookError(RuntimeError):
    """Raised when an archive-backed quick look cannot complete."""


@dataclass(frozen=True)
class QuicklookDiagnostics:
    """Optional timing and memory evidence for one quick-look operation."""

    stage_seconds: Mapping[str, float] = field(default_factory=dict)
    retained_array_bytes: int = 0
    peak_rss_bytes: int | None = None


@contextmanager
def _timed_stage(
    timings: dict[str, float], name: str, enabled: bool
):
    """Record one wall-clock stage when diagnostics are enabled."""

    if not enabled:
        yield
        return
    started = perf_counter()
    try:
        yield
    finally:
        timings[name] = perf_counter() - started


def _peak_rss_bytes() -> int | None:
    """Return process peak RSS in bytes when the platform exposes it."""

    if resource is None:
        return None
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux and the other common Unix implementations
    # report KiB.
    return value if sys.platform == "darwin" else value * 1024


def _algorithm_array_bytes(result: AlgorithmResult) -> int:
    """Return the memory held by an algorithm result's NumPy arrays."""

    return sum(
        int(np.asarray(value).nbytes)
        for value in result.arrays.values()
        if isinstance(value, np.ndarray)
    )


def _spatial_array_bytes(result: SpatialQuicklook) -> int:
    """Return the NumPy memory retained by a spatial quick-look product."""

    arrays = (
        result.image,
        result.spatial_weight,
        result.spatial_support,
        result.spatial_x_coordinates,
        result.spatial_y_coordinates,
        result.extracted_spectra,
        result.extraction_variance,
        result.extraction_valid_fraction,
        result.effective_aperture_width,
        result.extraction_valid,
    )
    return sum(int(np.asarray(value).nbytes) for value in arrays if isinstance(value, np.ndarray))


def _topology_array_bytes(topology: AmplifierTopologyResult) -> int:
    """Return NumPy memory held by one local detector topology."""

    trace_bytes = sum(
        int(np.asarray(trace.detector_x).nbytes + np.asarray(trace.detector_y).nbytes)
        for trace in topology.topology.traces.values()
    )
    return trace_bytes + _algorithm_array_bytes(topology.trace_result)


def _amplifier_array_bytes(
    loaded: "RawFrameData | None",
    detector: AlgorithmResult | None,
    topology: AmplifierTopologyResult,
    product: SpatialQuicklook,
) -> int:
    """Estimate arrays retained by one amplifier evidence record."""

    raw_data = getattr(loaded, "data", None)
    raw_bytes = 0 if not isinstance(raw_data, np.ndarray) else int(raw_data.nbytes)
    return (
        raw_bytes
        + (0 if detector is None else _algorithm_array_bytes(detector))
        + _topology_array_bytes(topology)
        + _spatial_array_bytes(product)
    )


def spatial_defaults_for(
    instrument: Instrument | str,
) -> Mapping[str, object]:
    """Return the workflow defaults for one supported instrument."""

    return INSTRUMENT_SPATIAL_DEFAULTS[Instrument.from_value(instrument)]


@dataclass(frozen=True)
class SpatialQuicklook:
    """Collapsed values, spatial image, coordinates, and QA evidence."""

    fiber_values: Mapping[str, float]
    image: np.ndarray
    instrument: Instrument | None = None
    fiber_positions: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    fiber_errors: Mapping[str, float] = field(default_factory=dict)
    spatial_weight: np.ndarray | None = None
    spatial_support: np.ndarray | None = None
    spatial_x_coordinates: np.ndarray | None = None
    spatial_y_coordinates: np.ndarray | None = None
    spatial_gaussian_fwhm_arcsec: float | None = None
    spatial_pixel_scale_arcsec: float | None = None
    intended_fiducial: tuple[float, float] | None = None
    collapse_columns: int | None = None
    collapse_statistic: str | None = None
    extraction_width: float | None = None
    extracted_spectra: np.ndarray | None = None
    extraction_variance: np.ndarray | None = None
    extraction_valid_fraction: np.ndarray | None = None
    effective_aperture_width: np.ndarray | None = None
    extraction_valid: np.ndarray | None = None


@dataclass(frozen=True)
class PointingQuicklook(SpatialQuicklook):
    """Standard-star image with intended and measured positions as evidence."""

    measured_centroid: Centroid = field(
        default_factory=lambda: Centroid(float("nan"), float("nan"), 0.0, 0)
    )
    requested_position: tuple[float, float] | None = None

    @property
    def offset(self) -> tuple[float, float] | None:
        """Return measured-minus-requested ``(x, y)`` when available."""

        if self.requested_position is None:
            return None
        return (
            self.measured_centroid.x - self.requested_position[0],
            self.measured_centroid.y - self.requested_position[1],
        )


@dataclass(frozen=True)
class LRS2ChannelQuicklook:
    """Spatial evidence for one complete two-amplifier LRS2 channel.

    The component products retain the detector, trace, extraction, and
    amplifier-specific evidence.  The channel-level arrays begin only after
    each amplifier has been reduced to collapsed fiber values and physical
    IFU positions.
    """

    channel: str
    amplifier_products: Mapping[str, SpatialQuicklook]
    fiber_positions: np.ndarray
    fiber_values: np.ndarray
    image: np.ndarray
    spatial_x_coordinates: np.ndarray
    spatial_y_coordinates: np.ndarray
    spatial_support: np.ndarray
    spatial_weight: np.ndarray
    intended_fiducial: tuple[float, float]
    measured_centroid: Centroid | None
    spatial_gaussian_fwhm_arcsec: float
    spatial_pixel_scale_arcsec: float
    fiber_errors: np.ndarray | None = None
    spatial_variance: np.ndarray | None = None

    @property
    def gaussian_fwhm_arcsec(self) -> float:
        """Return the configured Gaussian FWHM using the short result name."""

        return self.spatial_gaussian_fwhm_arcsec

    @property
    def pixel_scale_arcsec(self) -> float:
        """Return the configured output pixel scale using the short name."""

        return self.spatial_pixel_scale_arcsec

    @property
    def offset(self) -> tuple[float, float] | None:
        """Return measured-minus-intended position when a centroid exists."""

        if self.measured_centroid is None:
            return None
        return (
            self.measured_centroid.x - self.intended_fiducial[0],
            self.measured_centroid.y - self.intended_fiducial[1],
        )


@dataclass(frozen=True)
class AmplifierTopologyResult:
    """Trace and physical-fiber topology prepared for one amplifier."""

    topology: FiberTopology
    trace_result: AlgorithmResult
    physical_identity: PhysicalAmplifierIdentity
    trace_provenance: TopologyReference
    position_provenance: TopologyReference
    trace_source: "ArchiveMember | None" = None
    detector_column_start: int | None = None
    detector_column_stop: int | None = None


@dataclass(frozen=True)
class LRS2AmplifierQuicklookEvidence:
    """All evidence retained while building one LRS2 amplifier product."""

    frame: "ArchiveMember"
    loaded: "RawFrameData | None"
    detector: AlgorithmResult | None
    topology: AmplifierTopologyResult
    product: SpatialQuicklook
    diagnostics: QuicklookDiagnostics = field(default_factory=QuicklookDiagnostics)


@dataclass(frozen=True)
class AmplifierQuicklookEvidence:
    """Evidence for one archive-backed amplifier quick look.

    This generic envelope is used by the high-level VIRUS orchestration.  The
    existing LRS2-specific envelope remains available because it is part of
    the established result API.
    """

    frame: "ArchiveMember"
    loaded: "RawFrameData | None"
    detector: AlgorithmResult | None
    topology: AmplifierTopologyResult
    product: SpatialQuicklook
    diagnostics: QuicklookDiagnostics = field(default_factory=QuicklookDiagnostics)


@dataclass(frozen=True)
class LRS2QuicklookSet:
    """Amplifier evidence and channel products for one LRS2 exposure."""

    amplifier_evidence: Mapping[str, LRS2AmplifierQuicklookEvidence]
    channels: Mapping[str, LRS2ChannelQuicklook]

    @property
    def amplifier_products(self) -> dict[str, SpatialQuicklook]:
        """Return the independently generated amplifier products by token."""

        return {
            token: evidence.product
            for token, evidence in self.amplifier_evidence.items()
        }


@dataclass(frozen=True)
class VIRUSIFUQuicklookSet:
    """Amplifier evidence grouped by one VIRUS IFU slot."""

    ifu_slot: str
    amplifier_evidence: Mapping[str, AmplifierQuicklookEvidence]
    product: SpatialQuicklook | None = None
    diagnostics: QuicklookDiagnostics = field(default_factory=QuicklookDiagnostics)
    unavailable_amplifiers: Mapping[str, str] = field(default_factory=dict)

    @property
    def amplifier_products(self) -> dict[str, SpatialQuicklook]:
        """Return the available amplifier products for this IFU."""

        return {
            token: evidence.product
            for token, evidence in self.amplifier_evidence.items()
        }


@dataclass(frozen=True)
class VIRUSQuicklookSet:
    """Per-IFU VIRUS quick-look evidence for one exposure.

    VIRUS does not yet have a package-level whole-instrument spatial
    composition.  This result therefore keeps the scientifically meaningful
    IFU grouping without selecting an arbitrary amplifier or inventing a
    focal-plane product.
    """

    ifus: Mapping[str, VIRUSIFUQuicklookSet]
    diagnostics: Mapping[str, QuicklookDiagnostics] = field(default_factory=dict)

    @property
    def amplifier_evidence(self) -> dict[str, AmplifierQuicklookEvidence]:
        """Return available amplifier evidence keyed by full token."""

        return {
            token: evidence
            for ifu in self.ifus.values()
            for token, evidence in ifu.amplifier_evidence.items()
        }

    @property
    def amplifier_products(self) -> dict[str, SpatialQuicklook]:
        """Return available amplifier products keyed by full token."""

        return {
            token: evidence.product
            for token, evidence in self.amplifier_evidence.items()
        }


def _identity_key(identity: PhysicalAmplifierIdentity) -> tuple[object, ...]:
    """Return the minimum physical identity used by trace resources."""

    def normalized(value: str | None) -> str | None:
        return None if value is None else str(value).strip().zfill(3)

    return (
        Instrument.from_value(identity.instrument),
        str(identity.ifu_slot).strip().zfill(3),
        str(identity.amplifier).strip().upper(),
        normalized(identity.ifuid),
        normalized(identity.specid),
    )


def _same_trace_identity(
    target: PhysicalAmplifierIdentity,
    candidate: PhysicalAmplifierIdentity,
) -> bool:
    """Compare the required trace-resource identity fields."""

    target_key = _identity_key(target)
    candidate_key = _identity_key(candidate)
    if target_key[:3] != candidate_key[:3]:
        return False
    return all(
        left is None or right is None or left == right
        for left, right in zip(target_key[3:], candidate_key[3:])
    )


def _trace_hardware_exception(identity: PhysicalAmplifierIdentity) -> bool:
    """Return whether the established one-row trace exception applies."""

    return (
        str(identity.specid).strip().zfill(3) == "504"
        and str(identity.ifuid).strip().zfill(3) == "018"
        and str(identity.amplifier).strip().upper() == "RU"
    )


def _as_datetime(value: object, fallback: date) -> datetime:
    """Normalize archive time metadata for target-relative ordering."""

    if isinstance(value, datetime):
        result = value
    elif isinstance(value, date):
        result = datetime.combine(value, datetime.min.time())
    elif value is None:
        result = datetime.combine(fallback, datetime.min.time())
    else:
        text = str(value).strip()
        try:
            result = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            try:
                result = datetime.strptime(text[:8], "%Y%m%d")
            except ValueError:
                result = datetime.combine(fallback, datetime.min.time())
    if result.tzinfo is not None:
        result = result.astimezone(timezone.utc).replace(tzinfo=None)
    return result


def _required_amplifier_tokens(identity: PhysicalAmplifierIdentity) -> tuple[str, ...]:
    """Return the complete amplifier set for one selected component."""

    slot = str(identity.ifu_slot).strip().zfill(3)
    return tuple(f"{slot}{amplifier}" for amplifier in ("LL", "LU", "RL", "RU"))


def _validate_trace_geometry(
    trace_result: AlgorithmResult,
    *,
    detector_rows: int,
    detector_columns: int,
    expected_fibers: int,
    aperture_width: float,
) -> None:
    """Validate only geometry required for fractional-aperture extraction."""

    trace_map = np.asarray(trace_result.get_array("fiber_trace_map"), dtype=float)
    if trace_map.shape != (expected_fibers, detector_columns):
        raise ValueError(
            "trace map shape must be "
            f"({expected_fibers}, {detector_columns}), found {trace_map.shape}"
        )
    reference = np.asarray(trace_result.get_array("trace_reference"))
    if reference.ndim != 2 or reference.shape[0] != expected_fibers:
        raise ValueError("trace reference and trace map fiber counts differ")
    width = float(aperture_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("extraction aperture width must be positive and finite")
    if detector_rows <= 0:
        raise ValueError("prepared detector must contain rows")
    if not np.all(np.isfinite(trace_map)):
        raise ValueError("trace map contains non-finite detector centers")

    half_width = width / 2.0
    if np.any(trace_map < half_width) or np.any(
        trace_map > detector_rows - half_width
    ):
        raise ValueError(
            "trace centers do not leave the requested aperture wholly inside "
            "the prepared detector"
        )


class _QuickTraceProvider:
    """Resolve and cache flat-derived local traces for one quick-look night."""

    def __init__(
        self,
        candidates: tuple[tuple["Exposure", date], ...],
        *,
        trace_root: str | Path,
    ) -> None:
        self._candidates = candidates
        self._trace_root = Path(trace_root)
        self._cache: dict[
            tuple[object, ...],
            tuple[AlgorithmResult, TopologyReference, "ArchiveMember"],
        ] = {}
        self._cache_lock = Lock()

    def _select_candidate(
        self,
        target_exposure: "Exposure",
        target_frame: "ArchiveMember",
        target_identity: PhysicalAmplifierIdentity,
        *,
        at: date | datetime | str | None,
    ) -> tuple["Exposure", "ArchiveMember"]:
        identity = target_frame.identity
        if identity is None:
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                "stage: target identity, reason: amplifier identity is unavailable"
            )
        target_token = str(identity.amplifier_token).strip().upper()
        target_time = _as_datetime(at, date.today())
        if target_exposure.metadata.observation_time is not None:
            target_time = _as_datetime(
                target_exposure.metadata.observation_time, target_time.date()
            )
        slot = str(target_identity.ifu_slot).strip().zfill(3)
        matches: list[
            tuple[float, datetime, str, str, "Exposure", "ArchiveMember"]
        ] = []
        for candidate, fallback_date in self._candidates:
            classification = candidate.classification
            if classification.quicklook_kind != "flat":
                continue
            if classification.applicable_ifu_slots and slot not in {
                str(value).strip().zfill(3)
                for value in classification.applicable_ifu_slots
            }:
                continue
            for member in candidate.frames:
                member_identity = member.identity
                if member_identity is None:
                    continue
                if str(member_identity.amplifier_token).strip().upper() != target_token:
                    continue
                try:
                    candidate_identity = candidate.identity_for(member)
                except ValueError:
                    continue
                if not _same_trace_identity(target_identity, candidate_identity):
                    continue
                candidate_time = _as_datetime(
                    candidate.metadata.observation_time, fallback_date
                )
                delta = abs((candidate_time - target_time).total_seconds())
                matches.append(
                    (
                        delta,
                        candidate_time,
                        str(candidate.exposure_id),
                        str(member.member_name),
                        candidate,
                        member,
                    )
                )
                break
        if not matches:
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                f"amplifier {target_token}, stage: flat resolution, reason: "
                "no suitable classified flat exists"
            )
        matches.sort(key=lambda item: item[:4])
        return matches[0][4], matches[0][5]

    def _require_flat_component(
        self,
        exposure: "Exposure",
        target_identity: PhysicalAmplifierIdentity,
    ) -> "ArchiveMember":
        expected = _required_amplifier_tokens(target_identity)
        by_token: dict[str, "ArchiveMember"] = {}
        for member in exposure.frames:
            identity = member.identity
            if identity is None:
                continue
            token = str(identity.amplifier_token).strip().upper()
            if token not in expected:
                continue
            if token in by_token:
                raise QuicklookError(
                    f"Quick-look failed for exposure {exposure.exposure_id}, "
                    f"amplifier {token}, stage: flat completeness, reason: "
                    "duplicate required amplifier frame"
                )
            by_token[token] = member
        missing = [token for token in expected if token not in by_token]
        if missing:
            raise QuicklookError(
                f"Quick-look failed for exposure {exposure.exposure_id}, "
                f"amplifier {missing[0]}, stage: flat completeness, reason: "
                f"required amplifier frame is missing; expected {', '.join(expected)}"
            )
        target_token = (
            f"{str(target_identity.ifu_slot).strip().zfill(3)}"
            f"{str(target_identity.amplifier).strip().upper()}"
        )
        return by_token[target_token]

    def resolve(
        self,
        target_exposure: "Exposure",
        target_frame: "ArchiveMember",
        *,
        loader: "RawFrameLoader | None",
        at: date | datetime | str | None,
        column_width: int,
        extraction_width: float,
        timings: dict[str, float] | None = None,
    ) -> tuple[AlgorithmResult, TopologyReference, "ArchiveMember"]:
        """Return a validated quick trace and its calibration provenance."""

        try:
            target_identity = target_exposure.identity_for(target_frame)
        except Exception as error:
            token = (
                "unknown"
                if target_frame.identity is None
                else str(target_frame.identity.amplifier_token).strip().upper()
            )
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                f"amplifier {token}, stage: target identity, reason: {error}"
            ) from error
        with _timed_stage(
            timings if timings is not None else {},
            "flat_candidate_selection",
            timings is not None,
        ):
            candidate, _ = self._select_candidate(
                target_exposure,
                target_frame,
                target_identity,
                at=at,
            )
            selected_frame = self._require_flat_component(candidate, target_identity)
        cache_key = (
            str(selected_frame.archive_path),
            selected_frame.outer_tar_member,
            selected_frame.member_name,
            _identity_key(target_identity),
            int(column_width),
            5,
            1,
        )
        with self._cache_lock:
            cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            with _timed_stage(
                timings if timings is not None else {},
                "flat_archive_load",
                timings is not None,
            ):
                loaded = candidate.load_frame(selected_frame, loader=loader)
        except Exception as error:
            token = f"{target_identity.ifu_slot}{target_identity.amplifier}"
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                f"amplifier {token}, stage: flat archive load, reason: {error}"
            ) from error
        try:
            with _timed_stage(
                timings if timings is not None else {},
                "flat_detector_preparation",
                timings is not None,
            ):
                detector = reduce_amplifier_array(loaded.data, dict(loaded.header))
                prepared = detector.get_array("oriented_detector_image")
        except Exception as error:
            token = f"{target_identity.ifu_slot}{target_identity.amplifier}"
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                f"amplifier {token}, stage: flat detector preparation, reason: {error}"
            ) from error

        try:
            with _timed_stage(
                timings if timings is not None else {},
                "trace_reference_resolution",
                timings is not None,
            ):
                flat_start, flat_stop = central_column_bounds(
                    prepared.shape[1], column_width
                )
                trace_reference, trace_provenance = VirusTopologyLoader(
                    trace_root=self._trace_root
                ).resolve_trace_reference(target_identity, at=at)
        except Exception as error:
            token = f"{target_identity.ifu_slot}{target_identity.amplifier}"
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                f"amplifier {token}, stage: trace reference resolution, reason: {error}"
            ) from error

        try:
            with _timed_stage(
                timings if timings is not None else {},
                "fit_fiber_traces",
                timings is not None,
            ):
                trace_result = fit_fiber_traces(
                    prepared[:, flat_start:flat_stop],
                    trace_reference,
                    specid=target_identity.specid,
                    ifuid=target_identity.ifuid,
                    amplifier=target_identity.amplifier,
                    n_chunks=5,
                    degree=1,
                    fit_method="fast",
                    detector_column_start=flat_start,
                    timings=timings,
                )
        except Exception as error:
            token = f"{target_identity.ifu_slot}{target_identity.amplifier}"
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                f"amplifier {token}, stage: fit_fiber_traces, reason: {error}"
            ) from error

        try:
            with _timed_stage(
                timings if timings is not None else {},
                "trace_validation",
                timings is not None,
            ):
                expected_fibers = trace_reference.shape[0] - (
                    1 if _trace_hardware_exception(target_identity) else 0
                )
                _validate_trace_geometry(
                    trace_result,
                    detector_rows=prepared.shape[0],
                    detector_columns=flat_stop - flat_start,
                    expected_fibers=expected_fibers,
                    aperture_width=extraction_width,
                )
        except Exception as error:
            token = f"{target_identity.ifu_slot}{target_identity.amplifier}"
            raise QuicklookError(
                f"Quick-look failed for exposure {target_exposure.exposure_id}, "
                f"amplifier {token}, stage: trace validation, reason: {error}"
            ) from error

        resolved = (trace_result, trace_provenance, selected_frame)
        with self._cache_lock:
            self._cache[cache_key] = resolved
        return resolved


def _dense_trace_map(topology: FiberTopology, detector_columns: int) -> np.ndarray:
    """Convert topology trace samples to a dense ``(fiber, column)`` map."""

    if detector_columns <= 0:
        raise ValueError("detector must contain at least one detector column")
    x_target = np.arange(detector_columns, dtype=float)
    rows: list[np.ndarray] = []
    for fiber_id in topology.fiber_ids:
        trace = topology.traces[fiber_id]
        x_values = np.asarray(trace.detector_x, dtype=float)
        y_values = np.asarray(trace.detector_y, dtype=float)
        valid = np.isfinite(x_values) & np.isfinite(y_values)
        if not np.any(valid):
            rows.append(np.full(detector_columns, np.nan))
            continue
        if np.count_nonzero(valid) == 1:
            rows.append(np.full(detector_columns, y_values[valid][0]))
            continue
        x_values = x_values[valid]
        y_values = y_values[valid]
        order = np.argsort(x_values)
        x_values = x_values[order]
        y_values = y_values[order]
        unique_x, inverse = np.unique(x_values, return_inverse=True)
        if unique_x.size != x_values.size:
            sums = np.zeros(unique_x.size, dtype=float)
            counts = np.zeros(unique_x.size, dtype=float)
            np.add.at(sums, inverse, y_values)
            np.add.at(counts, inverse, 1.0)
            y_values = sums / counts
            x_values = unique_x
        rows.append(np.interp(x_target, x_values, y_values))
    return np.asarray(rows, dtype=float)


def _has_dense_trace_geometry(topology: FiberTopology, detector_columns: int) -> bool:
    """Whether topology samples already describe every detector column."""

    return all(
        trace.detector_x.size == detector_columns
        and np.all(np.isfinite(trace.detector_y))
        for trace in topology.traces.values()
    )


def _resolve_spatial_parameters(
    instrument: Instrument | str,
    *,
    gaussian_fwhm: float | None,
    pixel_scale: float | None,
    spatial_defaults: Mapping[str, object] | None,
) -> tuple[Instrument, float, float, tuple[float, float]]:
    """Resolve workflow defaults and caller overrides before splatting."""

    parsed_instrument = Instrument.from_value(instrument)
    defaults = (
        spatial_defaults
        if spatial_defaults is not None
        else spatial_defaults_for(parsed_instrument)
    )
    resolved_fwhm = defaults["gaussian_fwhm_arcsec"] if gaussian_fwhm is None else gaussian_fwhm
    resolved_pixel_scale = defaults["pixel_scale_arcsec"] if pixel_scale is None else pixel_scale
    fiducial = defaults.get("intended_fiducial", (0.0, 0.0))
    if len(fiducial) != 2:
        raise ValueError("intended_fiducial must contain two coordinates")
    return (
        parsed_instrument,
        float(resolved_fwhm),
        float(resolved_pixel_scale),
        (float(fiducial[0]), float(fiducial[1])),
    )


def _spatial_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    aperture: int | float | None = None,
    gaussian_fwhm: float | None = None,
    pixel_scale: float | None = None,
    grid_padding_arcsec: float | None = None,
    instrument: Instrument | str = Instrument.VIRUS,
    spatial_defaults: Mapping[str, object] | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
    intended_fiducial: tuple[float, float] | None = None,
    retain_extraction: bool = True,
    build_spatial: bool = True,
) -> SpatialQuicklook:
    """Run the in-memory detector-to-fiber-to-spatial quick-look path.

    ``retain_extraction`` and ``build_spatial`` let archive workflows keep
    only the collapsed fiber boundary until the final instrument-level
    composition.  Their defaults preserve the richer direct in-memory API.
    """

    image = np.asarray(detector, dtype=float)
    if image.ndim != 2:
        raise ValueError("detector must be a two-dimensional array")
    parsed_instrument, resolved_fwhm, resolved_pixel_scale, _ = (
        _resolve_spatial_parameters(
            instrument,
            gaussian_fwhm=gaussian_fwhm,
            pixel_scale=pixel_scale,
            spatial_defaults=spatial_defaults,
        )
    )
    defaults = (
        spatial_defaults
        if spatial_defaults is not None
        else spatial_defaults_for(parsed_instrument)
    )
    resolved_grid_padding = defaults.get("grid_padding_arcsec")
    if grid_padding_arcsec is not None:
        resolved_grid_padding = grid_padding_arcsec
    dense = trace_map is not None or _has_dense_trace_geometry(topology, image.shape[1])
    if dense:
        traces = _dense_trace_map(topology, image.shape[1]) if trace_map is None else np.asarray(trace_map, dtype=float)
        if traces.shape != (len(topology.fiber_ids), image.shape[1]):
            raise ValueError("trace_map must have shape (nfiber, detector_columns)")
        variance = (
            np.ones_like(image, dtype=float)
            if detector_variance is None
            else np.asarray(detector_variance, dtype=float)
        )
        extraction = extract_fractional_aperture(
            image,
            variance,
            traces,
            width=extraction_width if aperture is None else float(aperture),
        )
        collapsed = collapse_extracted_spectra(
            extraction.get_array("spectrum"),
            extraction.get_array("variance"),
            collapse_columns=collapse_columns,
            statistic=statistic,
        )
        values = collapsed.get_array("fiber_values")
        errors_array = collapsed.arrays.get("fiber_errors")
        fiber_errors = (
            {}
            if errors_array is None
            else {
                fiber_id: float(errors_array[index])
                for index, fiber_id in enumerate(topology.fiber_ids)
            }
        )
        positions = np.asarray(
            [
                [topology.locations[fiber_id].ifu_x, topology.locations[fiber_id].ifu_y]
                for fiber_id in topology.fiber_ids
            ],
            dtype=float,
        )
        fiber_positions = {
            fiber_id: (float(positions[index, 0]), float(positions[index, 1]))
            for index, fiber_id in enumerate(topology.fiber_ids)
        }
        spatial = (
            gaussian_splat(
                positions,
                values,
                None if errors_array is None else errors_array,
                fwhm=resolved_fwhm,
                pixel_scale=resolved_pixel_scale,
                grid_padding=resolved_grid_padding,
                output_shape=output_shape,
                origin=origin,
            )
            if build_spatial
            else None
        )
        return SpatialQuicklook(
            fiber_values={
                fiber_id: float(values[index])
                for index, fiber_id in enumerate(topology.fiber_ids)
            },
            image=(
                np.empty((0, 0), dtype=float)
                if spatial is None
                else spatial.image
            ),
            instrument=parsed_instrument,
            fiber_positions=fiber_positions,
            fiber_errors=fiber_errors,
            spatial_weight=None if spatial is None else spatial.weight,
            spatial_support=None if spatial is None else spatial.support,
            spatial_x_coordinates=None if spatial is None else spatial.x_coordinates,
            spatial_y_coordinates=None if spatial is None else spatial.y_coordinates,
            spatial_gaussian_fwhm_arcsec=resolved_fwhm,
            spatial_pixel_scale_arcsec=resolved_pixel_scale,
            intended_fiducial=intended_fiducial,
            collapse_columns=int(collapsed.scalars["collapse_columns_requested"]),
            collapse_statistic=str(collapsed.scalars["collapse_statistic"]),
            extraction_width=float(extraction.scalars["aperture_width_pixels"]),
            extracted_spectra=(
                extraction.get_array("spectrum") if retain_extraction else None
            ),
            extraction_variance=(
                extraction.get_array("variance") if retain_extraction else None
            ),
            extraction_valid_fraction=(
                extraction.get_array("valid_pixel_fraction")
                if retain_extraction
                else None
            ),
            effective_aperture_width=(
                extraction.get_array("effective_aperture_width")
                if retain_extraction
                else None
            ),
            extraction_valid=(
                extraction.get_array("extraction_valid")
                if retain_extraction
                else None
            ),
        )

    # Compatibility adapter for the original sparse FiberTopology examples:
    # their trace samples are individual detector points rather than a dense
    # detector-column trace map.  New pipeline data take the shared path above.
    legacy_aperture = 1 if aperture is None else aperture
    fiber_values = collapse_fibers(
        image, topology, aperture=int(legacy_aperture), statistic=statistic
    )
    return SpatialQuicklook(
        fiber_values=fiber_values,
        image=spatial_image_from_fiber_values(fiber_values, topology),
        instrument=parsed_instrument,
        fiber_positions={
            fiber_id: (
                float(topology.locations[fiber_id].ifu_x),
                float(topology.locations[fiber_id].ifu_y),
            )
            for fiber_id in topology.fiber_ids
        },
        spatial_gaussian_fwhm_arcsec=resolved_fwhm,
        spatial_pixel_scale_arcsec=resolved_pixel_scale,
        intended_fiducial=intended_fiducial,
        collapse_columns=None,
        collapse_statistic=statistic,
        extraction_width=(None if aperture is None else float(aperture)),
    )


def run_ldls_flat_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    aperture: int | float | None = None,
    gaussian_fwhm: float | None = None,
    pixel_scale: float | None = None,
    grid_padding_arcsec: float | None = None,
    instrument: Instrument | str = Instrument.VIRUS,
    spatial_defaults: Mapping[str, object] | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
    retain_extraction: bool = True,
    build_spatial: bool = True,
) -> SpatialQuicklook:
    """Create a collapsed spatial product for an LDLS-flat detector image."""

    return _spatial_quicklook(
        detector,
        topology,
        detector_variance=detector_variance,
        trace_map=trace_map,
        extraction_width=extraction_width,
        collapse_columns=collapse_columns,
        statistic=statistic,
        aperture=aperture,
        gaussian_fwhm=gaussian_fwhm,
        pixel_scale=pixel_scale,
        grid_padding_arcsec=grid_padding_arcsec,
        instrument=instrument,
        spatial_defaults=spatial_defaults,
        output_shape=output_shape,
        origin=origin,
        retain_extraction=retain_extraction,
        build_spatial=build_spatial,
    )


def run_standard_star_quicklook(
    detector: np.ndarray,
    topology: FiberTopology,
    *,
    requested_position: tuple[float, float] | None = None,
    detector_variance: np.ndarray | None = None,
    trace_map: np.ndarray | None = None,
    extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    aperture: int | float | None = None,
    gaussian_fwhm: float | None = None,
    pixel_scale: float | None = None,
    grid_padding_arcsec: float | None = None,
    instrument: Instrument | str = Instrument.VIRUS,
    spatial_defaults: Mapping[str, object] | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
    retain_extraction: bool = True,
    build_spatial: bool = True,
) -> PointingQuicklook:
    """Create a spatial product with measured and intended positions.

    The existing weighted centroid is exposed as a measured location. It is
    evidence for an observer and does not produce an automatic quality
    decision.
    """

    _, _, _, default_fiducial = _resolve_spatial_parameters(
        instrument,
        gaussian_fwhm=gaussian_fwhm,
        pixel_scale=pixel_scale,
        spatial_defaults=spatial_defaults,
    )
    intended_position = (
        default_fiducial if requested_position is None else requested_position
    )

    spatial = _spatial_quicklook(
        detector,
        topology,
        detector_variance=detector_variance,
        trace_map=trace_map,
        extraction_width=extraction_width,
        collapse_columns=collapse_columns,
        statistic=statistic,
        aperture=aperture,
        gaussian_fwhm=gaussian_fwhm,
        pixel_scale=pixel_scale,
        grid_padding_arcsec=grid_padding_arcsec,
        instrument=instrument,
        spatial_defaults=spatial_defaults,
        output_shape=output_shape,
        origin=origin,
        intended_fiducial=intended_position,
        retain_extraction=retain_extraction,
        build_spatial=build_spatial,
    )
    x = np.array([topology.locations[fiber_id].ifu_x for fiber_id in topology.fiber_ids])
    y = np.array([topology.locations[fiber_id].ifu_y for fiber_id in topology.fiber_ids])
    values = np.array([spatial.fiber_values[fiber_id] for fiber_id in topology.fiber_ids])
    measured = weighted_centroid(x, y, values)
    return PointingQuicklook(
        fiber_values=spatial.fiber_values,
        image=spatial.image,
        instrument=spatial.instrument,
        fiber_positions=spatial.fiber_positions,
        measured_centroid=measured,
        requested_position=intended_position,
        fiber_errors=spatial.fiber_errors,
        spatial_weight=spatial.spatial_weight,
        spatial_support=spatial.spatial_support,
        spatial_x_coordinates=spatial.spatial_x_coordinates,
        spatial_y_coordinates=spatial.spatial_y_coordinates,
        spatial_gaussian_fwhm_arcsec=spatial.spatial_gaussian_fwhm_arcsec,
        spatial_pixel_scale_arcsec=spatial.spatial_pixel_scale_arcsec,
        intended_fiducial=spatial.intended_fiducial,
        collapse_columns=spatial.collapse_columns,
        collapse_statistic=spatial.collapse_statistic,
        extraction_width=spatial.extraction_width,
        extracted_spectra=spatial.extracted_spectra,
        extraction_variance=spatial.extraction_variance,
        extraction_valid_fraction=spatial.extraction_valid_fraction,
        effective_aperture_width=spatial.effective_aperture_width,
        extraction_valid=spatial.extraction_valid,
    )


def build_amplifier_topology(
    prepared_detector: np.ndarray,
    physical_identity: PhysicalAmplifierIdentity,
    *,
    trace_root: str | Path,
    at: date | datetime | str | None = None,
    resource_root: str | Path | None = None,
    trace_result: AlgorithmResult | None = None,
    trace_provenance: TopologyReference | None = None,
    trace_source: "ArchiveMember | None" = None,
    detector_column_start: int | None = None,
    detector_column_stop: int | None = None,
) -> AmplifierTopologyResult:
    """Build trace and physical-fiber topology for one prepared amplifier.

    This is the workflow boundary between detector preparation and the
    in-memory quick-look algorithms.  It accepts a prepared detector array
    and an already resolved physical identity; archive discovery and raw-file
    loading remain outside this function.
    """

    image = np.asarray(prepared_detector, dtype=float)
    if image.ndim != 2:
        raise ValueError("prepared_detector must be a two-dimensional array")
    identity = PhysicalAmplifierIdentity(
        instrument=Instrument.from_value(physical_identity.instrument),
        ifu_slot=str(physical_identity.ifu_slot).strip().zfill(3),
        amplifier=str(physical_identity.amplifier).strip().upper(),
        ifuid=(
            None
            if physical_identity.ifuid is None
            else str(physical_identity.ifuid).strip()
        ),
        specid=(
            None
            if physical_identity.specid is None
            else str(physical_identity.specid).strip()
        ),
        controller=physical_identity.controller,
    )
    trace_loader = VirusTopologyLoader(trace_root=trace_root)
    if trace_result is None:
        trace_reference, trace_provenance = trace_loader.resolve_trace_reference(
            identity, at=at
        )
        trace_result = fit_fiber_traces(
            image,
            trace_reference,
            specid=identity.specid,
            ifuid=identity.ifuid,
            amplifier=identity.amplifier,
        )
    elif trace_provenance is None:
        _, trace_provenance = trace_loader.resolve_trace_reference(identity, at=at)
    trace_map = trace_result.get_array("fiber_trace_map")

    if identity.instrument is Instrument.LRS2:
        channel = lrs2_channel_for(identity.ifu_slot, identity.amplifier)
        if channel is None:
            raise ValueError(
                f"No authoritative LRS2 channel for {identity.ifu_slot}"
                f"{identity.amplifier}"
            )
        positions, position_provenance = LRS2FiberPositionLoader(
            resource_root=resource_root
        ).fiber_positions(channel, identity.amplifier)
    else:
        if identity.ifuid is None:
            raise ValueError("VIRUS topology requires IFUID")
        positions, position_provenance = trace_loader.fiber_positions(
            identity.ifuid, identity.amplifier
        )

    # This is the established hardware exception in the supplied trace
    # algorithm.  The trace result has one fewer row while the position table
    # still contains the reference row, which is removed here to preserve the
    # detector-to-position correspondence.
    if positions.shape[0] != trace_map.shape[0]:
        special_case = _trace_hardware_exception(identity)
        if not special_case or positions.shape[0] != trace_map.shape[0] + 1:
            raise ValueError("Trace and position fiber counts do not match")
        positions = positions[:-1]

    nfiber, ncolumn = trace_map.shape
    fiber_ids = tuple(
        f"{identity.ifu_slot}{identity.amplifier}-{index:03d}"
        for index in range(nfiber)
    )
    detector_x = np.tile(np.arange(ncolumn, dtype=float), (nfiber, 1))
    topology = FiberTopology.from_arrays(
        identity.amplifier,
        fiber_ids,
        detector_x,
        trace_map,
        positions[:, 0],
        positions[:, 1],
    )
    return AmplifierTopologyResult(
        topology=topology,
        trace_result=trace_result,
        physical_identity=identity,
        trace_provenance=trace_provenance,
        position_provenance=position_provenance,
        trace_source=trace_source,
        detector_column_start=detector_column_start,
        detector_column_stop=detector_column_stop,
    )


def _run_archive_amplifier_quicklook(
    exposure: "Exposure",
    frame: "ArchiveMember",
    *,
    trace_root: str | Path,
    at: date | datetime | str | None,
    quicklook_kind: str,
    loader: "RawFrameLoader | None",
    resource_root: str | Path | None,
    requested_position: tuple[float, float] | None,
    detector_extraction_width: float,
    collapse_columns: int,
    collapse_statistic: str,
    gaussian_fwhm_arcsec: float | None,
    pixel_scale_arcsec: float | None,
    grid_padding_arcsec: float | None,
    output_shape: tuple[int, int] | None,
    origin: tuple[float, float] | None,
    trace_provider: _QuickTraceProvider | None,
    timing: bool = False,
    memory_check: bool = False,
    detailed_evidence: bool = False,
) -> tuple[
    "RawFrameData | None",
    AlgorithmResult | None,
    AmplifierTopologyResult,
    SpatialQuicklook,
    QuicklookDiagnostics,
]:
    """Run the shared archive-backed amplifier path once."""

    timings: dict[str, float] = {}
    stage_prefix = "target" if quicklook_kind in {"standard", "target"} else "flat"
    total_started = perf_counter() if timing else 0.0
    try:
        identity = exposure.identity_for(frame)
    except Exception as error:
        token = (
            "unknown"
            if frame.identity is None
            else str(frame.identity.amplifier_token).strip().upper()
        )
        raise QuicklookError(
            f"Quick-look failed for exposure {exposure.exposure_id}, "
            f"amplifier {token}, stage: target identity, reason: {error}"
        ) from error
    token = f"{identity.ifu_slot}{identity.amplifier}"
    try:
        with _timed_stage(timings, f"{stage_prefix}_archive_load", timing):
            loaded = exposure.load_frame(frame, loader=loader)
        with _timed_stage(timings, f"{stage_prefix}_detector_preparation", timing):
            detector = reduce_amplifier_array(loaded.data, dict(loaded.header))
    except Exception as error:
        raise QuicklookError(
            f"Quick-look failed for exposure {exposure.exposure_id}, "
            f"amplifier {token}, stage: target detector preparation, reason: {error}"
        ) from error

    prepared_full = detector.get_array("oriented_detector_image")
    variance_full = detector.get_array("detector_variance")
    detector_column_start, detector_column_stop = central_column_bounds(
        prepared_full.shape[1], collapse_columns
    )
    prepared_detector = prepared_full[:, detector_column_start:detector_column_stop]
    detector_variance = variance_full[:, detector_column_start:detector_column_stop]
    trace_result: AlgorithmResult | None = None
    trace_provenance: TopologyReference | None = None
    trace_source: "ArchiveMember | None" = None
    try:
        if quicklook_kind == "flat":
            with _timed_stage(timings, "trace_reference_resolution", timing):
                trace_reference, trace_provenance = VirusTopologyLoader(
                    trace_root=trace_root
                ).resolve_trace_reference(identity, at=at)
            with _timed_stage(timings, "fit_fiber_traces", timing):
                trace_result = fit_fiber_traces(
                    prepared_detector,
                    trace_reference,
                    specid=identity.specid,
                    ifuid=identity.ifuid,
                    amplifier=identity.amplifier,
                    n_chunks=5,
                    degree=1,
                    fit_method="fast",
                    detector_column_start=detector_column_start,
                    timings=timings if timing else None,
                )
                trace_source = frame
            with _timed_stage(timings, "trace_validation", timing):
                trace_reference_array = trace_result.get_array("trace_reference")
                _validate_trace_geometry(
                    trace_result,
                    detector_rows=prepared_detector.shape[0],
                    detector_columns=prepared_detector.shape[1],
                    expected_fibers=trace_reference_array.shape[0],
                    aperture_width=detector_extraction_width,
                )
        elif quicklook_kind in {"standard", "target"}:
            if trace_provider is None:
                raise QuicklookError(
                    "a flat-derived trace provider is required for target extraction"
                )
            provider_timings = {} if timing else None
            trace_result, trace_provenance, trace_source = trace_provider.resolve(
                exposure,
                frame,
                loader=loader,
                at=at,
                column_width=collapse_columns,
                extraction_width=detector_extraction_width,
                timings=provider_timings,
            )
            if provider_timings is not None:
                timings.update(provider_timings)
        else:
            raise ValueError(f"Unsupported quicklook_kind: {quicklook_kind!r}")

        trace_map = trace_result.get_array("fiber_trace_map")
        trace_reference = trace_result.get_array("trace_reference")
        _validate_trace_geometry(
            trace_result,
            detector_rows=prepared_detector.shape[0],
            detector_columns=prepared_detector.shape[1],
            expected_fibers=trace_reference.shape[0],
            aperture_width=detector_extraction_width,
        )
    except Exception as error:
        reason = str(error)
        raise QuicklookError(
            f"Quick-look failed for exposure {exposure.exposure_id}, "
            f"amplifier {token}, stage: quick-trace fitting or validation, "
            f"reason: {reason}"
        ) from error

    try:
        with _timed_stage(timings, "topology", timing):
            topology_result = build_amplifier_topology(
                prepared_detector,
                identity,
                trace_root=trace_root,
                at=at,
                resource_root=resource_root,
                trace_result=trace_result,
                trace_provenance=trace_provenance,
                trace_source=trace_source,
                detector_column_start=detector_column_start,
                detector_column_stop=detector_column_stop,
            )
    except Exception as error:
        raise QuicklookError(
            f"Quick-look failed for exposure {exposure.exposure_id}, "
            f"amplifier {token}, stage: topology construction, reason: {error}"
        ) from error
    instrument = Instrument.from_value(topology_result.physical_identity.instrument)
    try:
        with _timed_stage(timings, "local_extraction_collapse", timing):
            if quicklook_kind == "flat":
                product = run_ldls_flat_quicklook(
                    prepared_detector,
                    topology_result.topology,
                    detector_variance=detector_variance,
                    extraction_width=detector_extraction_width,
                    collapse_columns=collapse_columns,
                    statistic=collapse_statistic,
                    gaussian_fwhm=gaussian_fwhm_arcsec,
                    pixel_scale=pixel_scale_arcsec,
                    grid_padding_arcsec=grid_padding_arcsec,
                    instrument=instrument,
                    output_shape=output_shape,
                    origin=origin,
                    retain_extraction=detailed_evidence,
                    build_spatial=detailed_evidence,
                )
            elif quicklook_kind == "standard":
                product = run_standard_star_quicklook(
                    prepared_detector,
                    topology_result.topology,
                    requested_position=requested_position,
                    detector_variance=detector_variance,
                    extraction_width=detector_extraction_width,
                    collapse_columns=collapse_columns,
                    statistic=collapse_statistic,
                    gaussian_fwhm=gaussian_fwhm_arcsec,
                    pixel_scale=pixel_scale_arcsec,
                    grid_padding_arcsec=grid_padding_arcsec,
                    instrument=instrument,
                    output_shape=output_shape,
                    origin=origin,
                    retain_extraction=detailed_evidence,
                    build_spatial=detailed_evidence,
                )
            elif quicklook_kind == "target":
                product = _spatial_quicklook(
                    prepared_detector,
                    topology_result.topology,
                    detector_variance=detector_variance,
                    extraction_width=detector_extraction_width,
                    collapse_columns=collapse_columns,
                    statistic=collapse_statistic,
                    gaussian_fwhm=gaussian_fwhm_arcsec,
                    pixel_scale=pixel_scale_arcsec,
                    grid_padding_arcsec=grid_padding_arcsec,
                    instrument=instrument,
                    output_shape=output_shape,
                    origin=origin,
                    retain_extraction=detailed_evidence,
                    build_spatial=detailed_evidence,
                )
            else:
                raise ValueError(f"Unsupported quicklook_kind: {quicklook_kind!r}")
    except Exception as error:
        stage = "target extraction" if quicklook_kind == "target" else "quick-look extraction"
        raise QuicklookError(
            f"Quick-look failed for exposure {exposure.exposure_id}, "
            f"amplifier {token}, stage: {stage}, reason: {error}"
        ) from error
    if timing:
        timings["total"] = perf_counter() - total_started
    retained_loaded = loaded if detailed_evidence else None
    retained_detector = detector if detailed_evidence else None
    diagnostics = QuicklookDiagnostics(
        stage_seconds=timings,
        retained_array_bytes=(
            _amplifier_array_bytes(
                retained_loaded,
                retained_detector,
                topology_result,
                product,
            )
            if memory_check
            else 0
        ),
        peak_rss_bytes=_peak_rss_bytes() if memory_check else None,
    )
    return retained_loaded, retained_detector, topology_result, product, diagnostics


def _unpack_archive_quicklook_result(result: tuple[object, ...]):
    """Accept legacy four-item test adapters while exposing diagnostics."""

    if len(result) == 4:
        return (*result, QuicklookDiagnostics())
    if len(result) == 5:
        return result
    raise ValueError("archive amplifier quick-look returned an invalid result tuple")


def run_lrs2_channel_quicklooks(
    exposure: "Exposure",
    *,
    trace_root: str | Path,
    at: date | datetime | str | None = None,
    frame_type: str | None = None,
    quicklook_kind: str = "flat",
    loader: "RawFrameLoader | None" = None,
    resource_root: str | Path | None = None,
    requested_position: tuple[float, float] | None = None,
    detector_extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    collapse_statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    gaussian_fwhm_arcsec: float | None = None,
    pixel_scale_arcsec: float | None = None,
    grid_padding_arcsec: float | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
    trace_provider: _QuickTraceProvider | None = None,
    detailed_evidence: bool = False,
) -> LRS2QuicklookSet:
    """Run the established amplifier path and compose all four LRS2 channels.

    Each of the eight expected amplifiers is loaded, reduced, traced,
    extracted, and collapsed independently. Composition happens only through
    the existing physical-fiber/value boundary in
    :func:`combine_lrs2_channels`.
    """

    exposure_instruments = {
        Instrument.from_value(identity.instrument)
        for identity in exposure.physical_identities.values()
    }
    if exposure_instruments != {Instrument.LRS2}:
        raise ValueError("run_lrs2_channel_quicklooks requires an LRS2 exposure")
    kind = str(quicklook_kind).strip().casefold()
    if kind not in {"flat", "standard", "target"}:
        raise ValueError("quicklook_kind must be 'flat', 'standard', or 'target'")

    if frame_type is None:
        frame_type = exposure.metadata.frame_type
    if frame_type is None:
        raise ValueError(
            "frame_type is required when the exposure contains multiple frame types"
        )
    requested_frame_type = str(frame_type).strip().casefold()
    expected_tokens = tuple(
        token
        for channel in ("UV", "Orange", "Red", "Far-Red")
        for token in lrs2_amplifier_tokens_for_channel(channel)
    )
    frames_by_token: dict[str, ArchiveMember] = {}
    for frame in exposure.frames:
        identity = frame.identity
        if identity is None:
            continue
        token = str(identity.amplifier_token).strip().upper()
        if token not in expected_tokens:
            continue
        if str(identity.frame_type).strip().casefold() != requested_frame_type:
            continue
        if token in frames_by_token:
            raise ValueError(
                f"Multiple {frame_type} frames found for LRS2 amplifier {token}"
            )
        frames_by_token[token] = frame

    missing = [token for token in expected_tokens if token not in frames_by_token]
    if missing:
        raise QuicklookError(
            f"Quick-look failed for exposure {exposure.exposure_id}, "
            "stage: target amplifier completeness, reason: missing required "
            f"amplifier frame(s): {', '.join(missing)}"
        )

    active_loader = loader
    evidence: dict[str, LRS2AmplifierQuicklookEvidence] = {}
    for token in expected_tokens:
        frame = frames_by_token[token]
        archive_result = _run_archive_amplifier_quicklook(
            exposure,
            frame,
            trace_root=trace_root,
            at=at,
            quicklook_kind=kind,
            loader=active_loader,
            resource_root=resource_root,
            requested_position=requested_position,
            detector_extraction_width=detector_extraction_width,
            collapse_columns=collapse_columns,
            collapse_statistic=collapse_statistic,
            gaussian_fwhm_arcsec=gaussian_fwhm_arcsec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            grid_padding_arcsec=grid_padding_arcsec,
            output_shape=output_shape,
            origin=origin,
            trace_provider=trace_provider,
            detailed_evidence=detailed_evidence,
        )
        loaded, detector, topology_result, product, diagnostics = (
            _unpack_archive_quicklook_result(archive_result)
        )
        evidence[token] = LRS2AmplifierQuicklookEvidence(
            frame=frame,
            loaded=loaded,
            detector=detector,
            topology=topology_result,
            product=product,
            diagnostics=diagnostics,
        )

    channels = combine_lrs2_channels(
        {token: item.product for token, item in evidence.items()},
        gaussian_fwhm_arcsec=gaussian_fwhm_arcsec,
        pixel_scale_arcsec=pixel_scale_arcsec,
        grid_padding_arcsec=grid_padding_arcsec,
        output_shape=output_shape,
        origin=origin,
    )
    return LRS2QuicklookSet(
        amplifier_evidence=evidence,
        channels=channels,
    )


def _compose_virus_ifu_quicklook(
    amplifier_evidence: Mapping[str, AmplifierQuicklookEvidence],
    *,
    ifu_slot: str,
    gaussian_fwhm_arcsec: float | None,
    pixel_scale_arcsec: float | None,
    grid_padding_arcsec: float | None,
    output_shape: tuple[int, int] | None,
    origin: tuple[float, float] | None,
) -> SpatialQuicklook:
    """Build one IFU image from the available amplifier fiber products."""

    expected_tokens = tuple(f"{ifu_slot}{amp}" for amp in ("LL", "LU", "RL", "RU"))
    available_tokens = tuple(
        token for token in expected_tokens if token in amplifier_evidence
    )
    if not available_tokens:
        raise QuicklookError(f"Cannot compose VIRUS IFU {ifu_slot}; no amplifier products")

    fiber_values: dict[str, float] = {}
    fiber_positions: dict[str, tuple[float, float]] = {}
    fiber_errors: dict[str, float] = {}
    first_product = amplifier_evidence[available_tokens[0]].product
    all_errors_available = True
    for token in available_tokens:
        product = amplifier_evidence[token].product
        for fiber_id, value in product.fiber_values.items():
            if fiber_id in fiber_values:
                raise QuicklookError(
                    f"Cannot compose VIRUS IFU {ifu_slot}; duplicate fiber {fiber_id}"
                )
            try:
                position = product.fiber_positions[fiber_id]
            except KeyError as error:
                raise QuicklookError(
                    f"Cannot compose VIRUS IFU {ifu_slot}; missing position for fiber "
                    f"{fiber_id}"
                ) from error
            fiber_values[fiber_id] = float(value)
            fiber_positions[fiber_id] = (float(position[0]), float(position[1]))
            if fiber_id in product.fiber_errors:
                fiber_errors[fiber_id] = float(product.fiber_errors[fiber_id])
            else:
                all_errors_available = False

    if not fiber_values:
        raise QuicklookError(f"Cannot compose VIRUS IFU {ifu_slot}; no fiber values")

    positions = np.asarray(tuple(fiber_positions.values()), dtype=float)
    values = np.asarray(tuple(fiber_values.values()), dtype=float)
    errors = (
        np.asarray(tuple(fiber_errors[fiber_id] for fiber_id in fiber_values), dtype=float)
        if all_errors_available
        else None
    )
    _, resolved_fwhm, resolved_pixel_scale, _ = _resolve_spatial_parameters(
        Instrument.VIRUS,
        gaussian_fwhm=gaussian_fwhm_arcsec,
        pixel_scale=pixel_scale_arcsec,
        spatial_defaults=None,
    )
    spatial = gaussian_splat(
        positions,
        values,
        errors,
        fwhm=resolved_fwhm,
        pixel_scale=resolved_pixel_scale,
        grid_padding=(
            grid_padding_arcsec
            if grid_padding_arcsec is not None
            else float(VIRUS_SPATIAL_DEFAULTS["grid_padding_arcsec"])
        ),
        output_shape=output_shape,
        origin=origin,
    )
    common = dict(
        fiber_values=fiber_values,
        image=spatial.image,
        instrument=Instrument.VIRUS,
        fiber_positions=fiber_positions,
        fiber_errors=fiber_errors if all_errors_available else {},
        spatial_weight=spatial.weight,
        spatial_support=spatial.support,
        spatial_x_coordinates=spatial.x_coordinates,
        spatial_y_coordinates=spatial.y_coordinates,
        spatial_gaussian_fwhm_arcsec=resolved_fwhm,
        spatial_pixel_scale_arcsec=resolved_pixel_scale,
        intended_fiducial=first_product.intended_fiducial,
        collapse_columns=first_product.collapse_columns,
        collapse_statistic=first_product.collapse_statistic,
        extraction_width=first_product.extraction_width,
    )
    if isinstance(first_product, PointingQuicklook):
        measured = weighted_centroid(positions[:, 0], positions[:, 1], values)
        return PointingQuicklook(
            **common,
            measured_centroid=measured,
            requested_position=first_product.requested_position,
        )
    return SpatialQuicklook(**common)


def run_virus_ifu_quicklooks(
    exposure: "Exposure",
    *,
    trace_root: str | Path,
    at: date | datetime | str | None = None,
    frame_type: str | None = None,
    quicklook_kind: str = "flat",
    loader: "RawFrameLoader | None" = None,
    resource_root: str | Path | None = None,
    requested_position: tuple[float, float] | None = None,
    detector_extraction_width: float = 5.0,
    collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
    collapse_statistic: str = DEFAULT_COLLAPSE_STATISTIC,
    gaussian_fwhm_arcsec: float | None = None,
    pixel_scale_arcsec: float | None = None,
    grid_padding_arcsec: float | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
    trace_provider: _QuickTraceProvider | None = None,
    timing: bool = False,
    memory_check: bool = False,
    detailed_evidence: bool = False,
) -> VIRUSQuicklookSet:
    """Run archive-backed VIRUS amplifier quick looks and compose each IFU.

    Each complete IFU returns one physical IFU-plane image built from all
    four amplifier products. The individual amplifier evidence remains
    available for diagnostics; no whole-VIRUS focal-plane image is inferred.
    """

    exposure_instruments = {
        Instrument.from_value(identity.instrument)
        for identity in exposure.physical_identities.values()
    }
    if exposure_instruments != {Instrument.VIRUS}:
        raise ValueError("run_virus_ifu_quicklooks requires a VIRUS exposure")
    kind = str(quicklook_kind).strip().casefold()
    if kind not in {"flat", "standard", "target"}:
        raise ValueError(
            "quicklook_kind must be 'flat', 'standard', or 'target'"
        )
    if frame_type is None:
        frame_type = exposure.metadata.frame_type
    if frame_type is None:
        raise ValueError(
            "frame_type is required when the exposure contains multiple frame types"
        )
    requested_frame_type = str(frame_type).strip().casefold()
    expected_amplifiers = ("LL", "LU", "RL", "RU")
    frames_by_ifu: dict[str, dict[str, "ArchiveMember"]] = {}
    for frame in exposure.frames:
        identity = frame.identity
        if identity is None or str(identity.frame_type).strip().casefold() != requested_frame_type:
            continue
        try:
            physical = exposure.identity_for(frame)
        except Exception as error:
            token = str(identity.amplifier_token).strip().upper()
            raise QuicklookError(
                f"Quick-look failed for exposure {exposure.exposure_id}, "
                f"amplifier {token}, stage: target identity, reason: {error}"
            ) from error
        if physical.instrument is not Instrument.VIRUS:
            continue
        slot = str(physical.ifu_slot).strip().zfill(3)
        amplifier = str(physical.amplifier).strip().upper()
        if amplifier not in expected_amplifiers:
            continue
        by_amplifier = frames_by_ifu.setdefault(slot, {})
        token = f"{slot}{amplifier}"
        if amplifier in by_amplifier:
            raise ValueError(
                f"Multiple {frame_type} frames found for VIRUS amplifier {token}"
            )
        by_amplifier[amplifier] = frame

    if not frames_by_ifu:
        raise QuicklookError(
            f"Quick-look failed for exposure {exposure.exposure_id}, "
            "stage: target amplifier completeness, reason: no required "
            "VIRUS IFU amplifier frames found"
        )

    # Validate every discovered IFU before loading any detector data. This
    # keeps structural archive failures fail-fast even when several IFUs are
    # present in one exposure.
    for slot, by_amplifier in frames_by_ifu.items():
        missing = tuple(
            f"{slot}{amplifier}"
            for amplifier in expected_amplifiers
            if amplifier not in by_amplifier
        )
        if missing:
            raise QuicklookError(
                f"Quick-look failed for exposure {exposure.exposure_id}, "
                f"IFU {slot}, stage: target amplifier completeness, reason: "
                f"missing required amplifier frame(s): {', '.join(missing)}"
            )

    def process_ifu(
        slot: str,
    ) -> tuple[str, VIRUSIFUQuicklookSet, QuicklookDiagnostics]:
        by_amplifier = frames_by_ifu[slot]
        ifu_started = perf_counter() if timing else 0.0

        def process_amplifier(amplifier: str):
            frame = by_amplifier[amplifier]
            result = _run_archive_amplifier_quicklook(
                exposure,
                frame,
                trace_root=trace_root,
                at=at,
                quicklook_kind=kind,
                loader=loader,
                resource_root=resource_root,
                requested_position=requested_position,
                detector_extraction_width=detector_extraction_width,
                collapse_columns=collapse_columns,
                collapse_statistic=collapse_statistic,
                gaussian_fwhm_arcsec=gaussian_fwhm_arcsec,
                pixel_scale_arcsec=pixel_scale_arcsec,
                grid_padding_arcsec=grid_padding_arcsec,
                output_shape=output_shape,
                origin=origin,
                trace_provider=trace_provider,
                timing=timing,
                memory_check=memory_check,
                detailed_evidence=detailed_evidence,
            )
            return _unpack_archive_quicklook_result(result)

        processed: dict[str, tuple[object, ...]] = {}
        unavailable_amplifiers: dict[str, str] = {}
        for amplifier in expected_amplifiers:
            try:
                processed[amplifier] = process_amplifier(amplifier)
            except QuicklookError as error:
                unavailable_amplifiers[f"{slot}{amplifier}"] = str(error)

        evidence: dict[str, AmplifierQuicklookEvidence] = {}
        for amplifier in expected_amplifiers:
            if amplifier not in processed:
                continue
            loaded, detector, topology_result, product, diagnostics = processed[amplifier]
            token = f"{slot}{amplifier}"
            evidence[token] = AmplifierQuicklookEvidence(
                frame=by_amplifier[amplifier],
                loaded=loaded,
                detector=detector,
                topology=topology_result,
                product=product,
                diagnostics=diagnostics,
            )

        composition_timings: dict[str, float] = {}
        with _timed_stage(composition_timings, "ifu_composition", timing):
            product = _compose_virus_ifu_quicklook(
                evidence,
                ifu_slot=slot,
                gaussian_fwhm_arcsec=gaussian_fwhm_arcsec,
                pixel_scale_arcsec=pixel_scale_arcsec,
                grid_padding_arcsec=grid_padding_arcsec,
                output_shape=output_shape,
                origin=origin,
            )
        if timing:
            composition_timings["total"] = perf_counter() - ifu_started
        stage_seconds = {
            f"amplifier.{token}.{stage}": seconds
            for token, item in evidence.items()
            for stage, seconds in item.diagnostics.stage_seconds.items()
        }
        stage_seconds.update(
            composition_timings
        )
        retained_bytes = (
            sum(item.diagnostics.retained_array_bytes for item in evidence.values())
            + (_spatial_array_bytes(product) if memory_check else 0)
        )
        peak_values = [
            item.diagnostics.peak_rss_bytes
            for item in evidence.values()
            if item.diagnostics.peak_rss_bytes is not None
        ]
        if memory_check:
            current_peak = _peak_rss_bytes()
            if current_peak is not None:
                peak_values.append(current_peak)
        diagnostics = QuicklookDiagnostics(
            stage_seconds=stage_seconds,
            retained_array_bytes=retained_bytes,
            peak_rss_bytes=max(peak_values) if peak_values else None,
        )
        return (
            slot,
            VIRUSIFUQuicklookSet(
                ifu_slot=slot,
                amplifier_evidence=evidence,
                product=product,
                unavailable_amplifiers=unavailable_amplifiers,
                diagnostics=diagnostics,
            ),
            diagnostics,
        )

    processed_ifus: dict[str, tuple[VIRUSIFUQuicklookSet, QuicklookDiagnostics]] = {}
    slots = tuple(sorted(frames_by_ifu))
    for slot in slots:
        _, result, diagnostics = process_ifu(slot)
        processed_ifus[slot] = (result, diagnostics)

    ifus = {slot: processed_ifus[slot][0] for slot in slots}
    ifu_diagnostics = {slot: processed_ifus[slot][1] for slot in slots}
    return VIRUSQuicklookSet(ifus=ifus, diagnostics=ifu_diagnostics)


def _canonical_lrs2_channel_products(
    channel: str,
    amplifier_products: Mapping[str, SpatialQuicklook],
) -> tuple[str, dict[str, SpatialQuicklook]]:
    """Validate and order the two products belonging to ``channel``."""

    component = lrs2_component_for_channel(channel)
    expected_tokens = lrs2_amplifier_tokens_for_channel(component.name)
    expected_set = set(expected_tokens)
    normalized: dict[str, SpatialQuicklook] = {}

    for supplied_key, product in amplifier_products.items():
        key = str(supplied_key).strip().upper()
        candidates = [token for token in expected_tokens if key == token]
        # The full token is the normal interface.  Accepting the two-letter
        # suffix is useful for direct in-memory calls, but only within the
        # already selected channel so it cannot infer a cross-slot pairing.
        if not candidates:
            candidates = [token for token in expected_tokens if key == token[-2:]]
        if len(candidates) != 1:
            raise ValueError(
                f"Amplifier {supplied_key!r} does not belong uniquely to LRS2 "
                f"channel {component.name!r}; expected {expected_tokens}"
            )
        token = candidates[0]
        if token in normalized:
            raise ValueError(f"Duplicate LRS2 amplifier product for {token}")
        if not isinstance(product, SpatialQuicklook):
            raise TypeError("amplifier_products must contain SpatialQuicklook results")
        if product.instrument is not None and Instrument.from_value(product.instrument) is not Instrument.LRS2:
            raise ValueError(f"LRS2 channel product {token} has non-LRS2 instrument")
        normalized[token] = product

    missing = [token for token in expected_tokens if token not in normalized]
    if missing:
        raise ValueError(
            f"Cannot create complete LRS2 channel {component.name!r}; "
            f"missing amplifier product(s): {', '.join(missing)}"
        )
    if set(normalized) != expected_set:
        raise ValueError(
            f"LRS2 channel {component.name!r} received an unexpected amplifier pair"
        )
    return component.name, {token: normalized[token] for token in expected_tokens}


def _channel_product_arrays(
    amplifier_token: str,
    product: SpatialQuicklook,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return position/value/error arrays in the product's fiber order."""

    fiber_ids = tuple(product.fiber_values)
    if set(fiber_ids) != set(product.fiber_positions):
        raise ValueError(
            f"LRS2 amplifier product {amplifier_token} has mismatched fiber "
            "value and position identifiers"
        )
    if len(fiber_ids) != 140:
        raise ValueError(
            f"LRS2 amplifier product {amplifier_token} must contain 140 fibers; "
            f"found {len(fiber_ids)}"
        )
    positions = np.asarray(
        [product.fiber_positions[fiber_id] for fiber_id in fiber_ids], dtype=float
    )
    values = np.asarray(
        [product.fiber_values[fiber_id] for fiber_id in fiber_ids], dtype=float
    )
    if positions.shape != (140, 2) or values.shape != (140,):
        raise ValueError(
            f"LRS2 amplifier product {amplifier_token} has invalid fiber array shapes"
        )

    errors: np.ndarray | None = None
    if product.fiber_errors and set(product.fiber_errors) == set(fiber_ids):
        errors = np.asarray(
            [product.fiber_errors[fiber_id] for fiber_id in fiber_ids], dtype=float
        )
    return positions, values, errors


def combine_lrs2_channel_products(
    channel: str,
    amplifier_products: Mapping[str, SpatialQuicklook],
    *,
    gaussian_fwhm_arcsec: float | None = None,
    pixel_scale_arcsec: float | None = None,
    grid_padding_arcsec: float | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> LRS2ChannelQuicklook:
    """Combine two independently reduced LRS2 amplifiers into one channel.

    Composition begins at the collapsed physical-fiber boundary.  The
    detector arrays, traces, extracted spectra, and detector variances remain
    available only on the two underlying :class:`SpatialQuicklook` products.
    """

    canonical_channel, products = _canonical_lrs2_channel_products(
        channel, amplifier_products
    )
    positions_and_values = [
        _channel_product_arrays(token, products[token])
        for token in products
    ]
    positions = np.concatenate(
        [item[0] for item in positions_and_values], axis=0
    )
    values = np.concatenate([item[1] for item in positions_and_values], axis=0)
    errors_by_amp = [item[2] for item in positions_and_values]
    errors = (
        np.concatenate([error for error in errors_by_amp if error is not None], axis=0)
        if all(error is not None for error in errors_by_amp)
        else None
    )
    if positions.shape != (280, 2) or values.shape != (280,):
        raise ValueError(
            f"A complete LRS2 channel must contain 280 fibers; found {values.size}"
        )

    defaults = spatial_defaults_for(Instrument.LRS2)
    resolved_fwhm = (
        float(defaults["gaussian_fwhm_arcsec"])
        if gaussian_fwhm_arcsec is None
        else float(gaussian_fwhm_arcsec)
    )
    resolved_pixel_scale = (
        float(defaults["pixel_scale_arcsec"])
        if pixel_scale_arcsec is None
        else float(pixel_scale_arcsec)
    )
    resolved_grid_padding = (
        defaults.get("grid_padding_arcsec")
        if grid_padding_arcsec is None
        else grid_padding_arcsec
    )
    intended = tuple(defaults["intended_fiducial"])
    spatial = gaussian_splat(
        positions,
        values,
        errors,
        fwhm=resolved_fwhm,
        pixel_scale=resolved_pixel_scale,
        grid_padding=resolved_grid_padding,
        output_shape=output_shape,
        origin=origin,
    )

    pointing_products = [isinstance(product, PointingQuicklook) for product in products.values()]
    if any(pointing_products) and not all(pointing_products):
        raise TypeError(
            "LRS2 channel products must all be standard-star or all be flat products"
        )
    measured = (
        weighted_centroid(positions[:, 0], positions[:, 1], values)
        if all(pointing_products)
        else None
    )
    return LRS2ChannelQuicklook(
        channel=canonical_channel,
        amplifier_products=products,
        fiber_positions=positions,
        fiber_values=values,
        image=spatial.image,
        spatial_x_coordinates=spatial.x_coordinates,
        spatial_y_coordinates=spatial.y_coordinates,
        spatial_support=spatial.support,
        spatial_weight=spatial.weight,
        intended_fiducial=(float(intended[0]), float(intended[1])),
        measured_centroid=measured,
        spatial_gaussian_fwhm_arcsec=resolved_fwhm,
        spatial_pixel_scale_arcsec=resolved_pixel_scale,
        fiber_errors=errors,
        spatial_variance=spatial.variance,
    )


def combine_lrs2_channels(
    amplifier_products: Mapping[str, SpatialQuicklook],
    *,
    gaussian_fwhm_arcsec: float | None = None,
    pixel_scale_arcsec: float | None = None,
    grid_padding_arcsec: float | None = None,
    output_shape: tuple[int, int] | None = None,
    origin: tuple[float, float] | None = None,
) -> dict[str, LRS2ChannelQuicklook]:
    """Combine a complete eight-amplifier LRS2 exposure into four channels.

    The function requires all eight authoritative amplifier tokens.  Missing
    data therefore remain explicit instead of being represented as a partial
    channel with an apparently complete result type.
    """

    expected_tokens = {
        token
        for channel in ("UV", "Orange", "Red", "Far-Red")
        for token in lrs2_amplifier_tokens_for_channel(channel)
    }
    canonical_products: dict[str, SpatialQuicklook] = {}
    for supplied_key, product in amplifier_products.items():
        token = str(supplied_key).strip().upper()
        if token in canonical_products:
            raise ValueError(f"Duplicate LRS2 amplifier product for {token}")
        canonical_products[token] = product
    supplied_tokens = set(canonical_products)
    missing = sorted(expected_tokens - supplied_tokens)
    unexpected = sorted(supplied_tokens - expected_tokens)
    if missing:
        raise ValueError(
            "Cannot create complete LRS2 channel set; missing amplifier product(s): "
            + ", ".join(missing)
        )
    if unexpected:
        raise ValueError(
            "Unexpected amplifier product(s) for LRS2 channel set: "
            + ", ".join(unexpected)
        )
    return {
        channel: combine_lrs2_channel_products(
            channel,
            {token: canonical_products[token] for token in lrs2_amplifier_tokens_for_channel(channel)},
            gaussian_fwhm_arcsec=gaussian_fwhm_arcsec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            grid_padding_arcsec=grid_padding_arcsec,
            output_shape=output_shape,
            origin=origin,
        )
        for channel in ("UV", "Orange", "Red", "Far-Red")
    }

"""Small interactive orchestration objects for archive-backed quick looks."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from html import escape
from pathlib import Path
from typing import Any

from .algorithms.collapse import DEFAULT_COLLAPSE_COLUMNS, DEFAULT_COLLAPSE_STATISTIC
from . import workflows
from .classification import ExposureClassification, StandardStarCatalog
from .config import QuicklookConfig
from .discovery import DiscoveredObservation, discover_observations
from .instrument import Instrument
from .observation import Exposure, Observation, load_observation
from .raw import RawFrameLoader


_TABLE_COLUMNS = (
    "Row",
    "Observation",
    "Exposure ID",
    "UTC/time",
    "Frame type",
    "OBJECT",
    "Quick-look kind",
    "Calibration / standard",
    "IFU slot",
    "Exposure time [s]",
    "Program",
    "Amplifiers/components",
)


def _package_trace_root() -> Path:
    """Return the package/repository root containing ``Fiber_Locations``."""

    # In the supported source and editable installations this module lives at
    # ``<repository>/src/hetquicklook/highlevel.py``.  Resolve from the
    # package location so notebook and process working directories do not
    # affect dated trace lookup.
    return Path(__file__).resolve().parents[2]


def _display_value(value: Any) -> str:
    """Convert metadata to a compact, safe table value."""

    if value is None:
        return "—"
    if isinstance(value, str):
        return value if value else "—"
    if isinstance(value, (datetime, date)):
        if isinstance(value, datetime):
            return value.isoformat(sep=" ")
        return value.isoformat()
    if isinstance(value, (tuple, list, set, frozenset)):
        values = tuple(_display_value(item) for item in value)
        return ", ".join(values) if values else "—"
    return str(value)


def _classification_label(classification: ExposureClassification) -> str:
    """Return the user-facing label without reimplementing classification."""

    if classification.quicklook_kind == "flat":
        return "flat"
    if classification.standard_star is True:
        return "standard"
    if classification.standard_star is False:
        return "science"
    return "science" if classification.standard_target is None else "unclassified"


def _calibration_label(classification: ExposureClassification) -> str | None:
    if classification.calibration_source is not None:
        return classification.calibration_source
    if classification.standard_star is True and classification.standard_target:
        return f"standard: {classification.standard_target}"
    return None


def _exposure_row(row: int, observation: Observation, exposure: Exposure) -> dict[str, Any]:
    metadata = exposure.metadata
    classification = exposure.classification
    physical_slots = tuple(
        sorted(
            {
                str(identity.ifu_slot).strip().zfill(3)
                for identity in exposure.physical_identities.values()
            }
        )
    )
    if not physical_slots and metadata.requested_ifuslot is not None:
        physical_slots = (metadata.requested_ifuslot,)
    return {
        "Row": row,
        "Observation": observation.observation_id,
        "Exposure ID": exposure.exposure_id,
        "UTC/time": metadata.observation_time,
        "Frame type": metadata.frame_type or metadata.frame_types,
        "OBJECT": metadata.object_name,
        "Quick-look kind": _classification_label(classification),
        "Calibration / standard": _calibration_label(classification),
        "IFU slot": physical_slots,
        "Exposure time [s]": metadata.exposure_time_s,
        "Program": metadata.program_id or metadata.qprog,
        "Amplifiers/components": exposure.amplifier_tokens,
    }


@dataclass(frozen=True)
class QuicklookSite:
    """Reusable filesystem configuration for an interactive session.

    The dated ``Fiber_Locations`` tree is resolved from the package location
    by default.  ``trace_root`` remains available as an explicit override for
    deployments that keep an alternate trace tree outside the repository.
    """

    raw_roots: Mapping[Instrument | str, str | Path]
    trace_root: str | Path | None = None
    resource_root: str | Path | None = None
    standard_catalog: StandardStarCatalog | None = None
    _configs: Mapping[str, QuicklookConfig] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        normalized: dict[str, Path] = {}
        for instrument, root in self.raw_roots.items():
            parsed = Instrument.from_value(instrument)
            if parsed.value in normalized:
                raise ValueError(f"Duplicate raw root for instrument {parsed.value!r}")
            normalized[parsed.value] = Path(root).expanduser().resolve()
        if not normalized:
            raise ValueError("raw_roots must contain at least one instrument root")
        object.__setattr__(self, "raw_roots", normalized)
        trace_root = (
            _package_trace_root()
            if self.trace_root is None
            else Path(self.trace_root).expanduser().resolve()
        )
        object.__setattr__(self, "trace_root", trace_root)
        normalized_resource = (
            None
            if self.resource_root is None
            else Path(self.resource_root).expanduser().resolve()
        )
        object.__setattr__(self, "resource_root", normalized_resource)
        object.__setattr__(
            self,
            "_configs",
            {name: QuicklookConfig(root) for name, root in normalized.items()},
        )

    def config_for(self, instrument: Instrument | str) -> QuicklookConfig:
        """Return the existing discovery configuration for one instrument."""

        parsed = Instrument.from_value(instrument)
        try:
            return self._configs[parsed.value]
        except KeyError as error:
            available = ", ".join(sorted(self._configs))
            raise ValueError(
                f"No raw root configured for instrument {parsed.value!r}; "
                f"available roots: {available}"
            ) from error

    def night(
        self,
        date: str | date,
        *,
        instrument: Instrument | str,
    ) -> "QuicklookNight":
        """Discover and load exposure metadata for one instrument night."""

        parsed = Instrument.from_value(instrument)
        discovered = discover_observations(
            self.config_for(parsed), instrument=parsed, date=date
        )
        observations = tuple(
            load_observation(item, standard_catalog=self.standard_catalog)
            for item in discovered
        )
        return QuicklookNight(
            site=self,
            date=date,
            instrument=parsed,
            discovered=discovered,
            observations=observations,
        )


@dataclass(frozen=True)
class QuicklookNight:
    """A stable, flattened exposure inventory for one date and instrument."""

    site: QuicklookSite
    date: str | date
    instrument: Instrument
    discovered: tuple[DiscoveredObservation, ...]
    observations: tuple[Observation, ...]
    _exposures: tuple["QuicklookExposure", ...] = field(init=False, repr=False)
    _trace_provider: workflows._QuickTraceProvider = field(init=False, repr=False)

    def __post_init__(self) -> None:
        parsed = Instrument.from_value(self.instrument)
        object.__setattr__(self, "instrument", parsed)
        if len(self.discovered) != len(self.observations):
            raise ValueError("discovered and observations must have matching lengths")
        entries: list[QuicklookExposure] = []
        for observation in self.observations:
            if Instrument.from_value(observation.discovered.instrument) is not parsed:
                raise ValueError("all observations must match the night instrument")
            for raw_exposure in sorted(
                observation.exposures, key=lambda item: item.exposure_id
            ):
                entries.append(
                    QuicklookExposure(
                        night=self,
                        observation=observation,
                        raw_exposure=raw_exposure,
                        row=len(entries),
                    )
                )
        object.__setattr__(self, "_exposures", tuple(entries))
        candidates = tuple(
            (raw_exposure, observation.discovered.date)
            for observation in self.observations
            for raw_exposure in observation.exposures
        )
        object.__setattr__(
            self,
            "_trace_provider",
            workflows._QuickTraceProvider(
                candidates,
                trace_root=self.site.trace_root,
            ),
        )

    @property
    def exposures(self) -> tuple["QuicklookExposure", ...]:
        """Return the flattened, display-ordered exposure wrappers."""

        return self._exposures

    def __len__(self) -> int:
        return len(self._exposures)

    def __iter__(self) -> Iterator["QuicklookExposure"]:
        return iter(self._exposures)

    def __getitem__(
        self, index: int | slice
    ) -> "QuicklookExposure | tuple[QuicklookExposure, ...]":
        return self._exposures[index]

    def exposure(self, exposure_id: str) -> "QuicklookExposure":
        """Find an exposure by encoded ID, rejecting archive ambiguity."""

        matches = [item for item in self._exposures if item.exposure_id == exposure_id]
        if not matches:
            raise KeyError(exposure_id)
        if len(matches) > 1:
            observations = ", ".join(item.observation.observation_id for item in matches)
            raise ValueError(
                f"Exposure ID {exposure_id!r} is ambiguous across observations: "
                f"{observations}"
            )
        return matches[0]

    def _rows(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            _exposure_row(item.row, item.observation, item.raw_exposure)
            for item in self._exposures
        )

    def _repr_html_(self) -> str:
        """Render a compact escaped exposure inventory for Jupyter."""

        header = "".join(
            f"<th style='border:1px solid #bbb;padding:4px 6px;text-align:left'>"
            f"{escape(column)}</th>"
            for column in _TABLE_COLUMNS
        )
        body: list[str] = []
        for row in self._rows():
            cells = "".join(
                "<td style='border:1px solid #bbb;padding:4px 6px'>"
                f"{escape(_display_value(row[column]))}</td>"
                for column in _TABLE_COLUMNS
            )
            body.append(f"<tr>{cells}</tr>")
        if not body:
            body.append(
                f"<tr><td colspan='{len(_TABLE_COLUMNS)}' "
                "style='padding:6px'>No exposures discovered.</td></tr>"
            )
        caption = (
            f"{self.instrument.value.upper()} {escape(_display_value(self.date))}: "
            f"{len(self)} exposure(s)"
        )
        return (
            f"<p><strong>{caption}</strong></p>"
            "<div style='overflow-x:auto'><table style='border-collapse:collapse'>"
            f"<thead><tr>{header}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table></div>"
        )

    def __repr__(self) -> str:
        return (
            f"QuicklookNight(date={self.date!r}, instrument={self.instrument.value!r}, "
            f"exposures={len(self)})"
        )


@dataclass(frozen=True)
class QuicklookExposure:
    """Thin interactive wrapper around one authoritative raw exposure."""

    night: QuicklookNight
    observation: Observation
    raw_exposure: Exposure
    row: int

    @property
    def exposure(self) -> Exposure:
        """Short alias for the underlying authoritative exposure."""

        return self.raw_exposure

    @property
    def exposure_id(self) -> str:
        return self.raw_exposure.exposure_id

    @property
    def metadata(self):
        return self.raw_exposure.metadata

    @property
    def classification(self) -> ExposureClassification:
        return self.raw_exposure.classification

    @property
    def instrument(self) -> Instrument:
        return self.night.instrument

    @property
    def kind(self) -> str:
        return _classification_label(self.classification)

    def _resolve_kind(self, kind: str | None) -> str:
        if kind is None or str(kind).strip().casefold() in {"", "auto"}:
            if self.classification.quicklook_kind == "flat":
                return "flat"
            if self.classification.standard_star is True:
                return "standard"
            if self.metadata.frame_class == "science":
                return "target"
            raise ValueError(
                f"Exposure {self.exposure_id!r} is classified as {self.kind!r} "
                "and has no supported automatic quick look; use an established "
                "flat or recognized standard-star exposure, or pass an explicit kind."
            )
        resolved = str(kind).strip().casefold()
        if resolved not in {"flat", "standard", "target"}:
            raise ValueError(
                "kind must be 'flat', 'standard', 'target', or None for automatic dispatch"
            )
        return resolved

    def quicklook(
        self,
        *,
        kind: str | None = None,
        frame_type: str | None = None,
        loader: RawFrameLoader | None = None,
        requested_position: tuple[float, float] | None = None,
        detector_extraction_width: float = 5.0,
        collapse_columns: int = DEFAULT_COLLAPSE_COLUMNS,
        collapse_statistic: str = DEFAULT_COLLAPSE_STATISTIC,
        gaussian_fwhm_arcsec: float | None = None,
        pixel_scale_arcsec: float | None = None,
        grid_padding_arcsec: float | None = None,
        output_shape: tuple[int, int] | None = None,
        origin: tuple[float, float] | None = None,
        resource_root: str | Path | None = None,
    ) -> "QuicklookProduct":
        """Build a product using existing classification and workflows."""

        resolved_kind = self._resolve_kind(kind)
        if frame_type is None:
            frame_type = self.metadata.frame_type
        if frame_type is None:
            raise ValueError(
                f"Exposure {self.exposure_id!r} has multiple or missing frame types; "
                "pass frame_type explicitly."
            )
        trace_root = self.night.site.trace_root
        configured_resource_root = (
            self.night.site.resource_root if resource_root is None else resource_root
        )
        common = dict(
            trace_root=trace_root,
            at=self.observation.discovered.date,
            frame_type=frame_type,
            quicklook_kind=resolved_kind,
            loader=loader,
            resource_root=configured_resource_root,
            requested_position=requested_position,
            detector_extraction_width=detector_extraction_width,
            collapse_columns=collapse_columns,
            collapse_statistic=collapse_statistic,
            gaussian_fwhm_arcsec=gaussian_fwhm_arcsec,
            pixel_scale_arcsec=pixel_scale_arcsec,
            grid_padding_arcsec=grid_padding_arcsec,
            output_shape=output_shape,
            origin=origin,
            trace_provider=self.night._trace_provider,
        )
        if self.instrument is Instrument.LRS2:
            raw_result = workflows.run_lrs2_channel_quicklooks(
                self.raw_exposure,
                **common,
            )
        else:
            raw_result = workflows.run_virus_ifu_quicklooks(
                self.raw_exposure,
                **common,
            )
        return QuicklookProduct.from_result(self, resolved_kind, raw_result)

    def __repr__(self) -> str:
        return (
            f"QuicklookExposure(row={self.row}, exposure_id={self.exposure_id!r}, "
            f"instrument={self.instrument.value!r}, kind={self.kind!r})"
        )


@dataclass(frozen=True)
class QuicklookIFU:
    """High-level VIRUS product for one IFU's retained amplifier evidence."""

    raw_result: workflows.VIRUSIFUQuicklookSet

    @property
    def ifu_slot(self) -> str:
        return self.raw_result.ifu_slot

    @property
    def amplifier_evidence(self):
        return self.raw_result.amplifier_evidence

    @property
    def amplifier_products(self):
        return self.raw_result.amplifier_products

    def plot(
        self,
        *,
        title: str | None = None,
        percentiles: tuple[float, float] = (2.0, 98.0),
        show_fibers: bool = True,
        show_fiducial: bool = False,
        show_centroid: bool = False,
    ) -> Any:
        from .visualization import plot_virus_ifu_amplifiers

        return plot_virus_ifu_amplifiers(
            self.amplifier_products,
            ifu_slot=self.ifu_slot,
            title=title or f"VIRUS IFU {self.ifu_slot}",
            percentiles=percentiles,
            show_fibers=show_fibers,
            show_fiducial=show_fiducial,
            show_centroid=show_centroid,
        )


@dataclass(frozen=True)
class QuicklookProduct:
    """Presentation wrapper retaining the underlying workflow result tree."""

    exposure: QuicklookExposure
    kind: str
    instrument: Instrument
    raw_result: workflows.LRS2QuicklookSet | workflows.VIRUSQuicklookSet
    channels: Mapping[str, Any] = field(default_factory=dict)
    ifus: Mapping[str, QuicklookIFU] = field(default_factory=dict)
    amplifier_evidence: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_result(
        cls,
        exposure: QuicklookExposure,
        kind: str,
        result: workflows.LRS2QuicklookSet | workflows.VIRUSQuicklookSet,
    ) -> "QuicklookProduct":
        if isinstance(result, workflows.LRS2QuicklookSet):
            return cls(
                exposure=exposure,
                kind=kind,
                instrument=Instrument.LRS2,
                raw_result=result,
                channels=result.channels,
                amplifier_evidence=result.amplifier_evidence,
            )
        if isinstance(result, workflows.VIRUSQuicklookSet):
            ifus = {
                slot: QuicklookIFU(ifu_result)
                for slot, ifu_result in result.ifus.items()
            }
            return cls(
                exposure=exposure,
                kind=kind,
                instrument=Instrument.VIRUS,
                raw_result=result,
                ifus=ifus,
                amplifier_evidence=result.amplifier_evidence,
            )
        raise TypeError(f"Unsupported quick-look result: {type(result).__name__}")

    @property
    def amplifier_products(self):
        return {
            token: evidence.product
            for token, evidence in self.amplifier_evidence.items()
        }

    def plot(
        self,
        *,
        title: str | None = None,
        percentiles: tuple[float, float] = (2.0, 98.0),
        show_fibers: bool = True,
        show_fiducial: bool | None = None,
        show_centroid: bool | None = None,
        ifu: str | None = None,
    ) -> Any:
        """Dispatch to the instrument-level presentation available here."""

        standard = self.kind == "standard"
        show_fiducial = standard if show_fiducial is None else show_fiducial
        show_centroid = standard if show_centroid is None else show_centroid
        if self.instrument is Instrument.LRS2:
            from .visualization import plot_lrs2_channels

            return plot_lrs2_channels(
                self.channels,
                title=title or f"LRS2 {self.kind} channel quick look",
                percentiles=percentiles,
                show_fibers=show_fibers,
                show_fiducial=show_fiducial,
                show_centroid=show_centroid,
            )
        if not self.ifus:
            raise ValueError("VIRUS quick look contains no processable IFU products")
        if ifu is None:
            if len(self.ifus) != 1:
                available = ", ".join(sorted(self.ifus))
                raise ValueError(
                    "VIRUS exposure contains multiple IFUs; select one with "
                    f"product.plot(ifu=...) (available: {available})"
                )
            selected = next(iter(self.ifus.values()))
        else:
            try:
                selected = self.ifus[str(ifu).strip().zfill(3)]
            except KeyError as error:
                available = ", ".join(sorted(self.ifus))
                raise KeyError(f"Unknown VIRUS IFU {ifu!r}; available: {available}") from error
        return selected.plot(
            title=title or f"VIRUS IFU {selected.ifu_slot} {self.kind} quick look",
            percentiles=percentiles,
            show_fibers=show_fibers,
            show_fiducial=show_fiducial,
            show_centroid=show_centroid,
        )

    def __repr__(self) -> str:
        if self.instrument is Instrument.LRS2:
            detail = f"channels={tuple(self.channels)}"
        else:
            detail = f"ifus={tuple(self.ifus)}"
        return (
            f"QuicklookProduct(instrument={self.instrument.value!r}, "
            f"kind={self.kind!r}, {detail})"
        )

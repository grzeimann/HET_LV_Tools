"""Small deterministic exposure and standard-star classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet

from .instrument import Instrument


@dataclass(frozen=True)
class ObjectIntent:
    """Intent encoded by a VIRUS-style ``OBJECT`` value."""

    raw_object: str | None
    target: str | None = None
    requested_ifuslot: str | None = None
    track: str | None = None
    is_parallel: bool = False


def parse_object_intent(value: object) -> ObjectIntent:
    """Parse ``TARGET_IFUSLOT_TRACK`` from the right-hand side.

    A target may contain underscores. Only a three-digit slot and final ``E``
    or ``W`` track make the structured form valid.
    """

    if value in (None, ""):
        return ObjectIntent(raw_object=None)
    text = str(value).strip()
    if text.casefold() == "parallel":
        return ObjectIntent(raw_object=text, is_parallel=True)
    parts = text.rsplit("_", 2)
    if (
        len(parts) == 3
        and parts[0]
        and len(parts[1]) == 3
        and parts[1].isdigit()
        and parts[2].upper() in {"E", "W"}
    ):
        return ObjectIntent(
            raw_object=text,
            target=parts[0],
            requested_ifuslot=parts[1],
            track=parts[2].upper(),
        )
    return ObjectIntent(raw_object=text)


@dataclass(frozen=True)
class StandardStarCatalog:
    """Explicit standard-star membership boundary.

    ``names=None`` means that no canonical catalog is available. Membership
    then returns ``None`` rather than guessing from historical lists.
    """

    names: FrozenSet[str] | None = None
    source: str | None = None

    @classmethod
    def unavailable(cls, *, source: str | None = None) -> "StandardStarCatalog":
        """Return a catalog object representing unavailable knowledge."""

        return cls(names=None, source=source)

    @classmethod
    def from_names(
        cls,
        names: set[str] | frozenset[str] | tuple[str, ...],
        *,
        source: str | None = None,
    ) -> "StandardStarCatalog":
        """Construct an explicit exact-membership catalog."""

        return cls(
            names=frozenset(str(name).strip().casefold() for name in names),
            source=source,
        )

    @property
    def available(self) -> bool:
        """Whether this object contains a canonical name set."""

        return self.names is not None

    def contains(self, target: str | None) -> bool | None:
        """Return exact case-insensitive membership, or ``None`` if unavailable."""

        if self.names is None or target is None:
            return None
        return str(target).strip().casefold() in self.names


@dataclass(frozen=True)
class ExposureClassification:
    """Classification result kept separate from raw and header metadata."""

    quicklook_kind: str | None = None
    calibration_source: str | None = None
    applicable_ifu_slots: tuple[str, ...] = ()
    standard_target: str | None = None
    standard_star: bool | None = None
    standard_catalog_available: bool = False
    standard_catalog_source: str | None = None

    @property
    def known(self) -> bool:
        """Whether a flat source or standard-star result is known."""

        return self.quicklook_kind is not None or self.standard_star is not None


def classify_exposure(
    instrument: Instrument | str,
    *,
    frame_types: tuple[str, ...] | list[str],
    object_name: str | None,
    ifu_slots: tuple[str, ...] | list[str] = (),
    standard_catalog: StandardStarCatalog | None = None,
) -> ExposureClassification:
    """Apply the explicit VIRUS/LRS2 flat rules and standard boundary."""

    parsed_instrument = Instrument.from_value(instrument)
    normalized_types = {str(value).strip().casefold() for value in frame_types}
    normalized_object = None if object_name is None else str(object_name).strip().casefold()
    slots = tuple(sorted({str(value).strip().zfill(3) for value in ifu_slots}))
    quicklook_kind = None
    source = None
    applicable: tuple[str, ...] = ()
    if normalized_types == {"flt"}:
        if parsed_instrument is Instrument.VIRUS and normalized_object == "ldls_long":
            quicklook_kind = "flat"
            source = "LDLS"
            applicable = slots
        elif parsed_instrument is Instrument.LRS2:
            if normalized_object == "ldls_long_b" and "056" in slots:
                quicklook_kind = "flat"
                source = "LDLS"
                applicable = ("056",)
            elif normalized_object == "qth_r" and "066" in slots:
                quicklook_kind = "flat"
                source = "Qth"
                applicable = ("066",)

    catalog = standard_catalog or StandardStarCatalog.unavailable()
    object_intent = parse_object_intent(object_name)
    standard_star: bool | None = None
    if normalized_types == {"sci"} and object_intent.target is not None:
        standard_star = catalog.contains(object_intent.target)
    return ExposureClassification(
        quicklook_kind=quicklook_kind,
        calibration_source=source,
        applicable_ifu_slots=applicable,
        standard_target=object_intent.target,
        standard_star=standard_star,
        standard_catalog_available=catalog.available,
        standard_catalog_source=catalog.source,
    )


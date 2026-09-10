"""Small deterministic exposure and standard-star classification helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping

from .instrument import Instrument


# This is the HET/Hydra table used as the default quick-look catalog.  Keep the
# spelling here exact: historical Panacea spellings are handled by the
# explicit alias table below rather than by broad matching.
STANDARD_STAR_NAMES = frozenset(
    {
        "1732526",
        "1740346",
        "1743045",
        "1757132",
        "1802271",
        "1805292",
        "1808347",
        "1812095",
        "BD+023375",
        "BD+174708",
        "BD+210607",
        "BD+262606",
        "BD+292091",
        "BD+541216",
        "BD+601753",
        "FEIGE110",
        "FEIGE34",
        "G191B2B",
        "GD153",
        "GD71",
        "GJ754.1A",
        "GRW+705824",
        "HD37725",
        "HD55677",
        "HD106252",
        "HD116405",
        "HD142331",
        "HD180609",
        "HD209458",
        "HS2027+0651",
        "HZ21",
        "HZ4",
        "HZ44",
        "KF06T2",
        "KF08T3",
        "LDS749B",
        "P177D",
        "P330E",
        "SDSSJ151421",
        "SF1615+001A",
        "SNAP-2",
        "WD1026+453",
        "WD1327-083",
        "WD1657+343",
        "WD2341+322",
    }
)


# Only spellings demonstrated by the supplied historical Panacea list are
# included.  The mapping is case-insensitive when it is applied, but the
# published keys retain the historical spelling for inspection.
STANDARD_STAR_ALIASES = {
    "HZ_44": "HZ44",
    "HZ_21": "HZ21",
    "HZ_4": "HZ4",
    "FEIGE_34": "FEIGE34",
    "FEIGE_110": "FEIGE110",
    "GRW+70_5824": "GRW+705824",
    "BD+26+2606": "BD+262606",
    "BD_+17_4708": "BD+174708",
    "BD_+26_2606": "BD+262606",
}

_STANDARD_STAR_ALIASES_CASEFOLDED: Mapping[str, str] = {
    key.casefold(): value for key, value in STANDARD_STAR_ALIASES.items()
}


def normalize_standard_star_name(value: object) -> str | None:
    """Normalize one explicit standard-star spelling to a canonical name.

    The operation strips surrounding whitespace, applies a case-insensitive
    lookup in :data:`STANDARD_STAR_ALIASES`, and otherwise returns the
    spelling unchanged.  It deliberately does not remove punctuation or use
    substring matching.
    """

    if value in (None, ""):
        return None
    name = str(value).strip()
    if not name:
        return None
    return _STANDARD_STAR_ALIASES_CASEFOLDED.get(name.casefold(), name)


DEFAULT_STANDARD_STAR_CATALOG_SOURCE = "HET/Hydra canonical standard-star table"


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
    def default(cls) -> "StandardStarCatalog":
        """Return the static HET/Hydra default catalog."""

        return cls.from_names(
            STANDARD_STAR_NAMES,
            source=DEFAULT_STANDARD_STAR_CATALOG_SOURCE,
        )

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
        normalized = normalize_standard_star_name(target)
        return normalized is not None and normalized.casefold() in self.names


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

    catalog = standard_catalog or StandardStarCatalog.default()
    object_intent = parse_object_intent(object_name)
    standard_star: bool | None = None
    standard_target = normalize_standard_star_name(object_intent.target)
    if normalized_types == {"sci"} and object_intent.target is not None:
        standard_star = catalog.contains(standard_target)
    return ExposureClassification(
        quicklook_kind=quicklook_kind,
        calibration_source=source,
        applicable_ifu_slots=applicable,
        standard_target=standard_target,
        standard_star=standard_star,
        standard_catalog_available=catalog.available,
        standard_catalog_source=catalog.source,
    )

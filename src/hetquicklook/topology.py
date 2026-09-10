"""Small boundaries for authoritative detector and fiber configuration.

The loaders in this module describe where configuration comes from and how
the supplied pipeline conventions select rows. They do not provide fallback
coordinates when a required resource is absent.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import date, datetime
from importlib import resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Iterator

import numpy as np

from .instrument import PhysicalAmplifierIdentity


@dataclass(frozen=True)
class TopologyReference:
    """Provenance for one loaded topology resource."""

    kind: str
    path: Path
    identifier: str | None = None


class ConfigurationResourceError(FileNotFoundError):
    """Raised when an authoritative topology resource is unavailable."""


@contextmanager
def _materialized_resource(resource: Path | Traversable) -> Iterator[Path]:
    """Expose a package resource as a filesystem path while it is read."""

    if isinstance(resource, Path):
        with nullcontext(resource) as path:
            yield path
        return
    with resources.as_file(resource) as path:
        yield path


def _packaged_resource(*parts: str) -> Traversable:
    return resources.files("hetquicklook.resources").joinpath(*parts)


def _explicit_resource(
    root: Path, *parts: str, description: str = "topology resource"
) -> Path:
    candidates = (
        root.joinpath(*parts),
        root.joinpath(parts[-1]),
        root.joinpath("virus", parts[-1]),
        root.joinpath("lrs2", parts[-1]),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ConfigurationResourceError(
        f"Missing {description} {parts[-1]} under {root}"
    )


_VIRUS_AMP_SLICES = {
    "LU": (0, 112),
    "LL": (112, 224),
    "RL": (224, 336),
    "RU": (336, 448),
}
_VIRUS_RIGHT_REVERSALS = frozenset({"003", "004", "005", "008"})
_VIRUS_COORDINATE_SWAPS = {
    "007": (37, 38),
    "025": (208, 213),
    "030": (445, 446),
    "038": (302, 303),
    "041": (251, 252),
}
_VIRUS_ALTERNATE_SOURCES = {"004": "IFUcen_HETDEX_reverse_R.txt"}


class VirusTopologyLoader:
    """Resolve VIRUS fiber geometry, trace, and focal-plane resources."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        trace_root: str | Path | None = None,
        resource_root: str | Path | None = None,
    ):
        """Create a loader with package defaults and optional filesystem roots.

        ``root`` remains a compatibility shorthand for using one filesystem
        root for both static resources and external dated traces. New callers
        can supply ``trace_root`` independently; static resources then come
        from the installed package unless ``resource_root`` is also supplied.
        """

        if root is not None:
            root_path = Path(root).expanduser()
            trace_root = root_path if trace_root is None else trace_root
            resource_root = root_path if resource_root is None else resource_root
        self.trace_root = (
            Path(trace_root).expanduser() if trace_root is not None else None
        )
        self.resource_root = (
            Path(resource_root).expanduser() if resource_root is not None else None
        )

    def _static_resource(
        self, name: str, *, description: str = "VIRUS fiber-position resource"
    ) -> Path | Traversable:
        if self.resource_root is not None:
            return _explicit_resource(
                self.resource_root,
                "IFUcen_files",
                name,
                description=description,
            )
        packaged = _packaged_resource("virus", name)
        if not packaged.is_file():
            raise ConfigurationResourceError(f"Missing {description} {name}")
        return packaged

    def fiber_positions(
        self,
        ifuid: str,
        amplifier: str,
    ) -> tuple[np.ndarray, TopologyReference]:
        """Load the corrected 112-position table for one VIRUS amplifier."""

        ifuid_text = str(ifuid).strip()
        if not ifuid_text.isdigit() or len(ifuid_text) > 3:
            raise ValueError(f"Invalid VIRUS IFUID {ifuid!r}; expected one to three digits")
        normalized_ifuid = ifuid_text.zfill(3)
        normalized_amp = str(amplifier).strip().upper()
        try:
            start, stop = _VIRUS_AMP_SLICES[normalized_amp]
        except KeyError as error:
            raise ValueError(f"Unknown VIRUS amplifier: {amplifier!r}") from error
        filename = _VIRUS_ALTERNATE_SOURCES.get(
            normalized_ifuid, "IFUcen_HETDEX.txt"
        )
        resource = self._static_resource(filename)
        with _materialized_resource(resource) as path:
            try:
                table = np.loadtxt(path, usecols=(0, 1, 2, 4), skiprows=30)
            except (OSError, ValueError) as error:
                raise ConfigurationResourceError(
                    f"Cannot read VIRUS fiber-position resource: {path}"
                ) from error
        table = np.atleast_2d(np.asarray(table, dtype=float))
        if table.shape != (448, 4):
            raise ValueError(
                f"VIRUS fiber-position resource must provide 448 rows and 4 selected columns: {path}"
            )
        positions = table[:, (1, 3)].copy()
        if normalized_ifuid in _VIRUS_RIGHT_REVERSALS:
            positions[224:448] = positions[224:448][::-1]
        swap = _VIRUS_COORDINATE_SWAPS.get(normalized_ifuid)
        if swap is not None:
            first, second = swap
            positions[[first, second]] = positions[[second, first]]
        selected = positions[start:stop]
        selected = selected[::-1]
        return selected, TopologyReference(
            "virus_fiber_positions", Path(path), normalized_ifuid
        )

    def resolve_trace_reference(
        self,
        identity: PhysicalAmplifierIdentity,
        at: date | datetime | str | None = None,
    ) -> tuple[np.ndarray, TopologyReference]:
        """Resolve the nearest dated VIRUS trace file for a full identity."""

        if identity.specid is None or identity.ifuid is None:
            raise ValueError("Trace lookup requires SPECID and IFUID")
        if self.trace_root is None:
            raise ConfigurationResourceError(
                "No external VIRUS/LRS2 trace root configured; supply trace_root"
            )
        base = self.trace_root / "Fiber_Locations"
        if not base.is_dir():
            raise ConfigurationResourceError(f"Missing VIRUS trace directory: {base}")
        if at is None:
            target = date.today()
        elif isinstance(at, datetime):
            target = at.date()
        elif isinstance(at, date):
            target = at
        else:
            text = str(at).strip()
            try:
                target = datetime.strptime(text[:8], "%Y%m%d").date()
            except ValueError:
                target = date.fromisoformat(text[:10])
        candidates: list[tuple[int, Path]] = []
        for directory in base.iterdir():
            if not directory.is_dir():
                continue
            try:
                directory_date = datetime.strptime(directory.name, "%Y%m%d").date()
            except ValueError:
                continue
            filename = (
                f"fiber_loc_{identity.specid.zfill(3)}_"
                f"{identity.ifu_slot.zfill(3)}_{identity.ifuid.zfill(3)}_"
                f"{identity.amplifier}.txt"
            )
            path = directory / filename
            if path.is_file():
                candidates.append((abs((directory_date - target).days), path))
        if not candidates:
            raise ConfigurationResourceError(
                f"No dated VIRUS trace resource for {identity.key or identity.ifu_slot} under {base}"
            )
        _, path = min(candidates, key=lambda item: (item[0], str(item[1])))
        try:
            traces = np.loadtxt(path)
        except (OSError, ValueError) as error:
            raise ConfigurationResourceError(f"Cannot read VIRUS trace resource: {path}") from error
        return np.asarray(traces), TopologyReference("virus_trace", path, path.parent.name)

    def resolve_fplane(
        self,
        path: str | Path | None = None,
    ) -> tuple[dict[str, tuple[float, float]], TopologyReference]:
        """Load IFUSLOT focal-plane coordinates from ``fplaneall.txt``."""

        resource = (
            Path(path)
            if path is not None
            else self._static_resource(
                "fplaneall.txt", description="VIRUS focal-plane resource"
            )
        )
        if isinstance(resource, Path) and not resource.is_file():
            raise ConfigurationResourceError(
                f"Missing VIRUS focal-plane resource: {resource}"
            )
        with _materialized_resource(resource) as selected:
            result: dict[str, tuple[float, float]] = {}
            try:
                with selected.open() as stream:
                    for line in stream:
                        fields = line.split()
                        if not fields or fields[0].startswith("#") or len(fields) < 3:
                            continue
                        result[fields[0].zfill(3)] = (float(fields[1]), float(fields[2]))
            except OSError as error:
                raise ConfigurationResourceError(
                    f"Cannot read VIRUS focal-plane resource: {selected}"
                ) from error
        if "000" not in result or len(result) < 70:
            raise ValueError(
                f"VIRUS focal-plane resource is incomplete; expected slot 000 and at least 70 slots: {selected}"
            )
        return result, TopologyReference("virus_focal_plane", Path(selected))


_LRS2_FIBER_POSITION_FILES = {
    "UV": "LRS2_B_UV_mapping.txt",
    "ORANGE": "LRS2_B_OR_mapping.txt",
    "RED": "LRS2_R_NR_mapping.txt",
    "FAR-RED": "LRS2_R_FR_mapping.txt",
    "FAR RED": "LRS2_R_FR_mapping.txt",
}
_LRS2_AMP_SLICES = {
    "LL": (140, 280),
    "LU": (0, 140),
    "RL": (0, 140),
    "RU": (140, 280),
}


class LRS2FiberPositionLoader:
    """Load the four channel-specific LRS2 fiber mapping resources."""

    def __init__(
        self,
        root: str | Path | None = None,
        *,
        resource_root: str | Path | None = None,
    ):
        self.resource_root = Path(
            resource_root if resource_root is not None else root
        ).expanduser() if (resource_root is not None or root is not None) else None

    def _resource(self, channel: str) -> Path | Traversable:
        key = str(channel).strip().upper()
        try:
            filename = _LRS2_FIBER_POSITION_FILES[key]
        except KeyError as error:
            raise ValueError(f"Unknown LRS2 channel: {channel!r}") from error
        if self.resource_root is not None:
            return _explicit_resource(
                self.resource_root,
                "lrs2",
                filename,
                description="LRS2 fiber-position resource",
            )
        packaged = _packaged_resource("lrs2", filename)
        if not packaged.is_file():
            raise ConfigurationResourceError(f"Missing LRS2 fiber-position resource: {filename}")
        return packaged

    def resource_path(self, channel: str) -> Path | Traversable:
        """Return the configured resource handle for inspection."""

        return self._resource(channel)

    def fiber_positions(
        self,
        channel: str,
        amplifier: str,
        *,
        coordinate_columns: tuple[int, int] = (1, 2),
        skiprows: int = 5,
    ) -> tuple[np.ndarray, TopologyReference]:
        """Load and order one 140-fiber LRS2 amplifier mapping.

        The packaged mapping files establish a five-line header followed by
        six-column rows. Their physical x/y coordinates are columns 1 and 2;
        ``coordinate_columns`` remains overrideable for expert file variants.
        """

        if len(coordinate_columns) != 2:
            raise ValueError("LRS2 coordinate_columns must contain exactly two columns")
        resource = self._resource(channel)
        amp = str(amplifier).strip().upper()
        try:
            start, stop = _LRS2_AMP_SLICES[amp]
        except KeyError as error:
            raise ValueError(f"Unknown LRS2 amplifier: {amplifier!r}") from error
        with _materialized_resource(resource) as path:
            try:
                table = np.atleast_2d(
                    np.loadtxt(path, usecols=coordinate_columns, skiprows=skiprows)
                )
            except (OSError, ValueError) as error:
                raise ConfigurationResourceError(
                    f"Cannot read LRS2 fiber-position resource: {path}"
                ) from error
            if table.shape != (280, 2):
                raise ValueError(
                    f"LRS2 fiber-position resource must provide 280 coordinate rows: {path}"
                )
            selected = table[start:stop].copy()
            return selected[::-1], TopologyReference(
                "lrs2_fiber_positions", Path(path), str(channel)
            )


def virus_orientation(amplifier: str, raw_ampname: str | None = None) -> tuple[bool, bool]:
    """Return supported detector flips without applying data reduction."""

    amp = str(amplifier).strip().upper()
    flip_x = amp in {"LU", "RL"}
    flip_y = amp in {"LU", "RL"}
    if raw_ampname is not None and str(raw_ampname).replace(" ", "").upper() in {"LR", "UL"}:
        flip_x = not flip_x
    return flip_x, flip_y

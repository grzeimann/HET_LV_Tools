"""Small raw FITS loading boundary for tar-backed observations.

The implementation intentionally keeps discovery and physical loading
separate. It follows the VIRUSFlow contract that the primary HDU contains the
detector array and the relevant raw header, while leaving caching and indexed
tar access to the larger pipeline.
"""

from __future__ import annotations

from contextlib import contextmanager
import io
from pathlib import Path
import tarfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from astropy.io import fits

from .discovery import ArchiveMember, RawFrameIdentity


@dataclass(frozen=True)
class RawFrameData:
    """Detector data, primary header, and raw-member provenance."""

    data: np.ndarray
    header: Mapping[str, Any]
    path: str
    tar_member: str | None = None
    outer_tar_member: str | None = None
    identity: RawFrameIdentity | None = None

    @property
    def archive_path(self) -> Path:
        """Return the physical outer archive path."""

        return Path(self.path)

    @property
    def member_name(self) -> str | None:
        """Return the FITS member name."""

        return self.tar_member

    @property
    def provenance(self) -> tuple[Path, str | None, str | None]:
        """Return ``(outer archive, nested archive member, FITS member)``."""

        return (self.archive_path, self.outer_tar_member, self.tar_member)


class RawFrameLoader:
    """Load primary-HDU FITS data from direct or nested tar members."""

    @contextmanager
    def _open_payload(
        self,
        path: Path,
        tar_member: str | None,
        outer_tar_member: str | None,
    ) -> Iterator[Any]:
        if tar_member is None:
            with path.open("rb") as stream:
                yield stream
            return

        with tarfile.open(path, mode="r:*") as outer:
            if outer_tar_member is None:
                member = outer.getmember(tar_member)
                stream = outer.extractfile(member)
                if stream is None:
                    raise FileNotFoundError(f"Cannot extract {tar_member} from {path}")
                try:
                    yield stream
                finally:
                    stream.close()
                return

            nested = outer.getmember(outer_tar_member)
            nested_stream = outer.extractfile(nested)
            if nested_stream is None:
                raise FileNotFoundError(
                    f"Cannot extract {outer_tar_member} from {path}"
                )
            try:
                with tarfile.open(fileobj=nested_stream, mode="r:*") as inner:
                    member = inner.getmember(tar_member)
                    stream = inner.extractfile(member)
                    if stream is None:
                        raise FileNotFoundError(
                            f"Cannot extract {tar_member} from {outer_tar_member}"
                        )
                    try:
                        yield stream
                    finally:
                        stream.close()
            finally:
                nested_stream.close()

    @staticmethod
    def _source_details(
        source: ArchiveMember | str | Path,
        tar_member: str | None,
        outer_tar_member: str | None,
    ) -> tuple[Path, str | None, str | None, RawFrameIdentity | None]:
        if isinstance(source, ArchiveMember):
            return (
                source.archive_path,
                source.member_name,
                source.outer_tar_member,
                source.identity,
            )
        return Path(source), tar_member, outer_tar_member, None

    def read_header(
        self,
        source: ArchiveMember | str | Path,
        tar_member: str | None = None,
        *,
        outer_tar_member: str | None = None,
    ) -> dict[str, Any]:
        """Read only the primary FITS header for a source."""

        path, member_name, nested_member, _ = self._source_details(
            source, tar_member, outer_tar_member
        )
        with self._open_payload(path, member_name, nested_member) as stream:
            with fits.open(stream, memmap=False, lazy_load_hdus=True) as hdul:
                return dict(hdul[0].header)

    def load(
        self,
        source: ArchiveMember | str | Path,
        tar_member: str | None = None,
        *,
        outer_tar_member: str | None = None,
    ) -> RawFrameData:
        """Load primary-HDU detector data and header for one raw FITS source.

        ``source`` may be an :class:`ArchiveMember` or a physical FITS path.
        The latter form also accepts the VIRUSFlow-style ``tar_member`` and
        ``outer_tar_member`` arguments.
        """

        path, member_name, nested_member, identity = self._source_details(
            source, tar_member, outer_tar_member
        )
        if member_name is None:
            fits_source: str | io.BufferedIOBase = str(path)
            with fits.open(fits_source, memmap=False) as hdul:
                header = dict(hdul[0].header)
                data = np.array(hdul[0].data, copy=True)
        else:
            with self._open_payload(path, member_name, nested_member) as stream:
                with fits.open(stream, memmap=False) as hdul:
                    header = dict(hdul[0].header)
                    data = np.array(hdul[0].data, copy=True)
        if data.ndim == 0:
            raise ValueError(f"Primary HDU has no detector array: {path}::{member_name}")
        return RawFrameData(
            data=data,
            header=header,
            path=str(path),
            tar_member=member_name,
            outer_tar_member=nested_member,
            identity=identity,
        )

    def load_member(self, member: ArchiveMember) -> RawFrameData:
        """Load an inventoried archive member."""

        return self.load(member)


def load_frame(member: ArchiveMember) -> RawFrameData:
    """Load one inventoried FITS member with a short functional API."""

    return RawFrameLoader().load(member)


def read_fits_header(member: ArchiveMember) -> dict[str, Any]:
    """Read one inventoried member's primary header without loading its array."""

    return RawFrameLoader().read_header(member)

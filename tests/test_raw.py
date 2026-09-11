import io
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
import tarfile
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from hetquicklook.discovery import (
    DiscoveredObservation,
    inventory_members,
    parse_member_identity,
)
from hetquicklook.raw import RawFrameLoader


def _fits_bytes(data: np.ndarray, **header_values: object) -> bytes:
    stream = io.BytesIO()
    hdu = fits.PrimaryHDU(data=data)
    for key, value in header_values.items():
        hdu.header[key] = value
    hdu.writeto(stream)
    return stream.getvalue()


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def _observation(path: Path, *, outer_tar_member: str | None = None) -> DiscoveredObservation:
    return DiscoveredObservation(
        archive_path=path,
        date=date(2026, 9, 10),
        instrument="virus",
        outer_tar_member=outer_tar_member,
    )


def test_parser_uses_basename_and_preserves_encoded_identity() -> None:
    identity = parse_member_identity(
        "optional/prefix/20260511T035810.4_074LL_cmp.fits"
    )

    assert identity is not None
    assert identity.exposure_id == "20260511T035810.4"
    assert identity.amplifier_token == "074LL"
    assert identity.ifu_slot == "074"
    assert identity.amplifier == "LL"
    assert identity.frame_type == "cmp"


def test_inventory_is_sorted_and_keeps_malformed_members_observable(tmp_path: Path) -> None:
    archive_path = tmp_path / "observation.tar"
    with tarfile.open(archive_path, mode="w") as archive:
        _add_member(
            archive,
            "z/20260511T035810.4_074LL_cmp.fits",
            _fits_bytes(np.ones((2, 3), dtype=np.uint16)),
        )
        _add_member(archive, "a/not-a-frame.fits", b"malformed")
        _add_member(archive, "README.txt", b"metadata")

    members = inventory_members(_observation(archive_path))

    assert [member.member_name for member in members] == [
        "README.txt",
        "a/not-a-frame.fits",
        "z/20260511T035810.4_074LL_cmp.fits",
    ]
    malformed = members[1]
    assert malformed.identity is None
    assert malformed.parse_error is not None
    assert members[2].size > 0


def test_loader_reads_primary_hdu_and_retains_provenance(tmp_path: Path) -> None:
    archive_path = tmp_path / "observation.tar"
    data = np.array([[1, 2], [3, 4]], dtype=np.uint16)
    member_name = "prefix/20260511T035810.4_074LL_cmp.fits"
    with tarfile.open(archive_path, mode="w") as archive:
        _add_member(archive, member_name, _fits_bytes(data, OBJECT="skyflat"))

    member = inventory_members(_observation(archive_path))[0]
    loaded = RawFrameLoader().load(member)

    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.header["OBJECT"] == "skyflat"
    assert loaded.path == str(archive_path)
    assert loaded.tar_member == member_name
    assert loaded.outer_tar_member is None
    assert loaded.identity == member.identity


def test_loader_reuses_direct_archive_and_is_safe_for_threaded_reads(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "observation.tar"
    names = (
        "20260511T035810.4_074LL_cmp.fits",
        "20260511T035810.4_074LU_cmp.fits",
    )
    with tarfile.open(archive_path, mode="w") as archive:
        for value, name in enumerate(names, start=1):
            _add_member(archive, name, _fits_bytes(np.array([[value]])))
    members = inventory_members(_observation(archive_path))

    loader = RawFrameLoader()
    with patch("hetquicklook.raw.tarfile.open", wraps=tarfile.open) as open_tar:
        with ThreadPoolExecutor(max_workers=2) as executor:
            loaded = tuple(executor.map(loader.load, members))
        assert open_tar.call_count == 1
    np.testing.assert_array_equal(loaded[0].data, np.array([[1]]))
    np.testing.assert_array_equal(loaded[1].data, np.array([[2]]))
    loader.close()
    assert loader._archives == {}


def test_loader_reuses_nested_outer_and_inner_archives(tmp_path: Path) -> None:
    inner_bytes = io.BytesIO()
    names = (
        "virus0000001/exp01/virus/20260511T035810.4_074LL_cmp.fits",
        "virus0000001/exp01/virus/20260511T035810.4_074LU_cmp.fits",
    )
    with tarfile.open(fileobj=inner_bytes, mode="w") as inner:
        for value, name in enumerate(names, start=1):
            _add_member(inner, name, _fits_bytes(np.array([[value]])))
    outer_path = tmp_path / "20260511.tar"
    outer_member = "virus/virus0000001.tar"
    with tarfile.open(outer_path, mode="w") as outer:
        _add_member(outer, outer_member, inner_bytes.getvalue())

    members = inventory_members(_observation(outer_path, outer_tar_member=outer_member))
    loader = RawFrameLoader()
    with patch("hetquicklook.raw.tarfile.open", wraps=tarfile.open) as open_tar:
        loader.load(members[0])
        loader.load(members[1])
        assert open_tar.call_count == 2
    loader.close()
    assert loader._archives == {}
    assert loader._nested_archives == {}


def test_loader_supports_nested_date_tar_layout(tmp_path: Path) -> None:
    inner_bytes = io.BytesIO()
    with tarfile.open(fileobj=inner_bytes, mode="w") as inner:
        _add_member(
            inner,
            "virus0000001/exp01/virus/20260511T035810.4_074LL_cmp.fits",
            _fits_bytes(np.array([[7]], dtype=np.uint16)),
        )
    outer_path = tmp_path / "20260511.tar"
    outer_member = "virus/virus0000001.tar"
    with tarfile.open(outer_path, mode="w") as outer:
        _add_member(outer, outer_member, inner_bytes.getvalue())

    discovered = _observation(outer_path, outer_tar_member=outer_member)
    members = inventory_members(discovered)
    loaded = RawFrameLoader().load(members[0])

    assert discovered.observation_id == "virus0000001"
    assert members[0].outer_tar_member == outer_member
    np.testing.assert_array_equal(loaded.data, np.array([[7]], dtype=np.uint16))
    assert loaded.provenance == (outer_path, outer_member, members[0].member_name)

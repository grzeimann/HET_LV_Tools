import io
from pathlib import Path
import tarfile

import numpy as np
import pytest
from astropy.io import fits

from hetquicklook import (
    QuicklookConfig,
    discover_observations,
    load_observation,
    load_selected_exposure,
)
from hetquicklook.raw import RawFrameLoader
from hetquicklook.observation import summarize_observation


def _fits_bytes(data: np.ndarray, object_name: str) -> bytes:
    stream = io.BytesIO()
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["OBJECT"] = object_name
    hdu.writeto(stream)
    return stream.getvalue()


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_observation_represents_partial_components_and_groups_exposures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    archive_path = root / "20260910" / "lrs2" / "lrs20000001.tar"
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(archive_path, mode="w") as archive:
        _add_member(
            archive,
            "lrs20000001/exp01/lrs2/20260910T010101.1_056LL_twi.fits",
            _fits_bytes(np.array([[1]], dtype=np.uint16), "skyflat"),
        )
        _add_member(
            archive,
            "lrs20000001/exp02/lrs2/20260910T010202.2_056LL_twi.fits",
            _fits_bytes(np.array([[2]], dtype=np.uint16), "skyflat"),
        )

    discovered = discover_observations(
        QuicklookConfig(root), instrument="lrs2", date="20260910"
    )[0]
    observation = load_observation(discovered)

    assert len(observation.frames) == 2
    assert observation.metadata.object_name is None
    assert observation.exposure_for("20260910T010101.1").metadata.object_name == "skyflat"
    assert observation.exposure_for("20260910T010202.2").metadata.object_name == "skyflat"
    assert observation.metadata.exposure_time_s is None
    assert observation.metadata.requested_ra_deg is None
    assert set(observation.group_by_exposure()) == {
        "20260910T010101.1",
        "20260910T010202.2",
    }
    assert observation.frames_for(frame_type="twi")[0].identity is not None
    loaded = observation.load_frames(exposure_id="20260910T010202.2")
    np.testing.assert_array_equal(loaded[0].data, np.array([[2]], dtype=np.uint16))


@pytest.mark.parametrize(
    ("instrument", "observation_id", "amplifier_token"),
    (
        ("virus", "virus0000001", "074LL"),
        ("lrs2", "lrs20000001", "056LL"),
    ),
)
def test_observation_loads_het_directory_layout(
    tmp_path: Path,
    instrument: str,
    observation_id: str,
    amplifier_token: str,
) -> None:
    root = tmp_path / "root"
    observation_path = root / "20260910" / instrument / observation_id
    member_path = (
        observation_path
        / "exp01"
        / instrument
        / f"20260910T010101.1_{amplifier_token}_twi.fits"
    )
    member_path.parent.mkdir(parents=True)
    data = np.array([[3, 4]], dtype=np.uint16)
    member_path.write_bytes(_fits_bytes(data, "skyflat"))

    discovered = discover_observations(
        QuicklookConfig(root), instrument=instrument, date="20260910"
    )[0]
    observation = load_observation(discovered)

    assert discovered.storage_backend == "directory"
    assert [member.member_name for member in observation.members] == [
        f"exp01/{instrument}/20260910T010101.1_{amplifier_token}_twi.fits"
    ]
    exposure = observation.exposure_for("20260910T010101.1")
    loaded = exposure.load_frames()[0]
    np.testing.assert_array_equal(loaded.data, data)
    assert loaded.path == str(observation_path)
    assert loaded.tar_member == (
        f"exp01/{instrument}/20260910T010101.1_{amplifier_token}_twi.fits"
    )


def test_selected_exposure_reads_only_one_exposure_and_compact_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    observation_path = root / "20260910" / "virus" / "virus0000001"
    first = observation_path / "exp01" / "virus" / "20260910T010101.1_074LL_twi.fits"
    second = observation_path / "exp02" / "virus" / "20260910T010202.2_074LL_twi.fits"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(_fits_bytes(np.array([[1]], dtype=np.uint16), "first"))
    second.write_bytes(_fits_bytes(np.array([[2]], dtype=np.uint16), "second"))

    discovered = discover_observations(
        QuicklookConfig(root), instrument="virus", date="20260910"
    )[0]
    read_members: list[tuple[str, ...]] = []
    original_read_headers = RawFrameLoader.read_headers

    def record_read_headers(loader, members):
        members = tuple(members)
        read_members.append(tuple(member.member_name for member in members))
        return original_read_headers(loader, members)

    monkeypatch.setattr(RawFrameLoader, "read_headers", record_read_headers)
    observation = load_selected_exposure(discovered, "20260910T010202.2")

    assert observation.exposure_ids == ("20260910T010202.2",)
    assert [member.member_name for member in observation.members] == [
        "exp02/virus/20260910T010202.2_074LL_twi.fits"
    ]
    assert read_members == [
        ("exp02/virus/20260910T010202.2_074LL_twi.fits",)
    ]

    first_observation = load_selected_exposure(discovered)
    assert first_observation.exposure_ids == ("20260910T010101.1",)
    assert read_members == [
        ("exp02/virus/20260910T010202.2_074LL_twi.fits",),
        ("exp01/virus/20260910T010101.1_074LL_twi.fits",),
    ]


def test_observation_summary_reads_one_representative_header_per_exposure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    observation_path = root / "20260910" / "virus" / "virus0000001"
    members = (
        observation_path / "exp01" / "virus" / "20260910T010101.1_074LL_sci.fits",
        observation_path / "exp01" / "virus" / "20260910T010101.1_074LU_sci.fits",
        observation_path / "exp02" / "virus" / "20260910T010202.2_074LL_sci.fits",
        observation_path / "exp02" / "virus" / "20260910T010202.2_074LU_sci.fits",
    )
    for index, member in enumerate(members, start=1):
        member.parent.mkdir(parents=True, exist_ok=True)
        member.write_bytes(_fits_bytes(np.array([[index]], dtype=np.uint16), f"target-{index}"))

    discovered = discover_observations(
        QuicklookConfig(root), instrument="virus", date="20260910"
    )[0]
    read_members: list[tuple[str, ...]] = []
    original_read_headers = RawFrameLoader.read_headers

    def record_read_headers(loader, requested_members):
        requested_members = tuple(requested_members)
        read_members.append(
            tuple(member.member_name for member in requested_members)
        )
        return original_read_headers(loader, requested_members)

    monkeypatch.setattr(RawFrameLoader, "read_headers", record_read_headers)
    summaries = summarize_observation(discovered)

    assert [summary.exposure_id for summary in summaries] == [
        "20260910T010101.1",
        "20260910T010202.2",
    ]
    assert [summary.frame_count for summary in summaries] == [2, 2]
    assert [summary.metadata.object_name for summary in summaries] == [
        "target-1",
        "target-3",
    ]
    assert read_members == [
        (
            "exp01/virus/20260910T010101.1_074LL_sci.fits",
            "exp02/virus/20260910T010202.2_074LL_sci.fits",
        )
    ]

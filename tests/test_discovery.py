from pathlib import Path
import io
import tarfile

import hetquicklook.discovery as discovery
import pytest
from hetquicklook.config import QuicklookConfig
from hetquicklook.discovery import discover_observations


def test_discovery_reports_archives_without_completeness_checks(tmp_path: Path) -> None:
    virus_dir = tmp_path / "2026-09-10" / "virus"
    lrs2_dir = tmp_path / "20260911" / "lrs2"
    virus_dir.mkdir(parents=True)
    lrs2_dir.mkdir(parents=True)
    (virus_dir / "virus-001.tar.gz").touch()
    (virus_dir / "README.txt").touch()
    (lrs2_dir / "lrs2-002.tgz").touch()

    observations = discover_observations(QuicklookConfig(tmp_path))

    assert [item.observation_id for item in observations] == ["virus-001", "lrs2-002"]
    assert observations[0].instrument.value == "virus"
    assert observations[1].date.isoformat() == "2026-09-11"


def test_discovery_can_filter_by_date_and_instrument(tmp_path: Path) -> None:
    path = tmp_path / "20260910" / "VIRUS"
    path.mkdir(parents=True)
    archive = path / "one.tar"
    archive.touch()

    observations = discover_observations(
        QuicklookConfig(tmp_path), instrument="virus", date="2026-09-10"
    )

    assert observations[0].archive_path == archive


def test_filtered_discovery_probes_only_the_requested_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = tmp_path / "20260910" / "virus"
    requested.mkdir(parents=True)
    archive = requested / "one.tar"
    archive.touch()
    (tmp_path / "20200101" / "gc1").mkdir(parents=True)

    def fail_root_scan(root: Path):
        raise AssertionError("date-filtered discovery must not list all root dates")

    monkeypatch.setattr(discovery, "_date_directories", fail_root_scan)
    monkeypatch.setattr(discovery, "_date_archives", fail_root_scan)
    observations = discover_observations(
        QuicklookConfig(tmp_path), instrument="virus", date="20260910"
    )

    assert [item.archive_path for item in observations] == [archive]


def test_discovery_reports_nested_corral_virus_observations(tmp_path: Path) -> None:
    date_archive = tmp_path / "20260910.tar"
    inner = io.BytesIO()
    with tarfile.open(fileobj=inner, mode="w"):
        pass
    with tarfile.open(date_archive, mode="w") as archive:
        info = tarfile.TarInfo("virus/virus0000001.tar")
        info.size = len(inner.getvalue())
        archive.addfile(info, io.BytesIO(inner.getvalue()))

    observations = discover_observations(QuicklookConfig(tmp_path), instrument="virus")

    assert len(observations) == 1
    assert observations[0].observation_id == "virus0000001"
    assert observations[0].outer_tar_member == "virus/virus0000001.tar"


def test_discovery_reports_het_observation_directories(tmp_path: Path) -> None:
    virus = tmp_path / "20260910" / "virus" / "virus0000001"
    lrs2 = tmp_path / "20260910" / "lrs2" / "lrs20000001"
    (virus / "exp01" / "virus").mkdir(parents=True)
    (lrs2 / "exp01" / "lrs2").mkdir(parents=True)

    observations = discover_observations(
        QuicklookConfig(tmp_path), date="20260910"
    )

    assert [(item.instrument.value, item.observation_id) for item in observations] == [
        ("lrs2", "lrs20000001"),
        ("virus", "virus0000001"),
    ]
    assert all(item.storage_backend == "directory" for item in observations)

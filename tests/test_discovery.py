from pathlib import Path
import io
import tarfile

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

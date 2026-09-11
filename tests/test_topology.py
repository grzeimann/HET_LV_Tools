from datetime import date
from pathlib import Path

import numpy as np
import pytest

from hetquicklook import (
    ConfigurationResourceError,
    LRS2FiberPositionLoader,
    PhysicalAmplifierIdentity,
    VirusTopologyLoader,
    virus_orientation,
)


def _virus_geometry(path: Path) -> None:
    path.parent.mkdir(parents=True)
    with path.open("w") as stream:
        stream.write("# header\n" * 30)
        for row in range(448):
            stream.write(f"{row} {row + 1} 0 {row + 3} {row + 10}\n")


def test_virus_fiber_positions_apply_slice_and_extracted_order(tmp_path: Path) -> None:
    _virus_geometry(tmp_path / "IFUcen_files" / "IFUcen_HETDEX.txt")
    positions, reference = VirusTopologyLoader(tmp_path).fiber_positions("001", "LU")

    assert positions.shape == (112, 2)
    np.testing.assert_array_equal(positions[0], [112, 0])
    np.testing.assert_array_equal(positions[-1], [1, 0])
    assert reference.path.name == "IFUcen_HETDEX.txt"


def test_virus_trace_resolution_uses_nearest_dated_reference(tmp_path: Path) -> None:
    first = tmp_path / "Fiber_Locations" / "20260101"
    second = tmp_path / "Fiber_Locations" / "20260301"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    filename = "fiber_loc_412_074_043_LL.txt"
    np.savetxt(first / filename, [[1, 2]])
    np.savetxt(second / filename, [[3, 4]])
    identity = PhysicalAmplifierIdentity(
        instrument="virus", ifu_slot="074", amplifier="LL", ifuid="043", specid="412", controller="S/N 0021"
    )

    values, reference = VirusTopologyLoader(tmp_path).resolve_trace_reference(
        identity, date(2026, 2, 1)
    )

    np.testing.assert_array_equal(values, [3, 4])
    assert reference.path.parent.name == "20260301"


def test_missing_topology_resource_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationResourceError, match="fiber-position"):
        VirusTopologyLoader(tmp_path).fiber_positions("001", "LL")
    with pytest.raises(ConfigurationResourceError, match="fiber-position"):
        LRS2FiberPositionLoader(tmp_path).resource_path("UV")
    with pytest.raises(ConfigurationResourceError, match="focal-plane"):
        VirusTopologyLoader(tmp_path).resolve_fplane()


def test_supported_virus_orientation_is_only_a_transform_description() -> None:
    assert virus_orientation("LU") == (True, True)
    assert virus_orientation("RL", " LR ") == (False, True)


def test_lrs2_mapping_uses_explicit_columns_and_amp_order(tmp_path: Path) -> None:
    path = tmp_path / "LRS2_B_UV_mapping.txt"
    with path.open("w") as stream:
        stream.write("# header\n" * 5)
        np.savetxt(stream, np.column_stack((np.arange(280), np.arange(280) + 100)))

    positions, _ = LRS2FiberPositionLoader(tmp_path).fiber_positions(
        "UV", "LL", coordinate_columns=(0, 1)
    )

    assert positions.shape == (140, 2)
    np.testing.assert_array_equal(positions[0], [279, 379])
    np.testing.assert_array_equal(positions[-1], [140, 240])


def test_packaged_static_resources_are_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    virus = VirusTopologyLoader()
    fplane, _ = virus.resolve_fplane()
    normal, normal_reference = virus.fiber_positions("001", "LL")
    reverse, reverse_reference = virus.fiber_positions("004", "LL")
    lrs2 = LRS2FiberPositionLoader()

    assert len(fplane) >= 70
    assert normal.shape == (112, 2)
    assert reverse.shape == (112, 2)
    assert normal_reference.path.name == "IFUcen_HETDEX.txt"
    assert reverse_reference.path.name == "IFUcen_HETDEX_reverse_R.txt"
    for channel in ("UV", "Orange", "Red", "Far Red"):
        for amplifier in ("LL", "LU", "RL", "RU"):
            positions, _ = lrs2.fiber_positions(channel, amplifier)
            assert positions.shape == (140, 2)


def test_packaged_virus_positions_span_one_physical_ifu_plane() -> None:
    loader = VirusTopologyLoader()
    positions = np.concatenate(
        [loader.fiber_positions("001", amplifier)[0] for amplifier in ("LL", "LU", "RL", "RU")]
    )

    assert positions.shape == (448, 2)
    assert positions[:, 0].min() == pytest.approx(-24.15)
    assert positions[:, 0].max() == pytest.approx(24.15)
    assert positions[:, 1].min() == pytest.approx(-24.24)
    assert positions[:, 1].max() == pytest.approx(24.24)


def test_real_dated_trace_tree_resolves_virus_and_lrs2() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    loader = VirusTopologyLoader(trace_root=repository_root)
    virus_identity = PhysicalAmplifierIdentity(
        instrument="virus",
        ifu_slot="013",
        amplifier="LL",
        ifuid="043",
        specid="412",
        controller="S/N 0021",
    )
    lrs2_identity = PhysicalAmplifierIdentity(
        instrument="lrs2",
        ifu_slot="056",
        amplifier="LL",
        ifuid="7001",
        specid="503",
        controller="S/N 0086",
    )

    virus_trace, virus_reference = loader.resolve_trace_reference(
        virus_identity, "20230116"
    )
    lrs2_trace, lrs2_reference = loader.resolve_trace_reference(
        lrs2_identity, "20230116"
    )

    assert virus_trace.shape == (112, 2)
    assert lrs2_trace.shape == (140, 2)
    assert virus_reference.path.parent.name == "20230116"
    assert lrs2_reference.path.parent.name == "20181108"

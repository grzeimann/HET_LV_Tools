from datetime import date, datetime
from pathlib import Path

import numpy as np
import pytest

import hetquicklook.highlevel as highlevel
from hetquicklook import (
    Exposure,
    ExposureClassification,
    ExposureMetadata,
    Instrument,
    Observation,
    ObservationMetadata,
    QuicklookProduct,
    QuicklookSite,
)
from hetquicklook.discovery import DiscoveredObservation
from hetquicklook.workflows import (
    LRS2QuicklookSet,
    SpatialQuicklook,
    VIRUSIFUQuicklookSet,
    VIRUSQuicklookSet,
)


def _observation(
    tmp_path: Path,
    observation_id: str,
    exposure_specs: list[tuple[str, str, str | None]],
    *,
    instrument: Instrument = Instrument.LRS2,
    standard_ids: set[str] | None = None,
) -> Observation:
    discovered = DiscoveredObservation(
        archive_path=tmp_path / f"{observation_id}.tar",
        date=date(2026, 5, 12),
        instrument=instrument,
    )
    exposures = []
    for index, (exposure_id, frame_type, object_name) in enumerate(
        exposure_specs, start=1
    ):
        is_standard = standard_ids is not None and exposure_id in standard_ids
        classification = ExposureClassification(
            quicklook_kind="flat" if frame_type == "flt" else None,
            standard_star=(is_standard if frame_type == "sci" else None),
        )
        exposures.append(
            Exposure(
                exposure_id=exposure_id,
                frames=(),
                metadata=ExposureMetadata(
                    exposure_id=exposure_id,
                    frame_types=(frame_type,),
                    frame_class="science" if frame_type == "sci" else "calibration",
                    observation_time=datetime(2026, 5, 12, index, 0),
                    object_name=object_name,
                    exposure_time_s=30.0,
                    program_id="program-1",
                ),
                classification=classification,
                physical_identities={},
            )
        )
    return Observation(
        discovered=discovered,
        members=(),
        metadata=ObservationMetadata(
            observation_id=observation_id,
            archive_path=discovered.archive_path,
            date=discovered.date,
            instrument=instrument,
        ),
        exposures=tuple(exposures),
    )


def test_site_normalizes_roots_and_night_flattens_exposures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _observation(
        tmp_path,
        "obs-1",
        [("exp-b", "sci", "target-b"), ("exp-a", "sci", "target-a")],
    )
    second = _observation(tmp_path, "obs-2", [("exp-c", "flt", "ldls_long_B")])
    discovered = tuple(item.discovered for item in (first, second))
    loaded = iter((first, second))
    monkeypatch.setattr(highlevel, "discover_observations", lambda *args, **kwargs: discovered)
    monkeypatch.setattr(highlevel, "load_observation", lambda *args, **kwargs: next(loaded))

    site = QuicklookSite(
        raw_roots={"LRS2": "~/raw/lrs2"},
        trace_root="~/traces",
    )
    night = site.night("20260512", instrument="lrs2")

    assert site.raw_roots == {"lrs2": Path("~/raw/lrs2").expanduser().resolve()}
    assert site.trace_root == Path("~/traces").expanduser().resolve()
    assert [item.exposure_id for item in night] == ["exp-a", "exp-b", "exp-c"]
    assert [item.row for item in night] == [0, 1, 2]
    assert night[1].observation.observation_id == "obs-1"
    assert night[1].metadata.object_name == "target-b"


def test_site_default_trace_root_is_independent_of_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    site = QuicklookSite(raw_roots={"lrs2": tmp_path})

    assert site.trace_root == Path(highlevel.__file__).resolve().parents[2]


def test_night_html_is_compact_and_escapes_metadata(tmp_path: Path) -> None:
    observation = _observation(
        tmp_path,
        "obs-1",
        [("exp-1", "sci", "<unsafe & target>")],
    )
    # Construct the wrapper directly so this test does not need archive files.
    from hetquicklook.highlevel import QuicklookNight

    night = QuicklookNight(
        site=QuicklookSite(raw_roots={"lrs2": tmp_path}, trace_root=tmp_path),
        date="20260512",
        instrument=Instrument.LRS2,
        discovered=(observation.discovered,),
        observations=(observation,),
    )
    html = night._repr_html_()

    assert "&lt;unsafe &amp; target&gt;" in html
    assert "Exposure ID" in html
    assert "Frame type" in html
    assert "Header disagreements" not in html
    assert "<th" in html


def test_night_html_caps_large_virus_slot_and_amplifier_lists(
    tmp_path: Path,
) -> None:
    from hetquicklook.highlevel import QuicklookNight
    from hetquicklook.discovery import ArchiveMember, RawFrameIdentity
    from hetquicklook.instrument import PhysicalAmplifierIdentity

    frames = []
    identities = {}
    for slot_index in range(1, 76):
        slot = f"{slot_index:03d}"
        for amplifier in ("LL", "LU", "RL", "RU"):
            member = ArchiveMember(
                archive_path=tmp_path / "virus.tar",
                member_name=f"{slot}{amplifier}.fits",
                size=1,
                identity=RawFrameIdentity("exp-1", f"{slot}{amplifier}", "sci"),
            )
            frames.append(member)
            identities[member.member_name] = PhysicalAmplifierIdentity(
                instrument=Instrument.VIRUS,
                ifu_slot=slot,
                amplifier=amplifier,
                ifuid=slot,
                specid="412",
                controller="controller",
            )
    observation = _observation(
        tmp_path,
        "obs-1",
        [("exp-1", "sci", "target")],
        instrument=Instrument.VIRUS,
    )
    exposure = Exposure(
        exposure_id="exp-1",
        frames=tuple(frames),
        metadata=observation.exposures[0].metadata,
        classification=observation.exposures[0].classification,
        physical_identities=identities,
    )
    observation = Observation(
        discovered=observation.discovered,
        members=observation.members,
        metadata=observation.metadata,
        exposures=(exposure,),
    )
    night = QuicklookNight(
        site=QuicklookSite(raw_roots={"virus": tmp_path}, trace_root=tmp_path),
        date="20260512",
        instrument=Instrument.VIRUS,
        discovered=(observation.discovered,),
        observations=(observation,),
    )

    html = night._repr_html_()

    assert "001, 002, 003, 004, 005, 006, 007, 008, … (+67 more)" in html
    assert "001LL, 001LU, 001RL, 001RU, 002LL, 002LU, 002RL, 002RU, … (+292 more)" in html
    assert len(night._rows()[0]["IFU slot"]) == 75
    assert len(night._rows()[0]["Amplifiers/components"]) == 300


def test_virus_product_plot_without_ifu_uses_the_focal_plane_grid() -> None:
    products = {
        slot: VIRUSIFUQuicklookSet(
            ifu_slot=slot,
            amplifier_evidence={},
            product=SpatialQuicklook(
                fiber_values={"fiber": 1.0},
                fiber_positions={"fiber": (0.0, 0.0)},
                image=np.array([[1.0, 2.0], [3.0, 4.0]]),
                spatial_x_coordinates=np.array([-0.5, 0.5]),
                spatial_y_coordinates=np.array([-0.5, 0.5]),
            ),
        )
        for slot in ("074", "075")
    }
    result = VIRUSQuicklookSet(ifus=products)
    product = QuicklookProduct.from_result(None, "target", result)

    figure = product.plot(
        show_fibers=False,
        cmap="magma",
        vmin=1.0,
        vmax=4.0,
    )

    image_axes = {
        axis.get_title(): axis
        for axis in figure.axes
        if axis.get_title() in {"074", "075"}
    }
    assert set(image_axes) == {"074", "075"}
    assert image_axes["074"].get_position().y0 > image_axes["075"].get_position().y0
    assert len(figure.axes) >= 100
    assert image_axes["074"].images[0].get_cmap().name == "magma"
    assert image_axes["074"].images[0].get_clim() == (1.0, 4.0)

    single_figure = product.plot(
        ifu="074",
        show_fibers=False,
        cmap="viridis",
        vmin=0.0,
        vmax=10.0,
    )
    single_image = single_figure.axes[0].images[0]
    assert single_image.get_cmap().name == "viridis"
    assert single_image.get_clim() == (0.0, 10.0)


def test_exposure_id_lookup_rejects_ambiguity(tmp_path: Path) -> None:
    first = _observation(tmp_path, "obs-1", [("same", "sci", "one")])
    second = _observation(tmp_path, "obs-2", [("same", "sci", "two")])
    from hetquicklook.highlevel import QuicklookNight

    night = QuicklookNight(
        site=QuicklookSite(raw_roots={"lrs2": tmp_path}, trace_root=tmp_path),
        date="20260512",
        instrument=Instrument.LRS2,
        discovered=(first.discovered, second.discovered),
        observations=(first, second),
    )

    with pytest.raises(ValueError, match="ambiguous"):
        night.exposure("same")


def test_quicklook_dispatches_from_existing_classification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation(tmp_path, "obs-1", [("exp-1", "flt", "ldls_long_B")])
    from hetquicklook.highlevel import QuicklookNight

    night = QuicklookNight(
        site=QuicklookSite(raw_roots={"lrs2": tmp_path}, trace_root=tmp_path),
        date="20260512",
        instrument=Instrument.LRS2,
        discovered=(observation.discovered,),
        observations=(observation,),
    )
    calls = {}

    def fake_run(exposure, **kwargs):
        calls.update(kwargs)
        return LRS2QuicklookSet(amplifier_evidence={}, channels={})

    monkeypatch.setattr(
        "hetquicklook.highlevel.workflows.run_lrs2_channel_quicklooks", fake_run
    )
    product = night[0].quicklook()

    assert isinstance(product, QuicklookProduct)
    assert product.kind == "flat"
    assert calls["quicklook_kind"] == "flat"
    assert calls["frame_type"] == "flt"
    assert calls["trace_provider"] is night._trace_provider


def test_science_dispatches_to_target_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation(tmp_path, "obs-1", [("exp-1", "sci", "unknown")])
    from hetquicklook.highlevel import QuicklookNight

    night = QuicklookNight(
        site=QuicklookSite(raw_roots={"lrs2": tmp_path}, trace_root=tmp_path),
        date="20260512",
        instrument=Instrument.LRS2,
        discovered=(observation.discovered,),
        observations=(observation,),
    )

    calls = {}

    def fake_run(exposure, **kwargs):
        calls.update(kwargs)
        return LRS2QuicklookSet(amplifier_evidence={}, channels={})

    monkeypatch.setattr(
        "hetquicklook.highlevel.workflows.run_lrs2_channel_quicklooks", fake_run
    )
    product = night[0].quicklook()

    assert product.kind == "target"
    assert calls["quicklook_kind"] == "target"


def test_quicklook_dispatches_recognized_standard_to_standard_workflow(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observation = _observation(
        tmp_path,
        "obs-1",
        [("exp-1", "sci", "HZ44_056_W")],
        standard_ids={"exp-1"},
    )
    from hetquicklook.highlevel import QuicklookNight

    night = QuicklookNight(
        site=QuicklookSite(raw_roots={"lrs2": tmp_path}, trace_root=tmp_path),
        date="20260512",
        instrument=Instrument.LRS2,
        discovered=(observation.discovered,),
        observations=(observation,),
    )
    calls = {}

    def fake_run(exposure, **kwargs):
        calls.update(kwargs)
        return LRS2QuicklookSet(amplifier_evidence={}, channels={})

    monkeypatch.setattr(
        "hetquicklook.highlevel.workflows.run_lrs2_channel_quicklooks", fake_run
    )
    product = night[0].quicklook()

    assert product.kind == "standard"
    assert calls["quicklook_kind"] == "standard"
    assert calls["frame_type"] == "sci"

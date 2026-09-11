from datetime import date
from pathlib import Path
import threading

import numpy as np
import pytest

import hetquicklook.workflows as workflows
from hetquicklook.classification import ExposureClassification
from hetquicklook.discovery import ArchiveMember, RawFrameIdentity
from hetquicklook.fibers import FiberTopology
from hetquicklook.instrument import Instrument, PhysicalAmplifierIdentity
from hetquicklook.metadata import ExposureMetadata
from hetquicklook.observation import Exposure
from hetquicklook.raw import RawFrameData
from hetquicklook.topology import TopologyReference
from hetquicklook.workflows import (
    AmplifierTopologyResult,
    LRS2ChannelQuicklook,
    LRS2_SPATIAL_DEFAULTS,
    PointingQuicklook,
    QuicklookError,
    SpatialQuicklook,
    VIRUS_SPATIAL_DEFAULTS,
    combine_lrs2_channel_products,
    combine_lrs2_channels,
    run_ldls_flat_quicklook,
    run_lrs2_channel_quicklooks,
    run_standard_star_quicklook,
    build_amplifier_topology,
)
from hetquicklook.algorithms.results import AlgorithmResult
from hetquicklook.algorithms.centroid import Centroid


def test_standard_star_workflow_returns_pointing_offset() -> None:
    topology = FiberTopology.from_arrays(
        "amp-a",
        ("left", "center", "right"),
        np.array([[1], [2], [3]]),
        np.array([[1], [1], [1]]),
        np.array([0, 1, 2]),
        np.array([0, 0, 0]),
    )
    detector = np.zeros((3, 4), dtype=float)
    detector[1, 2] = 10
    detector[1, 3] = 10

    result = run_standard_star_quicklook(
        detector, topology, requested_position=(1.0, 0.0)
    )

    assert result.measured_centroid.x == 1.5
    assert result.offset == (0.5, 0.0)


def test_ldls_workflow_produces_spatial_image() -> None:
    topology = FiberTopology.from_arrays(
        "amp-a",
        ("a", "b"),
        np.array([[0], [1]]),
        np.array([[0], [0]]),
        np.array([4, 5]),
        np.array([6, 6]),
    )
    result = run_ldls_flat_quicklook(np.array([[2.0, 3.0]]), topology)

    assert result.image.shape == (1, 2)
    assert result.image.tolist() == [[2.0, 3.0]]


def _dense_topology() -> FiberTopology:
    detector_columns = 220
    fiber_ids = ("f0", "f1", "f2")
    detector_x = np.tile(np.arange(detector_columns, dtype=float), (3, 1))
    detector_y = np.repeat(np.array([[10.0], [15.0], [20.0]]), detector_columns, axis=1)
    return FiberTopology.from_arrays(
        "amp-a",
        fiber_ids,
        detector_x,
        detector_y,
        np.array([-1.0, 0.0, 1.0]),
        np.zeros(3),
    )


def test_compact_spatial_path_discards_extraction_and_skips_splat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_splat(*args, **kwargs):
        raise AssertionError("compact amplifier path must not spatially splat")

    monkeypatch.setattr(workflows, "gaussian_splat", fail_splat)
    result = workflows._spatial_quicklook(
        np.ones((30, 220), dtype=float),
        _dense_topology(),
        detector_variance=np.ones((30, 220), dtype=float),
        collapse_columns=200,
        retain_extraction=False,
        build_spatial=False,
    )

    assert result.image.shape == (0, 0)
    assert len(result.fiber_values) == 3
    assert result.extracted_spectra is None
    assert result.extraction_variance is None
    assert result.extraction_valid_fraction is None
    assert result.effective_aperture_width is None
    assert result.extraction_valid is None
    assert result.spatial_support is None
    assert result.spatial_weight is None


def test_workflow_selects_instrument_spatial_defaults_and_pads_bounds() -> None:
    detector = np.ones((30, 220), dtype=float)
    virus = run_ldls_flat_quicklook(
        detector, _dense_topology(), instrument="virus"
    )
    lrs2 = run_ldls_flat_quicklook(
        detector, _dense_topology(), instrument="lrs2"
    )

    assert VIRUS_SPATIAL_DEFAULTS["gaussian_fwhm_arcsec"] == 1.5
    assert VIRUS_SPATIAL_DEFAULTS["pixel_scale_arcsec"] == 1.0
    assert VIRUS_SPATIAL_DEFAULTS["grid_padding_arcsec"] == 1.1
    assert LRS2_SPATIAL_DEFAULTS["gaussian_fwhm_arcsec"] == 1.2
    assert LRS2_SPATIAL_DEFAULTS["pixel_scale_arcsec"] == 0.4
    assert LRS2_SPATIAL_DEFAULTS["grid_padding_arcsec"] == 0.3
    assert virus.spatial_gaussian_fwhm_arcsec == 1.5
    assert virus.spatial_pixel_scale_arcsec == 1.0
    assert lrs2.spatial_gaussian_fwhm_arcsec == 1.2
    assert lrs2.spatial_pixel_scale_arcsec == 0.4
    assert np.diff(virus.spatial_x_coordinates).tolist() == [1.0] * (
        virus.spatial_x_coordinates.size - 1
    )
    np.testing.assert_allclose(np.diff(lrs2.spatial_x_coordinates), 0.4)
    assert virus.spatial_x_coordinates[0] == pytest.approx(-3.0)
    assert virus.spatial_x_coordinates[-1] == pytest.approx(3.0)
    assert lrs2.spatial_x_coordinates[0] == pytest.approx(-1.6)
    assert lrs2.spatial_x_coordinates[-1] == pytest.approx(1.6)


def test_workflow_spatial_overrides_are_passed_to_generic_splat() -> None:
    result = run_ldls_flat_quicklook(
        np.ones((30, 220), dtype=float),
        _dense_topology(),
        instrument="lrs2",
        gaussian_fwhm=2.0,
        pixel_scale=0.5,
        grid_padding_arcsec=0.0,
    )

    assert result.spatial_gaussian_fwhm_arcsec == 2.0
    assert result.spatial_pixel_scale_arcsec == 0.5
    assert result.spatial_x_coordinates[0] == pytest.approx(-1.0)
    assert result.spatial_x_coordinates[-1] == pytest.approx(1.0)
    np.testing.assert_allclose(np.diff(result.spatial_x_coordinates), 0.5)


def test_standard_workflow_exposes_both_fiducials_without_rejection() -> None:
    detector = np.ones((30, 220), dtype=float)
    virus = run_standard_star_quicklook(
        detector, _dense_topology(), instrument="virus"
    )
    lrs2 = run_standard_star_quicklook(
        detector, _dense_topology(), instrument="lrs2"
    )

    assert virus.intended_fiducial == (0.0, 0.0)
    assert virus.requested_position == (0.0, 0.0)
    assert lrs2.intended_fiducial == (0.0, 0.0)
    assert lrs2.requested_position == (0.0, 0.0)
    assert virus.offset is not None
    assert lrs2.offset is not None


def _lrs2_amplifier_product(
    token: str,
    *,
    value: float = 1.0,
    x_position: float = 0.0,
    pointing: bool = False,
) -> SpatialQuicklook:
    fiber_ids = tuple(f"{token}-{index:03d}" for index in range(140))
    positions = {
        fiber_id: (x_position, float(index % 14) * 0.1)
        for index, fiber_id in enumerate(fiber_ids)
    }
    values = {fiber_id: value for fiber_id in fiber_ids}
    product = SpatialQuicklook(
        fiber_values=values,
        image=np.zeros((3, 3)),
        instrument="lrs2",
        fiber_positions=positions,
        intended_fiducial=(0.0, 0.0),
    )
    if not pointing:
        return product
    return PointingQuicklook(
        fiber_values=product.fiber_values,
        image=product.image,
        instrument=product.instrument,
        fiber_positions=product.fiber_positions,
        intended_fiducial=product.intended_fiducial,
        measured_centroid=Centroid(99.0, 99.0, 1.0, 1),
        requested_position=(0.0, 0.0),
    )


def test_lrs2_channel_combination_uses_authoritative_pairs_and_280_fibers() -> None:
    products = {
        "056LL": _lrs2_amplifier_product("056LL", value=1.0),
        "056LU": _lrs2_amplifier_product("056LU", value=2.0),
    }

    result = combine_lrs2_channel_products("UV", products)

    assert isinstance(result, LRS2ChannelQuicklook)
    assert result.channel == "UV"
    assert tuple(result.amplifier_products) == ("056LL", "056LU")
    assert result.fiber_positions.shape == (280, 2)
    assert result.fiber_values.shape == (280,)
    assert result.fiber_values[:140].tolist() == [1.0] * 140
    assert result.fiber_values[140:].tolist() == [2.0] * 140
    assert result.spatial_gaussian_fwhm_arcsec == 1.2
    assert result.spatial_pixel_scale_arcsec == 0.4
    assert result.intended_fiducial == (0.0, 0.0)


def test_all_lrs2_channels_combine_into_four_280_fiber_products() -> None:
    tokens = (
        "056LL", "056LU", "056RL", "056RU",
        "066LL", "066LU", "066RL", "066RU",
    )
    products = {token: _lrs2_amplifier_product(token) for token in tokens}

    channels = combine_lrs2_channels(products)

    assert tuple(channels) == ("UV", "Orange", "Red", "Far-Red")
    assert all(result.fiber_positions.shape == (280, 2) for result in channels.values())
    assert all(result.fiber_values.shape == (280,) for result in channels.values())


def test_lrs2_channel_combination_preserves_position_value_correspondence() -> None:
    first = _lrs2_amplifier_product("056LL", value=10.0, x_position=-2.0)
    second = _lrs2_amplifier_product("056LU", value=20.0, x_position=3.0)

    result = combine_lrs2_channel_products(
        "UV", {"056LL": first, "056LU": second}
    )

    np.testing.assert_allclose(result.fiber_positions[:140, 0], -2.0)
    np.testing.assert_allclose(result.fiber_positions[140:, 0], 3.0)
    np.testing.assert_allclose(result.fiber_values[:140], 10.0)
    np.testing.assert_allclose(result.fiber_values[140:], 20.0)


def test_lrs2_channel_centroid_is_recomputed_from_all_fibers() -> None:
    first = _lrs2_amplifier_product(
        "056LL", value=1.0, x_position=-1.0, pointing=True
    )
    second = _lrs2_amplifier_product(
        "056LU", value=1.0, x_position=3.0, pointing=True
    )

    result = combine_lrs2_channel_products(
        "UV", {"056LL": first, "056LU": second}
    )

    assert result.measured_centroid is not None
    assert result.measured_centroid.x == 1.0
    assert result.measured_centroid.y == pytest.approx(0.65)
    assert result.offset == pytest.approx((1.0, 0.65))


def test_lrs2_channel_spatial_overrides_are_explicit() -> None:
    products = {
        "056LL": _lrs2_amplifier_product("056LL"),
        "056LU": _lrs2_amplifier_product("056LU"),
    }

    result = combine_lrs2_channel_products(
        "UV",
        products,
        gaussian_fwhm_arcsec=2.0,
        pixel_scale_arcsec=0.5,
    )

    assert result.spatial_gaussian_fwhm_arcsec == 2.0
    assert result.spatial_pixel_scale_arcsec == 0.5
    np.testing.assert_allclose(np.diff(result.spatial_x_coordinates), 0.5)


def test_lrs2_channel_does_not_accept_missing_or_wrong_amplifier() -> None:
    product = _lrs2_amplifier_product("056LL")
    with pytest.raises(ValueError, match="missing amplifier"):
        combine_lrs2_channel_products("UV", {"056LL": product})
    with pytest.raises(ValueError, match="does not belong"):
        combine_lrs2_channel_products(
            "UV",
            {"056LL": product, "056RL": _lrs2_amplifier_product("056RL")},
        )


def test_lrs2_channel_set_does_not_silently_create_partial_products() -> None:
    products = {
        "056LL": _lrs2_amplifier_product("056LL"),
        "056LU": _lrs2_amplifier_product("056LU"),
    }

    with pytest.raises(ValueError, match="missing amplifier"):
        combine_lrs2_channels(products)


@pytest.mark.parametrize(
    ("instrument", "nfiber"),
    ((Instrument.VIRUS, 112), (Instrument.LRS2, 140)),
)
def test_build_amplifier_topology_uses_trace_count_for_both_instruments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instrument: Instrument,
    nfiber: int,
) -> None:
    identity = PhysicalAmplifierIdentity(
        instrument=instrument,
        ifu_slot="001" if instrument is Instrument.VIRUS else "056",
        amplifier="LL",
        ifuid="001" if instrument is Instrument.VIRUS else "7001",
        specid="412" if instrument is Instrument.VIRUS else "503",
        controller="controller",
    )
    reference_path = tmp_path / "fiber_loc.txt"
    trace_reference = np.column_stack((np.arange(nfiber), np.zeros(nfiber)))
    monkeypatch.setattr(
        workflows.VirusTopologyLoader,
        "resolve_trace_reference",
        lambda self, resolved_identity, at=None: (
            trace_reference,
            TopologyReference("trace", reference_path),
        ),
    )
    monkeypatch.setattr(
        workflows,
        "fit_fiber_traces",
        lambda image, reference, **kwargs: AlgorithmResult(
            "trace",
            "test",
            {"fiber_trace_map": np.zeros((nfiber, image.shape[1]))},
            {},
        ),
    )

    result = build_amplifier_topology(
        np.zeros((4, 6)), identity, trace_root=tmp_path, at=date(2026, 1, 1)
    )

    assert len(result.topology.fiber_ids) == nfiber
    assert result.trace_result.get_array("fiber_trace_map").shape == (nfiber, 6)
    assert result.trace_provenance.path == reference_path
    assert result.position_provenance.path.name


def test_build_amplifier_topology_preserves_trace_hardware_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = PhysicalAmplifierIdentity(
        instrument=Instrument.VIRUS,
        ifu_slot="018",
        amplifier="RU",
        ifuid="018",
        specid="504",
        controller="controller",
    )
    trace_reference = np.column_stack((np.arange(111), np.zeros(111)))
    monkeypatch.setattr(
        workflows.VirusTopologyLoader,
        "resolve_trace_reference",
        lambda self, resolved_identity, at=None: (
            trace_reference,
            TopologyReference("trace", tmp_path / "fiber_loc.txt"),
        ),
    )
    monkeypatch.setattr(
        workflows,
        "fit_fiber_traces",
        lambda image, reference, **kwargs: AlgorithmResult(
            "trace",
            "test",
            {"fiber_trace_map": np.zeros((111, image.shape[1]))},
            {},
        ),
    )

    result = build_amplifier_topology(
        np.zeros((4, 6)), identity, trace_root=tmp_path
    )

    assert len(result.topology.fiber_ids) == 111


def test_lrs2_batch_workflow_retains_amplifier_evidence_and_channels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = (
        "056LL", "056LU", "056RL", "056RU",
        "066LL", "066LU", "066RL", "066RU",
    )
    frames = []
    identities = {}
    for token in tokens:
        identity = RawFrameIdentity("exp001", token, "twi")
        member = ArchiveMember(
            archive_path=tmp_path / "lrs2.tar",
            member_name=f"exp001_{token}_twi.fits",
            size=1,
            identity=identity,
        )
        frames.append(member)
        identities[member.member_name] = PhysicalAmplifierIdentity(
            instrument=Instrument.LRS2,
            ifu_slot=identity.ifu_slot,
            amplifier=identity.amplifier,
            ifuid="7001" if identity.ifu_slot == "056" else "7002",
            specid="503" if identity.ifu_slot == "056" else "502",
            controller="controller",
        )
    exposure = Exposure(
        exposure_id="exp001",
        frames=tuple(frames),
        metadata=ExposureMetadata(
            exposure_id="exp001", frame_types=("twi",), frame_class="calibration"
        ),
        classification=None,  # type: ignore[arg-type]
        physical_identities=identities,
    )

    class Loader:
        def load(self, member):
            return RawFrameData(
                data=np.zeros((2, 2)),
                header={},
                path=str(member.archive_path),
                tar_member=member.member_name,
                identity=member.identity,
            )

    detector = AlgorithmResult(
        "detector", "test", {
            "oriented_detector_image": np.zeros((10, 220)),
            "detector_variance": np.ones((10, 220)),
        }, {},
    )
    monkeypatch.setattr(workflows, "reduce_amplifier_array", lambda data, header: detector)
    trace_reference = np.array([[5.0, 0.0]])
    trace_calls = []
    monkeypatch.setattr(
        workflows.VirusTopologyLoader,
        "resolve_trace_reference",
        lambda self, resolved_identity, at=None: (
            trace_reference,
            TopologyReference("trace", tmp_path / "trace"),
        ),
    )
    def fake_trace(image, reference, **kwargs):
        trace_calls.append((image.shape, kwargs))
        return AlgorithmResult(
            "trace",
            "test",
            {
                "fiber_trace_map": np.full((1, image.shape[1]), 5.0),
                "trace_reference": reference,
            },
            {},
        )

    monkeypatch.setattr(workflows, "fit_fiber_traces", fake_trace)

    def fake_topology(prepared_detector, physical_identity, **kwargs):
        topology = FiberTopology.from_arrays(
            physical_identity.amplifier,
            ("fiber",),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([0.0]),
            np.array([0.0]),
        )
        return AmplifierTopologyResult(
            topology=topology,
            trace_result=AlgorithmResult("trace", "test", {}, {}),
            physical_identity=physical_identity,
            trace_provenance=TopologyReference("trace", tmp_path / "trace"),
            position_provenance=TopologyReference("positions", tmp_path / "positions"),
        )

    monkeypatch.setattr(workflows, "build_amplifier_topology", fake_topology)
    calls = []
    flat_kwargs = []

    def fake_flat(prepared_detector, topology, **kwargs):
        calls.append(topology.amplifier)
        flat_kwargs.append(kwargs)
        return _lrs2_amplifier_product(topology.amplifier, value=1.0)

    monkeypatch.setattr(workflows, "run_ldls_flat_quicklook", fake_flat)

    result = run_lrs2_channel_quicklooks(
        exposure, trace_root=tmp_path, frame_type="twi", loader=Loader()
    )

    assert tuple(result.channels) == ("UV", "Orange", "Red", "Far-Red")
    assert len(result.amplifier_evidence) == 8
    assert set(result.amplifier_products) == set(tokens)
    assert all(channel.fiber_values.shape == (280,) for channel in result.channels.values())
    assert all(item.loaded is None for item in result.amplifier_evidence.values())
    assert all(item.detector is None for item in result.amplifier_evidence.values())
    assert all(item.product.extracted_spectra is None for item in result.amplifier_evidence.values())
    assert len(trace_calls) == 8
    assert all(shape == (10, 200) for shape, _ in trace_calls)
    assert all(call["n_chunks"] == 5 for _, call in trace_calls)
    assert all(call["degree"] == 1 for _, call in trace_calls)
    assert all(call["fit_method"] == "fast" for _, call in trace_calls)
    assert all(call["detector_column_start"] == 10 for _, call in trace_calls)
    assert all(not kwargs["retain_extraction"] for kwargs in flat_kwargs)
    assert all(not kwargs["build_spatial"] for kwargs in flat_kwargs)
    assert len(calls) == 8


def test_lrs2_batch_workflow_fails_on_missing_required_amplifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = (
        "056LL", "056RL", "056RU",
        "066LL", "066LU", "066RL", "066RU",
    )
    frames = []
    identities = {}
    for token in tokens:
        identity = RawFrameIdentity("exp001", token, "twi")
        member = ArchiveMember(
            archive_path=tmp_path / "lrs2.tar",
            member_name=f"exp001_{token}_twi.fits",
            size=1,
            identity=identity,
        )
        frames.append(member)
        identities[member.member_name] = PhysicalAmplifierIdentity(
            instrument=Instrument.LRS2,
            ifu_slot=identity.ifu_slot,
            amplifier=identity.amplifier,
            ifuid="7001" if identity.ifu_slot == "056" else "7002",
            specid="503" if identity.ifu_slot == "056" else "502",
            controller="controller",
        )
    exposure = Exposure(
        exposure_id="exp001",
        frames=tuple(frames),
        metadata=ExposureMetadata(
            exposure_id="exp001", frame_types=("twi",), frame_class="calibration"
        ),
        classification=ExposureClassification(quicklook_kind="flat"),
        physical_identities=identities,
    )

    class Loader:
        def load(self, member):
            return RawFrameData(
                data=np.zeros((2, 2)),
                header={},
                path=str(member.archive_path),
                tar_member=member.member_name,
                identity=member.identity,
            )

    detector = AlgorithmResult(
        "detector", "test", {
            "oriented_detector_image": np.zeros((2, 2)),
            "detector_variance": np.ones((2, 2)),
        }, {},
    )
    monkeypatch.setattr(workflows, "reduce_amplifier_array", lambda data, header: detector)

    def fake_topology(prepared_detector, physical_identity, **kwargs):
        topology = FiberTopology.from_arrays(
            physical_identity.amplifier,
            ("fiber",),
            np.array([[0.0]]),
            np.array([[0.0]]),
            np.array([0.0]),
            np.array([0.0]),
        )
        return AmplifierTopologyResult(
            topology=topology,
            trace_result=AlgorithmResult("trace", "test", {}, {}),
            physical_identity=physical_identity,
            trace_provenance=TopologyReference("trace", tmp_path / "trace"),
            position_provenance=TopologyReference("positions", tmp_path / "positions"),
        )

    monkeypatch.setattr(workflows, "build_amplifier_topology", fake_topology)
    monkeypatch.setattr(
        workflows,
        "run_ldls_flat_quicklook",
        lambda prepared_detector, topology, **kwargs: _lrs2_amplifier_product(
            topology.amplifier, value=1.0
        ),
    )

    with pytest.raises(QuicklookError, match="056LU"):
        run_lrs2_channel_quicklooks(
            exposure,
            trace_root=tmp_path,
            frame_type="twi",
            loader=Loader(),
        )


def test_virus_archive_workflow_fails_on_incomplete_ifu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokens = (
        "074LL", "074LU", "074RL", "074RU",
        "075LL", "075LU", "075RL",
    )
    frames = []
    identities = {}
    for token in tokens:
        identity = RawFrameIdentity("exp001", token, "twi")
        member = ArchiveMember(
            archive_path=tmp_path / "virus.tar",
            member_name=f"exp001_{token}_twi.fits",
            size=1,
            identity=identity,
        )
        frames.append(member)
        identities[member.member_name] = PhysicalAmplifierIdentity(
            instrument=Instrument.VIRUS,
            ifu_slot=identity.ifu_slot,
            amplifier=identity.amplifier,
            ifuid=identity.ifu_slot,
            specid="412",
            controller="controller",
        )
    exposure = Exposure(
        exposure_id="exp001",
        frames=tuple(frames),
        metadata=ExposureMetadata(
            exposure_id="exp001", frame_types=("twi",), frame_class="calibration"
        ),
        classification=ExposureClassification(quicklook_kind="flat"),
        physical_identities=identities,
    )
    amplifier_product = SpatialQuicklook(
        fiber_values={"fiber": 1.0},
        image=np.ones((1, 1)),
        instrument=Instrument.VIRUS,
    )
    monkeypatch.setattr(
        workflows,
        "_run_archive_amplifier_quicklook",
        lambda *args, **kwargs: (object(), object(), object(), amplifier_product),
    )

    with pytest.raises(QuicklookError, match="075RU"):
        workflows.run_virus_ifu_quicklooks(
            exposure, trace_root=tmp_path, frame_type="twi"
        )


def test_virus_workflow_composes_one_ifu_image_and_reports_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot_ids = (("074", "043"), ("075", "044"))
    tokens = tuple(
        f"{slot}{amplifier}"
        for slot, _ifuid in slot_ids
        for amplifier in ("LL", "LU", "RL", "RU")
    )
    frames = []
    identities = {}
    products = {}
    for slot, ifuid in slot_ids:
        for amp_index, amplifier in enumerate(("LL", "LU", "RL", "RU")):
            token = f"{slot}{amplifier}"
            identity = RawFrameIdentity("exp001", token, "twi")
            member = ArchiveMember(
                archive_path=tmp_path / "virus.tar",
                member_name=f"exp001_{token}_twi.fits",
                size=1,
                identity=identity,
            )
            frames.append(member)
            identities[member.member_name] = PhysicalAmplifierIdentity(
                instrument=Instrument.VIRUS,
                ifu_slot=identity.ifu_slot,
                amplifier=identity.amplifier,
                ifuid=ifuid,
                specid="412",
                controller="controller",
            )
            values = {
                f"{token}-{index:03d}": float(index + amp_index)
                for index in range(112)
            }
            positions = {
                fiber_id: (
                    float(index % 16) - 7.5,
                    float(index // 16) + amp_index * 8.0,
                )
                for index, fiber_id in enumerate(values)
            }
            products[token] = SpatialQuicklook(
                fiber_values=values,
                fiber_positions=positions,
                image=np.ones((2, 2)),
                instrument=Instrument.VIRUS,
            )

    exposure = Exposure(
        exposure_id="exp001",
        frames=tuple(frames),
        metadata=ExposureMetadata(
            exposure_id="exp001", frame_types=("twi",), frame_class="calibration"
        ),
        classification=ExposureClassification(quicklook_kind="flat"),
        physical_identities=identities,
    )

    barrier = threading.Barrier(2)
    thread_names: set[str] = set()

    def fake_run(_exposure, frame, **kwargs):
        token = frame.identity.amplifier_token
        thread_names.add(threading.current_thread().name)
        barrier.wait(timeout=2.0)
        return (object(), object(), object(), products[token])

    monkeypatch.setattr(workflows, "_run_archive_amplifier_quicklook", fake_run)

    result = workflows.run_virus_ifu_quicklooks(
        exposure,
        trace_root=tmp_path,
        frame_type="twi",
        nworkers=2,
        timing=True,
        memory_check=True,
    )

    ifu = result.ifus["074"]
    assert ifu.product is not None
    assert len(ifu.product.fiber_values) == 448
    assert ifu.product.image.shape[0] > 20
    assert ifu.product.image.shape[1] > 10
    assert ifu.diagnostics.worker_count == 2
    assert ifu.diagnostics.retained_array_bytes > 0
    assert "ifu_composition" in ifu.diagnostics.stage_seconds
    assert set(result.diagnostics) == {"074", "075"}
    assert len(thread_names) == 2

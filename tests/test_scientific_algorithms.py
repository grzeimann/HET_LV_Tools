import numpy as np
import pytest
import hetquicklook.algorithms.trace as trace_algorithm

from hetquicklook import (
    STANDARD_STAR_ALIASES,
    STANDARD_STAR_NAMES,
    classify_exposure,
    normalize_standard_star_name,
)
from hetquicklook.algorithms.collapse import (
    collapse_extracted_spectra,
    select_central_columns,
)
from hetquicklook.algorithms.detector import (
    reduce_amplifier_array,
    orient_amplifier_image,
)
from hetquicklook.algorithms.extraction import (
    extract_fractional_aperture,
    fractional_aperture_geometry,
)
from hetquicklook.algorithms.spatial import gaussian_splat
from hetquicklook.algorithms.trace import fit_fiber_traces


def test_all_canonical_standard_names_are_exact_members() -> None:
    assert len(STANDARD_STAR_NAMES) == 45
    for name in STANDARD_STAR_NAMES:
        result = classify_exposure(
            "virus", frame_types=("sci",), object_name=f"{name}_056_W"
        )
        assert result.standard_target == name
        assert result.standard_star is True


def test_historical_aliases_normalize_without_broad_matching() -> None:
    for alias, canonical in STANDARD_STAR_ALIASES.items():
        assert normalize_standard_star_name(alias) == canonical
        result = classify_exposure(
            "virus", frame_types=("sci",), object_name=f"{alias}_056_W"
        )
        assert result.standard_target == canonical
        assert result.standard_star is True

    assert classify_exposure(
        "virus", frame_types=("sci",), object_name="NOTHZ44_056_W"
    ).standard_star is False
    parsed = classify_exposure(
        "virus",
        frame_types=("sci",),
        object_name="TARGET_WITH_UNDERSCORES_056_W",
    )
    assert parsed.standard_target == "TARGET_WITH_UNDERSCORES"
    assert parsed.standard_star is False


def test_detector_reduction_matches_shared_raw_contract() -> None:
    raw = np.full((3, 1064), 110.0)
    raw[:, :1032] = np.array([10.0, 20.0, 30.0])[:, None] + 110.0
    result = reduce_amplifier_array(
        raw,
        {"CCDPOS": "L", "CCDHALF": "L", "GAIN": 2.0, "RDNOISE": 3.0},
    )
    assert result.scalars["overscan_columns"] == 32
    np.testing.assert_allclose(result.get_array("overscan_model"), 110.0)
    expected = np.repeat(np.array([20.0, 40.0, 60.0])[:, None], 1032, axis=1)
    np.testing.assert_allclose(result.get_array("oriented_detector_image"), expected)
    np.testing.assert_allclose(result.get_array("detector_variance"), expected + 9.0)

    fallback = reduce_amplifier_array(raw)
    assert fallback.scalars["gain"] == 0.85
    assert fallback.scalars["read_noise"] == 3.0


@pytest.mark.parametrize("amplifier", ["LU", "RL"])
def test_detector_orientation_preserves_both_axis_flip(amplifier: str) -> None:
    image = np.arange(6).reshape(2, 3)
    np.testing.assert_array_equal(
        orient_amplifier_image(image, amplifier), [[5, 4, 3], [2, 1, 0]]
    )
    np.testing.assert_array_equal(
        orient_amplifier_image(image, "LL", "UL"), [[2, 1, 0], [5, 4, 3]]
    )


def _synthetic_flat(nfiber: int) -> tuple[np.ndarray, np.ndarray]:
    height = 4 * nfiber + 8
    columns = 80
    reference = np.column_stack((np.arange(nfiber) * 4.0 + 3.0, np.zeros(nfiber)))
    reference[17, 1] = 1.0
    y = np.arange(height, dtype=float)[:, None]
    x = np.arange(columns, dtype=float)[None, :]
    flat = np.zeros((height, columns), dtype=float)
    for center in reference[:, 0, None] + 0.2 * np.sin(x / 20.0):
        flat += np.exp(-0.5 * np.square((y - center) / 0.8))
    return flat, reference


@pytest.mark.parametrize("nfiber", [112, 140])
def test_trace_shape_and_qa_follow_reference_fiber_count(nfiber: int) -> None:
    flat, reference = _synthetic_flat(nfiber)
    result = fit_fiber_traces(
        flat, reference, specid="412", ifuid="043", amplifier="LL"
    )
    assert result.get_array("fiber_trace_map").shape == (nfiber, 80)
    assert result.get_array("sampled_trace_positions").shape == (nfiber, 40)
    assert result.get_array("trace_fit_residuals").shape == (nfiber, 40)
    assert result.get_array("per_fiber_trace_residual_rms").shape == (nfiber,)
    assert result.get_array("per_fiber_valid_sample_count").shape == (nfiber,)
    assert result.get_array("trace_interpolated_fiber_mask")[17] == 1


def test_quick_trace_is_local_and_records_requested_fit_parameters() -> None:
    flat, reference = _synthetic_flat(20)
    local_start, local_stop = 11, 61
    result = fit_fiber_traces(
        flat[:, local_start:local_stop],
        reference,
        specid="412",
        ifuid="043",
        amplifier="LL",
        n_chunks=5,
        degree=1,
        detector_column_start=local_start,
    )

    assert result.get_array("fiber_trace_map").shape == (20, local_stop - local_start)
    assert result.get_array("trace_sample_columns").shape == (5,)
    assert result.scalars["trace_n_chunks"] == 5
    assert result.scalars["trace_degree_requested"] == 1
    assert result.scalars["trace_column_start"] == local_start
    assert result.scalars["trace_column_stop"] == local_stop
    assert result.metadata["trace_model"] == "per_fiber_huber_polynomial"
    assert result.metadata["trace_fit_method"] == "robust"
    assert result.metadata["trace_column_bounds"] == [local_start, local_stop]


def test_fast_trace_fit_batches_complete_fibers_and_records_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.column_stack((np.arange(3, dtype=float), np.zeros(3)))
    image = np.ones((8, 10), dtype=float)
    x_chunks = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
    samples = 4.0 + 0.25 * x_chunks[None, :] + np.arange(3)[:, None]

    def fake_trace(_profile, n_fibers, _reference):
        assert n_fibers == 3
        return samples[:, fake_trace.calls].copy()

    fake_trace.calls = 0

    def traced_fake(profile, n_fibers, reference_value):
        result = fake_trace(profile, n_fibers, reference_value)
        fake_trace.calls += 1
        return result

    monkeypatch.setattr(trace_algorithm, "_trace_from_flat_chunk", traced_fake)
    result = trace_algorithm.fit_fiber_traces(
        image,
        reference,
        n_chunks=5,
        degree=1,
        fit_method="fast",
    )

    expected = 4.0 + 0.25 * np.arange(10, dtype=float)[None, :] + np.arange(3)[:, None]
    np.testing.assert_allclose(result.get_array("fiber_trace_map"), expected)
    assert result.scalars["trace_fit_method"] == "fast"
    assert result.metadata["trace_fit_method"] == "fast"
    assert result.metadata["trace_model"] == "per_fiber_ordinary_least_squares_polynomial"


def test_fast_trace_fit_falls_back_for_missing_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = np.column_stack((np.arange(2, dtype=float), np.zeros(2)))
    image = np.ones((6, 10), dtype=float)
    x_chunks = np.array([0.5, 2.5, 4.5, 6.5, 8.5])
    samples = 3.0 + 0.5 * x_chunks[None, :] + np.arange(2)[:, None]
    samples[1, 2] = 0.0
    calls = 0

    def fake_trace(_profile, _n_fibers, _reference):
        nonlocal calls
        result = samples[:, calls].copy()
        calls += 1
        return result

    monkeypatch.setattr(trace_algorithm, "_trace_from_flat_chunk", fake_trace)
    result = trace_algorithm.fit_fiber_traces(
        image,
        reference,
        n_chunks=5,
        degree=1,
        fit_method="fast",
    )

    assert result.get_array("per_fiber_valid_sample_count")[1] == 4
    np.testing.assert_allclose(
        result.get_array("fiber_trace_map")[1],
        4.0 + 0.5 * np.arange(10, dtype=float),
    )


def test_trace_timing_reports_accumulated_compact_substages() -> None:
    flat, reference = _synthetic_flat(20)
    timings: dict[str, float] = {}
    fit_fiber_traces(
        flat,
        reference,
        n_chunks=5,
        degree=1,
        fit_method="fast",
        timings=timings,
    )

    assert set(timings) == {
        "chunk_profile_collapse",
        "flat_profile_preprocessing",
        "percentile_filter",
        "background_polynomial",
        "gaussian_smoothing",
        "peak_detection_assignment",
        "polynomial_trace_fit",
        "trace_result_qa",
    }
    assert all(value >= 0.0 for value in timings.values())


def test_trace_preserves_demonstrated_virus_hardware_exception() -> None:
    flat = np.zeros((20, 80), dtype=float)
    y = np.arange(20, dtype=float)[:, None]
    for center in (5.0, 12.0):
        flat += np.exp(-0.5 * np.square((y - center) / 0.8))
    reference = np.array([[5.0, 0.0], [12.0, 0.0]])
    result = fit_fiber_traces(
        flat, reference, specid="504", ifuid="018", amplifier="RU"
    )
    assert result.get_array("fiber_trace_map").shape == (1, 80)
    assert result.get_array("trace_reference").shape == (1, 2)


@pytest.mark.parametrize("nfiber", [112, 140])
def test_fractional_extraction_is_shape_driven(nfiber: int) -> None:
    detector = np.ones((20, 4), dtype=float)
    variance = np.full_like(detector, 4.0)
    traces = np.full((nfiber, 4), 10.25)
    result = extract_fractional_aperture(detector, variance, traces)
    assert result.get_array("spectrum").shape == (nfiber, 4)
    np.testing.assert_allclose(result.get_array("spectrum"), 5.0)
    np.testing.assert_allclose(
        result.get_array("variance"), np.sum(4.0 * np.square(
            result.get_array("fractional_weights")
        ), axis=-1)
    )
    assert result.scalars["aperture_width_pixels"] == 5.0


def test_fractional_aperture_has_exact_edge_weights_and_edge_state() -> None:
    rows, weights, valid = fractional_aperture_geometry(
        np.full((1, 1), 5.25), 12, width=5.0
    )
    np.testing.assert_array_equal(rows[0, 0], [2, 3, 4, 5, 6, 7])
    np.testing.assert_allclose(weights[0, 0], [0.25, 1, 1, 1, 1, 0.75])
    assert valid[0, 0]

    edge = extract_fractional_aperture(
        np.ones((12, 1)), np.ones((12, 1)), np.array([[1.0]])
    )
    assert edge.get_array("extraction_valid")[0, 0] == 0
    assert np.isnan(edge.get_array("spectrum")[0, 0])


def test_collapse_selects_central_columns_and_keeps_statistic_explicit() -> None:
    spectra = np.arange(2 * 300, dtype=float).reshape(2, 300)
    selected = select_central_columns(spectra, collapse_columns=200)
    np.testing.assert_array_equal(selected, spectra[:, 50:250])
    np.testing.assert_array_equal(
        select_central_columns(np.arange(9, dtype=float)[None, :], collapse_columns=3),
        [[3.0, 4.0, 5.0]],
    )
    result = collapse_extracted_spectra(spectra, collapse_columns=200)
    assert result.scalars["collapse_statistic"] == "median"
    np.testing.assert_allclose(result.get_array("fiber_values"), np.median(selected, axis=1))


def test_collapse_default_is_finite_median_and_width_remains_configurable() -> None:
    spectra = np.full((1, 8), np.nan)
    spectra[0, 2:6] = [1.0, 2.0, 100.0, 4.0]
    result = collapse_extracted_spectra(spectra, collapse_columns=4)

    assert result.get_array("selected_spectra").shape == (1, 4)
    assert result.get_array("fiber_values")[0] == 3.0
    assert result.scalars["collapse_columns_selected"] == 4


def test_collapse_alternate_statistics_remain_explicit() -> None:
    spectra = np.array([[1.0, 2.0, 100.0, 4.0]])
    result = collapse_extracted_spectra(
        spectra, collapse_columns=4, statistic="mean"
    )
    assert result.scalars["collapse_statistic"] == "mean"
    assert result.get_array("fiber_values")[0] == pytest.approx(26.75)


def test_gaussian_splat_normalizes_one_shot_values_and_exposes_support() -> None:
    result = gaussian_splat(
        np.array([[0.0, 0.0]]),
        np.array([7.0]),
        np.array([2.0]),
        fwhm=1.8,
        pixel_scale=1.0,
        output_shape=(5, 5),
        origin=(-2.0, -2.0),
    )
    assert result.image[2, 2] == pytest.approx(7.0)
    assert result.variance is not None
    assert result.variance[2, 2] == pytest.approx(4.0)
    assert result.support[2, 2]
    assert np.isnan(result.image[0, 0])
    assert not result.support[0, 0]
    assert result.contribution_count.max() == 1


def test_gaussian_splat_supports_subpixel_and_multiple_fibers() -> None:
    result = gaussian_splat(
        np.array([[0.4, 0.3], [1.0, 0.0]]),
        np.array([2.0, 4.0]),
        output_shape=(7, 7),
        origin=(-3.0, -3.0),
    )
    assert result.image.shape == (7, 7)
    assert np.all(np.isfinite(result.image[result.support]))
    assert result.contribution_count.max() >= 2


def test_gaussian_splat_grid_padding_does_not_change_kernel_support() -> None:
    result = gaussian_splat(
        np.array([[0.0, 0.0], [1.0, 0.0]]),
        np.array([2.0, 4.0]),
        fwhm=1.8,
        pixel_scale=1.0,
        grid_padding=0.0,
    )

    np.testing.assert_allclose(result.x_coordinates, [0.0, 1.0])
    np.testing.assert_allclose(result.y_coordinates, [0.0])
    assert result.contribution_count.max() == 2

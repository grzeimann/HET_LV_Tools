from hetquicklook.classification import (
    StandardStarCatalog,
    classify_exposure,
    parse_object_intent,
)


def test_object_intent_is_parsed_from_the_right() -> None:
    parsed = parse_object_intent("TARGET_WITH_UNDERSCORE_056_W")

    assert parsed.target == "TARGET_WITH_UNDERSCORE"
    assert parsed.requested_ifuslot == "056"
    assert parsed.track == "W"


def test_parallel_object_is_explicit() -> None:
    assert parse_object_intent(" parallel ").is_parallel


def test_flat_classification_uses_instrument_and_object() -> None:
    virus = classify_exposure(
        "virus", frame_types=("flt",), object_name=" LDLS_LONG ", ifu_slots=("074",)
    )
    lrs2_b = classify_exposure(
        "lrs2", frame_types=("flt",), object_name="ldls_long_B", ifu_slots=("056", "066")
    )
    lrs2_r = classify_exposure(
        "lrs2", frame_types=("flt",), object_name="Qth_R", ifu_slots=("066",)
    )

    assert (virus.quicklook_kind, virus.calibration_source) == ("flat", "LDLS")
    assert lrs2_b.applicable_ifu_slots == ("056",)
    assert (lrs2_r.quicklook_kind, lrs2_r.calibration_source) == ("flat", "Qth")


def test_flat_classification_does_not_fuzzy_match() -> None:
    result = classify_exposure(
        "virus", frame_types=("flt",), object_name="ldls", ifu_slots=("074",)
    )

    assert not result.known


def test_standard_classification_is_unknown_without_canonical_catalog() -> None:
    result = classify_exposure(
        "virus", frame_types=("sci",), object_name="HZ44_056_W"
    )

    assert result.standard_target == "HZ44"
    assert result.standard_star is None
    assert result.standard_catalog_available is False


def test_standard_classification_accepts_explicit_catalog() -> None:
    result = classify_exposure(
        "virus",
        frame_types=("sci",),
        object_name="HZ44_056_W",
        standard_catalog=StandardStarCatalog.from_names({"hz44"}),
    )

    assert result.standard_star is True

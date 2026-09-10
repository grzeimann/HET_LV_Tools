import pytest

from hetquicklook.discovery import parse_member_identity
from hetquicklook.instrument import (
    Instrument,
    lrs2_channel_for_identity,
    lrs2_component_for,
    physical_identity_for,
    topology_for,
)


def test_instrument_topologies_are_explicit() -> None:
    assert topology_for("VIRUS").levels == ("amplifier", "ifu", "instrument")
    assert topology_for(Instrument.LRS2).levels == (
        "amplifier",
        "channel",
        "spectrograph",
        "instrument",
    )


def test_unknown_instrument_is_rejected() -> None:
    with pytest.raises(ValueError):
        topology_for("unknown")


def test_lrs2_channel_interpretation_is_separate_from_raw_parsing() -> None:
    identity = parse_member_identity("20260511T035810.4_056LL_cmp.fits")

    assert identity is not None
    assert lrs2_channel_for_identity(identity) == "UV"


def test_lrs2_static_component_identity_is_explicit() -> None:
    component = lrs2_component_for("066", "RU")

    assert component is not None
    assert (component.name, component.specid, component.ifu_slot, component.ifuid) == (
        "Far-Red",
        "502",
        "066",
        "7002",
    )
    identity = physical_identity_for("lrs2", parse_member_identity("x_066RU_flt.fits"))
    assert (identity.specid, identity.ifuid, identity.amplifier) == ("502", "7002", "RU")
    assert identity.complete is False


def test_lrs2_all_amp_channel_pairs_are_fixed() -> None:
    expected = {
        ("056", "LL"): "UV",
        ("056", "LU"): "UV",
        ("056", "RL"): "Orange",
        ("056", "RU"): "Orange",
        ("066", "LL"): "Red",
        ("066", "LU"): "Red",
        ("066", "RL"): "Far-Red",
        ("066", "RU"): "Far-Red",
    }
    assert {pair: lrs2_channel_for_identity(parse_member_identity(f"x_{pair[0]}{pair[1]}_flt.fits")) for pair in expected} == expected


def test_virus_identity_uses_supported_header_fields() -> None:
    raw = parse_member_identity("x_074LL_cmp.fits")
    assert raw is not None
    identity = physical_identity_for(
        "virus",
        raw,
        {"IFUID": "043", "SPECID": 412, "CONTID": "S/N 0021"},
    )

    assert identity.key == "074+043+412+LL+S/N 0021"

import pytest

from hetquicklook.discovery import parse_member_identity
from hetquicklook.instrument import Instrument, lrs2_channel_for_identity, topology_for


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

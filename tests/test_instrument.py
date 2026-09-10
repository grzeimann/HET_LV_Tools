import pytest

from hetquicklook.instrument import Instrument, topology_for


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


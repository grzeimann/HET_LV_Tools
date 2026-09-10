from hetquicklook.metadata import exposure_metadata_from_headers


def test_exposure_metadata_keeps_header_disagreement_observable() -> None:
    metadata = exposure_metadata_from_headers(
        "20260910T010101.1",
        ("sci", "sci"),
        {
            "a.fits": {
                "OBJECT": "HZ44_056_W",
                "QOBJECT": "HZ44",
                "QRA": 10.0,
                "QDEC": -2.0,
                "QPROG": "P1",
                "EXPTIME": 10,
            },
            "b.fits": {
                "OBJECT": "HZ44_056_W",
                "QOBJECT": "HZ44",
                "QRA": 10.0,
                "QDEC": -2.0,
                "QPROG": "P2",
                "EXPTIME": 10,
            },
        },
    )

    assert metadata.frame_class == "science"
    assert metadata.object_target == "HZ44"
    disagreement = {item.field: item for item in metadata.disagreements}
    assert disagreement["qprog"].values == ("P1", "P2")
    assert metadata.q_metadata_complete is True

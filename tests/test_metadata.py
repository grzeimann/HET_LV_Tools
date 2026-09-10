from datetime import date, datetime
import io
import tarfile

from astropy.io import fits

from hetquicklook.discovery import DiscoveredObservation
from hetquicklook.metadata import metadata_from_archive, metadata_from_header


def test_metadata_extracts_common_header_aliases(tmp_path) -> None:
    observation = DiscoveredObservation(
        archive_path=tmp_path / "obs.tar",
        date=date(2026, 9, 10),
        instrument="virus",
    )

    metadata = metadata_from_header(
        observation,
        {
            "OBJNAME": "HD 123",
            "PROGRAM": "eng-test",
            "EXPTIME": 12.5,
            "REQRA": "123.4",
            "REQDEC": -4.5,
            "DATE-OBS": "2026-09-10T05:06:07Z",
        },
        files=("primary.fits",),
    )

    assert metadata.object_name == "HD 123"
    assert metadata.program == "eng-test"
    assert metadata.exposure_time_s == 12.5
    assert metadata.requested_ra_deg == 123.4
    assert metadata.requested_dec_deg == -4.5
    assert metadata.observation_time == datetime.fromisoformat("2026-09-10T05:06:07+00:00")
    assert metadata.files == ("primary.fits",)


def test_metadata_reads_the_first_fits_member_from_an_archive(tmp_path) -> None:
    observation_path = tmp_path / "obs.tar"
    fits_buffer = io.BytesIO()
    fits.PrimaryHDU().writeto(fits_buffer)
    fits_bytes = fits_buffer.getvalue()
    with tarfile.open(observation_path, mode="w") as archive:
        member = tarfile.TarInfo("primary.fits")
        member.size = len(fits_bytes)
        archive.addfile(member, io.BytesIO(fits_bytes))

    observation = DiscoveredObservation(
        archive_path=observation_path,
        date=date(2026, 9, 10),
        instrument="lrs2",
    )
    metadata = metadata_from_archive(observation)

    assert metadata.files == ("primary.fits",)
    assert metadata.instrument.value == "lrs2"

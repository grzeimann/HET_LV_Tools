import io
from pathlib import Path
import tarfile

import numpy as np
from astropy.io import fits

from hetquicklook import QuicklookConfig, discover_observations, load_observation


def _fits_bytes(data: np.ndarray, object_name: str) -> bytes:
    stream = io.BytesIO()
    hdu = fits.PrimaryHDU(data=data)
    hdu.header["OBJECT"] = object_name
    hdu.writeto(stream)
    return stream.getvalue()


def _add_member(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def test_observation_represents_partial_components_and_groups_exposures(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    archive_path = root / "20260910" / "lrs2" / "lrs20000001.tar"
    archive_path.parent.mkdir(parents=True)
    with tarfile.open(archive_path, mode="w") as archive:
        _add_member(
            archive,
            "lrs20000001/exp01/lrs2/20260910T010101.1_056LL_twi.fits",
            _fits_bytes(np.array([[1]], dtype=np.uint16), "skyflat"),
        )
        _add_member(
            archive,
            "lrs20000001/exp02/lrs2/20260910T010202.2_056LL_twi.fits",
            _fits_bytes(np.array([[2]], dtype=np.uint16), "skyflat"),
        )

    discovered = discover_observations(
        QuicklookConfig(root), instrument="lrs2", date="20260910"
    )[0]
    observation = load_observation(discovered)

    assert len(observation.frames) == 2
    assert observation.metadata.object_name is None
    assert observation.exposure_for("20260910T010101.1").metadata.object_name == "skyflat"
    assert observation.exposure_for("20260910T010202.2").metadata.object_name == "skyflat"
    assert observation.metadata.exposure_time_s is None
    assert observation.metadata.requested_ra_deg is None
    assert set(observation.group_by_exposure()) == {
        "20260910T010101.1",
        "20260910T010202.2",
    }
    assert observation.frames_for(frame_type="twi")[0].identity is not None
    loaded = observation.load_frames(exposure_id="20260910T010202.2")
    np.testing.assert_array_equal(loaded[0].data, np.array([[2]], dtype=np.uint16))

import textwrap

import pytest

from app.config import load_cameras
from app.ingest.tagger import build_tags

SCENE_CONFIG = """
    scenes:
      - scene_name: storewide loss prevention
        cameras:
          - lp-camera1
          - lp-camera2
        zones:
          aisle1: HIGH_VALUE
          aisle2: CHECKOUT
    """


def _write(tmp_path, text):
    p = tmp_path / "scene-config.yaml"
    p.write_text(textwrap.dedent(text))
    return str(p)


def test_cameras_derived_from_scene_config(tmp_path):
    path = _write(tmp_path, SCENE_CONFIG)
    cams = load_cameras(path, rtsp_base_url="rtsp://mediaserver:8554", store_id="store-001")

    assert [c.camera_id for c in cams] == ["lp-camera1", "lp-camera2"]
    cam = cams[0]
    assert cam.area_label == "lp-camera1"
    assert cam.rtsp_url == "rtsp://mediaserver:8554/lp-camera1"
    assert cam.store_id == "store-001"
    assert cam.enabled is True


def test_tags_include_zone_names_only(tmp_path):
    path = _write(tmp_path, SCENE_CONFIG)
    cam = load_cameras(path)[0]
    # Only zone names (aisle1, aisle2) become tags; zone types are ignored.
    assert cam.extra_tags == ["aisle1", "aisle2"]


def test_build_tags_dedupes_name(tmp_path):
    path = _write(tmp_path, SCENE_CONFIG)
    cam = load_cameras(path, store_id="store-001")[0]
    # camera_id == area_label == lp-camera1, so it appears once.
    assert build_tags(cam) == "lp-camera1,store-001,aisle1,aisle2"


def test_rtsp_base_url_trailing_slash(tmp_path):
    path = _write(tmp_path, SCENE_CONFIG)
    cam = load_cameras(path, rtsp_base_url="rtsp://host:8554/")[0]
    assert cam.rtsp_url == "rtsp://host:8554/lp-camera1"


def test_blank_store_id_omitted(tmp_path):
    path = _write(tmp_path, SCENE_CONFIG)
    cam = load_cameras(path, store_id="")[0]
    assert cam.store_id is None


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cameras(str(tmp_path / "nope.yaml"))


def test_no_cameras_raises(tmp_path):
    path = _write(tmp_path, "scenes: []\n")
    with pytest.raises(ValueError, match="no cameras"):
        load_cameras(path)

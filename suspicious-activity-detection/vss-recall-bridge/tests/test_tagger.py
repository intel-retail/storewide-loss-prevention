from app.config import Camera
from app.ingest.tagger import build_tags


def _camera(**kwargs):
    base = dict(
        camera_id="cam1",
        area_label="lp-camera1",
        store_id="store-001",
        enabled=False,  # tagging is independent of enabled; avoids the source validator
    )
    base.update(kwargs)
    return Camera(**base)


def test_build_tags_basic_order():
    cam = _camera()
    assert build_tags(cam) == "cam1,lp-camera1,store-001"


def test_build_tags_includes_extra_tags():
    cam = _camera(extra_tags=["front-of-store", "aisle-7"])
    assert build_tags(cam) == "cam1,lp-camera1,store-001,front-of-store,aisle-7"


def test_build_tags_omits_missing_store_id():
    cam = _camera(store_id=None)
    assert build_tags(cam) == "cam1,lp-camera1"

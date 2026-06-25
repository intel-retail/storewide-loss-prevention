from datetime import datetime, timezone

from app.query.routes import build_tag_query, pad_window, to_hit


def test_build_tag_query_none_when_empty():
    assert build_tag_query(None) is None
    assert build_tag_query([]) is None


def test_build_tag_query_comma_joined():
    assert build_tag_query(["entrance", "cam-12"]) == "entrance,cam-12"


def test_pad_window_widens_both_ends():
    start = datetime(2026, 6, 25, 14, 0, tzinfo=timezone.utc)
    end = datetime(2026, 6, 25, 15, 0, tzinfo=timezone.utc)
    padded_start, padded_end = pad_window(start, end, 60)
    assert padded_start == datetime(2026, 6, 25, 13, 59, tzinfo=timezone.utc)
    assert padded_end == datetime(2026, 6, 25, 15, 1, tzinfo=timezone.utc)


def test_pad_window_none_passthrough():
    assert pad_window(None, None, 60) == (None, None)


def test_to_hit_maps_vss_metadata():
    meta = {
        "video_id": "abc123",
        "tags": "cam1,lp-camera1,store-001",
        "created_at": "2026-06-25T04:27:41+00:00",
        "segment_start": 32.0,
        "segment_end": 40.0,
        "relevance_score": 0.896,
        "video_url": "http://minio/clip.mp4",
    }
    hit = to_hit(meta)
    assert hit.video_id == "abc123"
    assert hit.tags == ["cam1", "lp-camera1", "store-001"]
    assert hit.segment_start == 32.0
    assert hit.segment_end == 40.0
    assert hit.score == 0.896
    assert hit.video_url == "http://minio/clip.mp4"


def test_in_video_position_filter_overlap():
    # mirrors routes.search() overlap test: keep if start < hi and end > lo
    hits = [
        {"segment_start": 0.0, "segment_end": 8.0},
        {"segment_start": 32.0, "segment_end": 40.0},
        {"segment_start": 120.0, "segment_end": 128.0},
    ]
    lo, hi = 60.0, 180.0
    kept = [h for h in hits if h["segment_start"] < hi and h["segment_end"] > lo]
    assert kept == [{"segment_start": 120.0, "segment_end": 128.0}]

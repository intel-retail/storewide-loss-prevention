from app.query.routes import build_tag_query, to_hit
from app.models import RecallSearchRequest


def test_build_tag_query_none_when_empty():
    assert build_tag_query(None) is None
    assert build_tag_query([]) is None


def test_build_tag_query_comma_joined():
    assert build_tag_query(["entrance", "cam-12"]) == "entrance,cam-12"


def test_build_tag_query_single_tag():
    assert build_tag_query(["cam-12"]) == "cam-12"


def test_cameras_accepts_comma_separated_string():
    req = RecallSearchRequest(query="x", cameras="lp-camera1, cam-2 ,cam-3")
    assert req.cameras == ["lp-camera1", "cam-2", "cam-3"]


def test_cameras_still_accepts_list():
    req = RecallSearchRequest(query="x", cameras=["lp-camera1", "cam-2"])
    assert req.cameras == ["lp-camera1", "cam-2"]


def test_cameras_list_trims_whitespace_and_drops_empty():
    req = RecallSearchRequest(query="x", cameras=[" lp-camera1 ", "  ", "cam-2"])
    assert req.cameras == ["lp-camera1", "cam-2"]


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


def test_dedup_keeps_best_segment_per_video():
    # mirrors routes.search() dedup: after sort-by-score, keep first per video_id
    results = sorted(
        [
            to_hit({"video_id": "a", "segment_start": 0, "segment_end": 8, "relevance_score": 0.87}),
            to_hit({"video_id": "b", "segment_start": 8, "segment_end": 16, "relevance_score": 1.0}),
            to_hit({"video_id": "a", "segment_start": 24, "segment_end": 32, "relevance_score": 0.86}),
        ],
        key=lambda h: h.score,
        reverse=True,
    )
    deduped = []
    seen = set()
    for hit in results:
        if hit.video_id and hit.video_id in seen:
            continue
        seen.add(hit.video_id)
        deduped.append(hit)
    assert [h.video_id for h in deduped] == ["b", "a"]
    # kept the higher-scoring segment (0-8, 0.87) for video "a", not 0.86
    assert next(h for h in deduped if h.video_id == "a").segment_start == 0


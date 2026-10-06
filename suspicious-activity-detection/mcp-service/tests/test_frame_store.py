from __future__ import annotations

from frame_store import SeaweedFrameStore


class FakePaginator:
    def __init__(self):
        self.calls = []

    def paginate(self, **kwargs):
        self.calls.append(kwargs)
        return [{
            "Contents": [
                {"Key": "person-1/alert-1/frames/200.jpg"},
                {"Key": "person-1/alert-1/frames/100.jpg"},
                {"Key": "person-1/alert-1/frames/ignore.txt"},
            ]
        }]


class FakeS3:
    def __init__(self):
        self.paginator = FakePaginator()

    def get_paginator(self, operation):
        assert operation == "list_objects_v2"
        return self.paginator


def test_frame_lookup_uses_alert_prefix_and_returns_sorted_s3_refs():
    client = FakeS3()
    frames = SeaweedFrameStore("http://seaweedfs:8333", client=client)

    result = frames.find_alert_frames("person-1", "alert-1")

    assert client.paginator.calls == [{
        "Bucket": "alerts",
        "Prefix": "person-1/alert-1/frames/",
    }]
    assert result == [
        "s3://alerts/person-1/alert-1/frames/100.jpg",
        "s3://alerts/person-1/alert-1/frames/200.jpg",
    ]


def test_frame_lookup_without_event_ids_returns_empty():
    frames = SeaweedFrameStore("http://seaweedfs:8333", client=FakeS3())
    assert frames.find_alert_frames("", "alert-1") == []
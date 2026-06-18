"""Shared fixtures for POI backend tests."""

from __future__ import annotations

import os
import threading
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Ensure config singleton is reset and uses test defaults
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("FAISS_INDEX_PATH", "/tmp/test_poi.index")
os.environ.setdefault("FAISS_ID_MAP_PATH", "/tmp/test_id_map.json")


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Reset singletons between tests to prevent state leaks."""
    from backend.core.config import Config

    Config.reset()
    yield
    Config.reset()


@pytest.fixture
def config():
    from backend.core.config import Config

    Config.reset()
    cfg = Config.get_instance()
    return cfg


@pytest.fixture
def random_embedding():
    """Return a random normalized 256-d float32 vector."""
    v = np.random.randn(256).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


@pytest.fixture
def sample_poi_dict():
    return {
        "event_type": "poi_enrollment",
        "timestamp": "2025-01-15T10:00:00Z",
        "poi_id": "poi-abc12345",
        "enrolled_by": "system",
        "severity": "high",
        "notes": "Test POI",
        "reference_images": [
            {
                "source": "uploaded_image",
                "embedding_id": "emb-poi-abc12345-ref-00",
                "vector_dim": 256,
                "image_path": "/uploads/poi-abc12345/ref_0.jpg",
            }
        ],
        "status": "active",
    }


@pytest.fixture
def mock_redis():
    """Mock Redis client that behaves like a dict-based store."""
    r = MagicMock()
    store = {}
    sets = {}

    def _get(key):
        return store.get(key)

    def _set(key, value):
        store[key] = value

    def _setex(key, ttl, value):
        store[key] = value

    def _delete(key):
        if key in store:
            del store[key]
            return 1
        return 0

    def _exists(key):
        return 1 if key in store else 0

    def _sadd(key, *members):
        if key not in sets:
            sets[key] = set()
        sets[key].update(members)

    def _smembers(key):
        return sets.get(key, set())

    def _srem(key, *members):
        if key in sets:
            sets[key] -= set(members)

    def _lpush(key, *values):
        if key not in store:
            store[key] = []
        for v in values:
            store[key].insert(0, v)

    def _ltrim(key, start, end):
        if key in store and isinstance(store[key], list):
            store[key] = store[key][start : end + 1]

    def _lrange(key, start, end):
        if key not in store or not isinstance(store[key], list):
            return []
        return store[key][start : end + 1]

    def _expire(key, ttl):
        pass

    def _pipeline():
        pipe = MagicMock()
        cmds = []

        def _pipe_delete(key):
            cmds.append(("delete", key))

        def _pipe_execute():
            for cmd, key in cmds:
                if cmd == "delete":
                    _delete(key)

        pipe.delete = _pipe_delete
        pipe.execute = _pipe_execute
        return pipe

    r.get = MagicMock(side_effect=_get)
    r.set = MagicMock(side_effect=_set)
    r.setex = MagicMock(side_effect=_setex)
    r.delete = MagicMock(side_effect=_delete)
    r.exists = MagicMock(side_effect=_exists)
    r.sadd = MagicMock(side_effect=_sadd)
    r.smembers = MagicMock(side_effect=_smembers)
    r.srem = MagicMock(side_effect=_srem)
    r.lpush = MagicMock(side_effect=_lpush)
    r.ltrim = MagicMock(side_effect=_ltrim)
    r.lrange = MagicMock(side_effect=_lrange)
    r.expire = MagicMock(side_effect=_expire)
    r.pipeline = MagicMock(side_effect=_pipeline)

    return r


@pytest.fixture
def mock_event_payload():
    """Sample MQTT scene event payload matching Scenescape format."""
    return {
        "id": "bfb9f86b-b152-4e7f-8099-7c251ed84630",
        "timestamp": "2025-01-15T12:30:00.000Z",
        "name": "storewide loss prevention",
        "objects": [
            {
                "id": "18f951ba-0409-4238-9ce5-4150054a31f7",
                "category": "person",
                "confidence": 0.95,
                "center_of_mass": {"x": 320, "y": 240, "width": 80, "height": 160},
                "visibility": ["camera-01"],
                "metadata": {
                    "reid": {
                        "embedding_vector": [np.random.randn(256).tolist()],
                    }
                },
            }
        ],
        "entered": [],
    }

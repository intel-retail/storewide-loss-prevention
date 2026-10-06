"""Unit tests for env/config helpers."""

from __future__ import annotations

import json

from config import configured_zones


def test_configured_zones_from_usecase_config(tmp_path):
    path = tmp_path / "zone_config.json"
    path.write_text(json.dumps({"zones": {"kitchen-prep": "HIGH_VALUE"}}), encoding="utf-8")

    assert configured_zones(str(path)) == ["kitchen-prep"]


def test_configured_zones_missing_file():
    assert configured_zones("/tmp/does-not-exist-zone-config.json") == []
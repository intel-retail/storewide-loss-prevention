"""Settings (env) and scene-config.yaml -> camera loader.

The bridge is stateless; this module only loads configuration. Cameras and their
region tags are derived entirely from SceneScape's scene-config.yaml, so an operator
adds a camera in one place (SceneScape) and the bridge picks it up automatically.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment configuration (build-spec §3.1)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    vss_base_url: str = "http://pipeline-manager:3000"
    dataprep_base_url: str = "http://vdms-dataprep:8000"
    scene_config: str = "configs/scene-config.yaml"
    rtsp_base_url: str = "rtsp://mediaserver:8554"
    store_id: str = ""
    clips_dir: str = "./clips"
    bridge_api_key: str = ""
    search_poll_seconds: float = 1.0
    search_poll_timeout_seconds: float = 30.0
    default_segment_seconds: int = 60
    search_timezone: str = "UTC"
    http_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def _zone_tags(zones: dict | None) -> list[str]:
    """Flatten a scene's zone map into tags: the zone names only.

    e.g. ``{aisle1: HIGH_VALUE, aisle2: CHECKOUT}`` -> ``[aisle1, aisle2]``,
    de-duplicated while preserving order. Zone types are intentionally ignored.
    """

    tags: list[str] = []
    for name in (zones or {}).keys():
        if name:
            tags.append(str(name))
    return list(dict.fromkeys(tags))


class Camera(BaseModel):
    """A camera derived from SceneScape's scene-config.yaml.

    ``camera_id``/``area_label`` are the SceneScape camera name, ``rtsp_url`` is built
    from that name, and ``extra_tags`` are the scene's zone names and types.
    """

    camera_id: str
    rtsp_url: str | None = None
    source_file: str | None = None
    area_label: str
    store_id: str | None = None
    enabled: bool = True
    segment_seconds: int = 60
    extra_tags: list[str] = []


def load_cameras(
    scene_config: str,
    rtsp_base_url: str = "rtsp://mediaserver:8554",
    store_id: str | None = None,
    default_segment_seconds: int = 60,
) -> list[Camera]:
    """Build the camera list from SceneScape's scene-config.yaml.

    One camera per name under ``scenes[].cameras``. Its RTSP source is
    ``{rtsp_base_url}/{camera_name}`` and its tags are the scene's zone names + types.
    Adding a camera in SceneScape is therefore the only step needed to ingest it.
    """

    p = Path(scene_config)
    if not p.exists():
        raise FileNotFoundError(f"scene config not found: {scene_config}")

    data = yaml.safe_load(p.read_text()) or {}
    base = rtsp_base_url.rstrip("/")

    cameras: list[Camera] = []
    seen: set[str] = set()
    for scene in data.get("scenes") or []:
        tags = _zone_tags(scene.get("zones"))
        for raw_name in scene.get("cameras") or []:
            name = str(raw_name)
            if name in seen:
                continue
            seen.add(name)
            cameras.append(
                Camera(
                    camera_id=name,
                    area_label=name,
                    rtsp_url=f"{base}/{name}",
                    store_id=store_id or None,
                    enabled=True,
                    segment_seconds=default_segment_seconds,
                    extra_tags=tags,
                )
            )

    if not cameras:
        raise ValueError(f"no cameras found under 'scenes' in {scene_config}")

    return cameras

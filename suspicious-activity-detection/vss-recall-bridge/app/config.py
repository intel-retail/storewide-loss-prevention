"""Settings (env) and cameras.yaml loader.

The bridge is stateless; this module only loads configuration. See build-spec §3.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment configuration (build-spec §3.1)."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    vss_base_url: str = "http://pipeline-manager:3000"
    dataprep_base_url: str = "http://vdms-dataprep:8000"
    cameras_config: str = "configs/cameras.yaml"
    clips_dir: str = "./clips"
    bridge_api_key: str = ""
    search_poll_seconds: float = 1.0
    search_poll_timeout_seconds: float = 30.0
    default_segment_seconds: int = 60
    window_pad_seconds: float = 0.0
    search_timezone: str = "UTC"
    http_timeout_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


class Camera(BaseModel):
    """A single camera entry from cameras.yaml (design doc §10.1)."""

    camera_id: str
    rtsp_url: str | None = None
    source_file: str | None = None
    area_label: str
    store_id: str | None = None
    enabled: bool = True
    segment_seconds: int = 60
    extra_tags: list[str] = []

    @model_validator(mode="after")
    def _require_a_source(self) -> "Camera":
        if self.enabled and not self.rtsp_url and not self.source_file:
            raise ValueError(
                f"camera '{self.camera_id}' has neither rtsp_url nor source_file"
            )
        return self


def load_cameras(path: str, default_segment_seconds: int = 60) -> list[Camera]:
    """Parse cameras.yaml and fail fast on misconfiguration (build-spec §3.2)."""

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"cameras config not found: {path}")

    data = yaml.safe_load(p.read_text()) or {}
    raw = data.get("cameras") or {}

    cameras: list[Camera] = []
    for camera_id, fields in raw.items():
        fields = dict(fields or {})
        fields.setdefault("segment_seconds", default_segment_seconds)
        cam = Camera(camera_id=camera_id, **fields)

        # Fail fast: an enabled file-ingest camera must point at a real file.
        if cam.enabled and not cam.rtsp_url and cam.source_file:
            if not Path(cam.source_file).exists():
                raise FileNotFoundError(
                    f"camera '{camera_id}' source_file does not exist: {cam.source_file}"
                )
        cameras.append(cam)

    return cameras

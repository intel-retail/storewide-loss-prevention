"""Camera -> tag list (build-spec §6.4, design doc §7)."""

from __future__ import annotations

from ..config import Camera


def build_tags(camera: Camera) -> str:
    """Return a comma-separated tag string for a clip from this camera.

    Order: camera_id, area_label, store_id, *extra_tags. Empty/None and duplicate
    parts dropped (camera_id and area_label are usually the same SceneScape name).
    VSS matches tags with subset/OR semantics (design doc §3.3).
    """

    parts: list[str | None] = [
        camera.camera_id,
        camera.area_label,
        camera.store_id,
        *camera.extra_tags,
    ]
    return ",".join(dict.fromkeys(p for p in parts if p))

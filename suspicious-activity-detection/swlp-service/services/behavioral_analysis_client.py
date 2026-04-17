# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Behavioral Analysis Client — calls external BehavioralAnalysis Service via HTTP.

The BehavioralAnalysis Service (separate container) handles:
  - Pose analysis (shelf-to-waist gesture detection via YOLO-Pose)

The BA service fetches frames directly from SeaweedFS (behavioral-frames bucket).
This client only sends entity_id + pattern_id.
"""

from typing import Any, Dict, Optional

import aiohttp
import structlog

from .config import ConfigService

logger = structlog.get_logger(__name__)


class BehavioralAnalysisClient:
    """
    HTTP client for the external BehavioralAnalysis Service.

    The BA service reads frames from SeaweedFS on its own —
    this client only sends entity_id and pattern_id.
    """

    def __init__(self, config: ConfigService) -> None:
        ba_cfg = config.get_behavioral_analysis_config()
        self.base_url = ba_cfg.get("base_url", "http://behavioral-analysis:8080")
        self.timeout = ba_cfg.get("timeout_seconds", 30)
        self.enabled = ba_cfg.get("enabled", True)

        logger.info(
            "BehavioralAnalysisClient initialized",
            base_url=self.base_url,
            enabled=self.enabled,
        )

    async def analyze(
        self,
        entity_id: str,
        pattern_id: str = "shelf_to_waist",
        region_id: Optional[str] = None,
        entry_timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Ask the BehavioralAnalysis Service to analyze frames for an entity.

        The service fetches frames from SeaweedFS itself.

        Returns:
            {
                "entity_id": str,
                "status": "no_data" | "accumulating" | "no_match" | "suspicious",
                "frames_available": int,
                "frames_required": int,
                "confidence": float | None,
                "pattern_id": str | None,
                "message": str | None,
            }
            or None on failure.
        """
        if not self.enabled:
            return None

        payload = {
            "entity_id": entity_id,
            "pattern_id": pattern_id,
        }
        if region_id:
            payload["region_id"] = region_id
        if entry_timestamp:
            payload["entry_timestamp"] = entry_timestamp

        return await self._post("/api/v1/analyze", payload)

    async def delete_frames(self, entity_id: str, region_id: Optional[str] = None) -> bool:
        """Tell the BA service to clear stored frames for an entity."""
        url = f"/api/v1/entities/{entity_id}/frames"
        if region_id:
            url += f"?region_id={region_id}"
        return await self._delete(url)

    async def health_check(self) -> bool:
        """Check if the BehavioralAnalysis service is available."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.base_url}/health",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    return resp.status == 200
        except Exception:
            return False

    # ---- internal ------------------------------------------------------------
    async def _post(self, path: str, payload: dict) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        body = await resp.text()
                        logger.error(
                            "BehavioralAnalysis request failed",
                            path=path,
                            status=resp.status,
                            body=body[:200],
                        )
                        return None
        except aiohttp.ClientError as e:
            logger.error("BehavioralAnalysis connection error", path=path, error=str(e))
            return None
        except Exception:
            logger.exception("BehavioralAnalysis call error", path=path)
            return None

    async def _delete(self, path: str) -> bool:
        url = f"{self.base_url}{path}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    url,
                    timeout=aiohttp.ClientTimeout(total=self.timeout),
                ) as resp:
                    return resp.status == 200
        except Exception:
            logger.exception("BehavioralAnalysis DELETE error", path=path)
            return False

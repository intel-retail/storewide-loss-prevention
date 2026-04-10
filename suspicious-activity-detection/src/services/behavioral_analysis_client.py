# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Behavioral Analysis Client — calls external BehavioralAnalysis Service via HTTP.

The BehavioralAnalysis Service (separate container) handles:
  - Pose analysis (shelf-to-waist gesture detection)
  - VLM-based concealment confirmation

This client sends frame data / references and receives analysis results.
Called conditionally when a person is in a HIGH_VALUE zone.
"""

from typing import Any, Dict, List, Optional

import aiohttp
import structlog

from .config import ConfigService

logger = structlog.get_logger(__name__)


class BehavioralAnalysisClient:
    """
    HTTP client for the external BehavioralAnalysis Service.

    Sends cropped person frames for analysis. The behavioral service
    owns all analysis logic (pose detection, VLM escalation, etc.).
    """

    def __init__(self, config: ConfigService) -> None:
        ba_cfg = config.get_behavioral_analysis_config()
        self.base_url = ba_cfg.get("base_url", "http://behavioral-analysis-service:8090")
        self.timeout = ba_cfg.get("timeout_seconds", 30)
        self.enabled = ba_cfg.get("enabled", True)

        logger.info(
            "BehavioralAnalysisClient initialized",
            base_url=self.base_url,
            enabled=self.enabled,
        )

    async def analyze(
        self,
        object_id: str,
        frame_keys: List[str],
        frames_base64: List[str],
        zone_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Send frames to the BehavioralAnalysis Service for analysis.

        The service handles pose detection, VLM escalation, etc. internally.

        Returns:
            {
                "concealment_suspected": bool,
                "confidence": float,
                "observation": str,
                ...
            }
            or None on failure.
        """
        if not self.enabled:
            return None

        payload = {
            "object_id": object_id,
            "frame_keys": frame_keys,
            "frames": frames_base64,
            "zone_info": zone_info or {},
        }

        return await self._post("/api/v1/analyze", payload)

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

# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""
Generic VLM Client for OpenVINO Model Server.

Sends images + configurable prompts to a VLM endpoint and parses
structured JSON responses. Knows nothing about specific behaviors —
the prompt and response schema are fully driven by configuration.
"""

import base64
import json
import logging
from dataclasses import dataclass
from typing import Any, Optional

import httpx
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VLMResult:
    """Result from VLM analysis."""

    raw_response: str
    parsed: Optional[dict[str, Any]] = None
    success: bool = False
    error: Optional[str] = None


class VLMClient:
    """
    Async client for OpenAI-compatible VLM endpoints (OVMS).

    Usage:
        client = VLMClient(endpoint="http://ovms-vlm:8000", model_name="Qwen/...")
        result = await client.analyze(frames, prompt="Describe what you see.")
    """

    def __init__(
        self,
        endpoint: str = "http://ovms-vlm:8000",
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        timeout: float = 60.0,
        max_tokens: int = 500,
        temperature: float = 0.1,
        max_image_size: int = 512,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_image_size = max_image_size

    def _encode_frame(self, frame: np.ndarray) -> str:
        """Resize and encode a frame as base64 JPEG."""
        h, w = frame.shape[:2]
        if max(h, w) > self.max_image_size:
            scale = self.max_image_size / max(h, w)
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    def _build_messages(
        self,
        frames: list[np.ndarray],
        prompt: str,
    ) -> list[dict]:
        """Build OpenAI-compatible chat messages with images."""
        content = []

        # Add images
        for frame in frames:
            b64 = self._encode_frame(frame)
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                }
            )

        # Add text prompt
        content.append({"type": "text", "text": prompt})

        return [{"role": "user", "content": content}]

    async def analyze(
        self,
        frames: list[np.ndarray],
        prompt: str,
    ) -> VLMResult:
        """
        Send frames and prompt to VLM and return parsed result.

        Args:
            frames: List of BGR numpy arrays (images)
            prompt: The analysis prompt to send

        Returns:
            VLMResult with parsed JSON response
        """
        if not frames:
            return VLMResult(
                raw_response="",
                success=False,
                error="No frames provided",
            )

        messages = self._build_messages(frames, prompt)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        url = f"{self.endpoint}/v3/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()

            data = response.json()
            raw_text = data["choices"][0]["message"]["content"]

            logger.debug(f"VLM raw response: {raw_text}")

            # Try to parse as JSON
            parsed = self._parse_json_response(raw_text)

            return VLMResult(
                raw_response=raw_text,
                parsed=parsed,
                success=parsed is not None,
            )

        except httpx.TimeoutException:
            logger.warning("VLM request timed out")
            return VLMResult(
                raw_response="",
                success=False,
                error="VLM request timed out",
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"VLM HTTP error: {e.response.status_code}")
            return VLMResult(
                raw_response="",
                success=False,
                error=f"HTTP {e.response.status_code}",
            )
        except Exception as e:
            logger.error(f"VLM request failed: {e}")
            return VLMResult(
                raw_response="",
                success=False,
                error=str(e),
            )

    @staticmethod
    def _parse_json_response(text: str) -> Optional[dict]:
        """
        Extract and parse JSON from VLM response text.

        Handles responses that may have markdown code blocks or
        extra text around the JSON.
        """
        text = text.strip()

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        if "```" in text:
            for block in text.split("```"):
                block = block.strip()
                if block.startswith("json"):
                    block = block[4:].strip()
                try:
                    return json.loads(block)
                except json.JSONDecodeError:
                    continue

        # Try finding JSON object in text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass

        logger.warning(f"Could not parse JSON from VLM response: {text[:200]}")
        return None

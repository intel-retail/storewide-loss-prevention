"""Async httpx client for the VSS pipeline-manager `/manager/*` API (build-spec §5).

The bridge keeps no local state: it uploads clips, triggers embedding, and runs the
stateful search (submit + poll). Transient 5xx / connection errors are retried.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_TERMINAL_STATES = {
    "completed",
    "complete",
    "success",
    "done",
    "finished",
    "ready",
}


def _is_retryable(exc: BaseException) -> bool:
    """Retry on connection errors and 5xx responses only (not 4xx)."""

    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return False


_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, max=8),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)


def _iso(dt: datetime) -> str:
    """Emit a naive UTC ISO string for VSS (which treats timeFilter as UTC).

    Timezone-aware inputs are converted to UTC; naive inputs are assumed UTC.
    """

    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.isoformat()


def _hit_metadata(item: Any) -> dict | None:
    if not isinstance(item, dict):
        return None
    meta = item.get("metadata")
    return meta if isinstance(meta, dict) else item


def _extract_hits(payload: Any) -> list[dict]:
    """Defensively flatten the polled search response into a list of metadata dicts.

    Handles both a flat ``{"results": [...]}`` and the nested
    ``{"results": [{"results": [...]}]}`` shape VSS can return.
    """

    results = payload.get("results") if isinstance(payload, dict) else payload
    if not results:
        return []

    hits: list[dict] = []
    for item in results:
        if isinstance(item, dict) and isinstance(item.get("results"), list):
            for inner in item["results"]:
                meta = _hit_metadata(inner)
                if meta:
                    hits.append(meta)
        else:
            meta = _hit_metadata(item)
            if meta:
                hits.append(meta)
    return hits


class VssClient:
    def __init__(
        self,
        base_url: str,
        timeout: float,
        poll_seconds: float,
        poll_timeout: float,
        dataprep_base_url: str = "http://vdms-dataprep:8000",
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._dataprep_base_url = dataprep_base_url.rstrip("/")
        self._poll_seconds = poll_seconds
        self._poll_timeout = poll_timeout
        self._client = httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._client.aclose()

    # --- ingest -----------------------------------------------------------------

    @_retry
    async def upload(self, *, file_path: str, name: str, tags: str) -> str:
        """POST {base_url}/videos (multipart) -> videoId."""

        with open(file_path, "rb") as fh:
            files = {"video": (os.path.basename(file_path), fh, "video/mp4")}
            data = {"name": name, "tags": tags}
            resp = await self._client.post(
                f"{self._base_url}/videos", files=files, data=data
            )
        resp.raise_for_status()
        body = resp.json()
        return body.get("videoId") or body.get("video_id")

    @_retry
    async def trigger_embedding(self, video_id: str) -> None:
        """POST {base_url}/videos/search-embeddings/{videoId}."""

        resp = await self._client.post(
            f"{self._base_url}/videos/search-embeddings/{video_id}"
        )
        resp.raise_for_status()

    # --- search (stateful submit + poll) ----------------------------------------

    async def search(
        self,
        *,
        query: str,
        tags: str | None,
        time_start: datetime | None,
        time_end: datetime | None,
    ) -> list[dict]:
        body: dict[str, Any] = {"query": query}
        if tags:
            body["tags"] = tags
        time_filter: dict[str, str] = {}
        if time_start is not None:
            time_filter["start"] = _iso(time_start)
        if time_end is not None:
            time_filter["end"] = _iso(time_end)
        if time_filter:
            body["timeFilter"] = time_filter

        query_id = await self._submit_search(body)
        return await self._poll_search(query_id)

    @_retry
    async def _submit_search(self, body: dict) -> str:
        resp = await self._client.post(f"{self._base_url}/search", json=body)
        resp.raise_for_status()
        data = resp.json()
        query_id = data.get("queryId") or data.get("query_id")
        if not query_id:
            raise RuntimeError(f"VSS search submit returned no queryId: {data!r}")
        return query_id

    async def _poll_search(self, query_id: str) -> list[dict]:
        deadline = time.monotonic() + self._poll_timeout
        last: list[dict] = []
        while time.monotonic() < deadline:
            payload = await self._get_search(query_id)
            state = ""
            if isinstance(payload, dict):
                state = str(payload.get("status") or payload.get("state") or "").lower()
            hits = _extract_hits(payload)
            if hits:
                return hits
            if state in _TERMINAL_STATES:
                return hits  # terminal, legitimately empty
            last = hits
            await asyncio.sleep(self._poll_seconds)
        logger.warning("search %s timed out after %ss", query_id, self._poll_timeout)
        return last

    @_retry
    async def _get_search(self, query_id: str) -> Any:
        resp = await self._client.get(f"{self._base_url}/search/{query_id}")
        resp.raise_for_status()
        return resp.json()

    # --- clip playback (proxy) --------------------------------------------------

    async def _resolve_clip_url(self, video_id: str) -> str:
        """Resolve the streamable download URL for a clip.

        VSS returns clip metadata from ``GET {base_url}/videos/{videoId}``. The
        pipeline-manager response wraps the entity under a ``video`` key and does
        not carry a direct ``video_url``; the bytes are served by the dataprep
        download endpoint, keyed by ``video_id`` + storage ``bucket``.
        """

        meta_resp = await self._client.get(f"{self._base_url}/videos/{video_id}")
        meta_resp.raise_for_status()
        body = meta_resp.json()
        meta = body.get("video", body) if isinstance(body, dict) else {}

        video_url = meta.get("video_url") or meta.get("videoUrl")
        if not video_url:
            data_store = meta.get("dataStore") or {}
            bucket = data_store.get("bucket")
            if bucket:
                video_url = (
                    f"{self._dataprep_base_url}/v1/dataprep/videos/download"
                    f"?video_id={video_id}&bucket_name={bucket}"
                )
        if not video_url:
            raise RuntimeError(f"no video_url for {video_id}: {meta!r}")
        return video_url

    async def fetch_clip(self, video_id: str) -> bytes:
        """Return the full clip bytes (upstream lacks HTTP Range support, so the
        whole small clip is buffered to enable seekable, range-capable serving)."""

        video_url = await self._resolve_clip_url(video_id)
        resp = await self._client.get(video_url)
        resp.raise_for_status()
        return resp.content

    async def iter_clip(self, video_id: str) -> AsyncIterator[bytes]:
        """Stream a clip's bytes for authenticated playback proxying (build-spec §8)."""

        video_url = await self._resolve_clip_url(video_id)
        async with self._client.stream("GET", video_url) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes():
                yield chunk

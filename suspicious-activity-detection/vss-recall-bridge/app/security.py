"""API-key auth dependency for the /api/v1/lp/recall/* routes (build-spec §8)."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from .config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Reject any request lacking a valid X-API-Key header."""

    settings = get_settings()
    if not settings.bridge_api_key or x_api_key != settings.bridge_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or missing API key",
        )

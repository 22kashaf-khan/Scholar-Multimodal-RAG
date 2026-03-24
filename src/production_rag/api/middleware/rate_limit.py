"""Redis-backed per-user/IP rate limiting via slowapi."""

from __future__ import annotations

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from production_rag.core.config import get_settings


def _get_user_or_ip(request: Request) -> str:
    """Rate limit key: JWT sub if authenticated, else IP address."""
    sub = getattr(getattr(request, "state", None), "sub", None)
    if sub:
        return sub
    return get_remote_address(request)


def create_limiter(redis_url: str | None = None) -> Limiter:
    """Create a Redis-backed rate limiter."""
    url = redis_url or get_settings().redis_url
    return Limiter(
        key_func=_get_user_or_ip,
        storage_uri=url,
    )

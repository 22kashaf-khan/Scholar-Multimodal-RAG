"""Health check router.

GET /health — liveness probe (always 200 if app is running)
GET /health/ready — readiness probe (checks Weaviate + Redis connectivity)
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(prefix="/health", tags=["health"])


class HealthResponse(BaseModel):
    status: str
    weaviate: str = "unknown"
    redis: str = "unknown"


@router.get("", response_model=HealthResponse)
async def liveness() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/live", response_model=HealthResponse)
async def liveness_alias() -> HealthResponse:
    """Alias for /health — used by Dockerfile HEALTHCHECK and Helm probes."""
    return HealthResponse(status="ok")


@router.get("/ready", response_model=HealthResponse)
async def readiness(request: Request) -> HealthResponse:
    weaviate_ok = "ok"
    redis_ok = "ok"

    try:
        client = request.app.state.weaviate_client
        if hasattr(client, "_get_client"):
            client._get_client().is_connected()
    except Exception:
        weaviate_ok = "error"

    try:
        redis = request.app.state.arq_pool
        await redis.ping()
    except Exception:
        redis_ok = "error"

    status = "ok" if weaviate_ok == "ok" and redis_ok == "ok" else "degraded"
    return HealthResponse(status=status, weaviate=weaviate_ok, redis=redis_ok)

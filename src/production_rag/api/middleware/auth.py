"""JWT authentication middleware using RS256 + JWKS."""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from jose.backends import RSAKey
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from production_rag.core.config import Settings, get_settings

log = structlog.get_logger(__name__)

# Public paths that do not require authentication (prefix-matched)
_PUBLIC_PATH_PREFIXES = ("/health", "/docs", "/openapi.json", "/redoc", "/metrics")


class JWTMiddleware(BaseHTTPMiddleware):
    """Validate Bearer JWT on all non-public routes."""

    def __init__(self, app: Any, settings: Settings | None = None) -> None:
        super().__init__(app)
        self._settings = settings or get_settings()
        self._jwks_cache: dict[str, Any] = {}
        self._jwks_fetched_at: float = 0.0
        self._jwks_ttl = 3600.0  # re-fetch JWKS every hour

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        path = request.url.path
        if request.method == "OPTIONS" or any(
            path == p or path.startswith(p + "/") or path.startswith(p + "?")
            for p in _PUBLIC_PATH_PREFIXES
        ):
            return await call_next(request)

        if not self._settings.jwks_uri:
            request.state.tenant_id = "default"
            request.state.scopes = ["*"]
            request.state.sub = "dev"
            return await call_next(request)

        token = self._extract_token(request)
        if not token:
            return JSONResponse(
                {"code": "missing_token", "message": "Authorization header required"},
                status_code=401,
            )

        try:
            payload = await self._decode(token)
        except HTTPException as e:
            return JSONResponse(
                {"code": "invalid_token", "message": e.detail},
                status_code=e.status_code,
            )
        except Exception:
            log.warning("jwt.decode_error")
            return JSONResponse(
                {"code": "invalid_token", "message": "Token validation failed"},
                status_code=401,
            )

        request.state.tenant_id = payload.get("tenant_id", "default")
        request.state.scopes = payload.get("scopes", [])
        request.state.sub = payload.get("sub", "")

        return await call_next(request)

    @staticmethod
    def _extract_token(request: Request) -> str | None:
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            return auth.split(" ", 1)[1]
        return None

    async def _get_jwks(self) -> dict[str, Any]:
        now = time.monotonic()
        if not self._jwks_cache or (now - self._jwks_fetched_at) > self._jwks_ttl:
            jwks_uri = self._settings.jwks_uri
            if not jwks_uri:
                # Dev mode: skip JWKS fetch
                log.warning("jwt.jwks_uri_not_configured", mode="dev_permissive")
                return {}
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(jwks_uri)
                resp.raise_for_status()
                self._jwks_cache = resp.json()
                self._jwks_fetched_at = now
        return self._jwks_cache

    async def _decode(self, token: str) -> dict[str, Any]:
        s = self._settings
        jwks = await self._get_jwks()

        options = {
            "verify_exp": True,
            "verify_nbf": True,
            "verify_iss": bool(s.jwt_issuer),
            "verify_aud": bool(s.jwt_audience),
        }

        if not jwks:
            payload: dict[str, Any] = jwt.get_unverified_claims(token)
            log.warning("jwt.unverified_decode", sub=payload.get("sub"))
            return payload

        try:
            payload = jwt.decode(
                token,
                jwks,
                algorithms=[s.jwt_algorithm],
                audience=s.jwt_audience or None,
                issuer=s.jwt_issuer or None,
                options=options,
            )
        except JWTError as e:
            raise HTTPException(status_code=401, detail="Token validation failed") from e

        return payload

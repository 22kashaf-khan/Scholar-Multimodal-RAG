"""FastAPI application entry point.

Startup: connects Weaviate, Redis, initialises the RAG chain.
Shutdown: closes Weaviate and Redis connections.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import structlog
from arq.connections import ArqRedis, create_pool
from arq.connections import RedisSettings as ArqRedisSettings
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from production_rag.api.middleware.auth import JWTMiddleware
from production_rag.api.middleware.rate_limit import create_limiter
from production_rag.api.routers import chat, health, ingest, tenants
from production_rag.core.config import get_settings
from production_rag.core.logging import configure_logging

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> Any:  # type: ignore[misc]
    """Startup / shutdown lifecycle manager."""
    settings = get_settings()
    configure_logging()

    log.info("app.startup")

    from production_rag.vectorstore.weaviate_client import WeaviateClient
    weaviate_client = WeaviateClient(settings)
    await weaviate_client.connect()
    await weaviate_client.create_schema(settings.embedding_dimension)

    from production_rag.vectorstore.tenant_manager import TenantManager
    tenant_manager = TenantManager(weaviate_client)

    arq_pool: ArqRedis = await create_pool(
        ArqRedisSettings.from_dsn(settings.redis_url)
    )

    from production_rag.ingestion.embedder import get_embedder
    from production_rag.core.llm_client import get_llm_client
    embedder = get_embedder(settings)
    llm = await get_llm_client()

    from production_rag.chains.rag_chain import RAGChain
    rag_chain = RAGChain(
        weaviate_client=weaviate_client,
        embedder=embedder,
        llm=llm,
        settings=settings,
    )

    app.state.settings = settings
    app.state.weaviate_client = weaviate_client
    app.state.tenant_manager = tenant_manager
    app.state.arq_pool = arq_pool
    app.state.rag_chain = rag_chain

    log.info("app.startup.done")
    yield

    log.info("app.shutdown")
    await arq_pool.close()
    await weaviate_client.close()
    log.info("app.shutdown.done")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Production RAG Pipeline",
        description="Expert-level RAG: Weaviate hybrid, RRF, MMR, RAPTOR, CRAG, Self-RAG",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    limiter = create_limiter(settings.redis_url)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]
    app.add_middleware(SlowAPIMiddleware)

    app.add_middleware(JWTMiddleware, settings=settings)

    app.include_router(health.router)
    app.include_router(chat.router)
    app.include_router(ingest.router)
    app.include_router(tenants.router)

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.error("app.unhandled_error", path=request.url.path, error=str(exc))
        return JSONResponse(
            {
                "code": "internal_error",
                "message": "An internal error occurred",
            },
            status_code=500,
        )

    return app


app = create_app()

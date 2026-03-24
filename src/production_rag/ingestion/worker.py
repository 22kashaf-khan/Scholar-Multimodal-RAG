"""Arq background worker for async document ingestion."""

from __future__ import annotations

import os
from typing import Any

import structlog
from arq import cron
from arq.connections import RedisSettings

from production_rag.core.config import get_settings
from production_rag.core.logging import configure_logging
from production_rag.core.types import ChunkStrategy
from production_rag.ingestion.loaders.arxiv_loader import ArXivLoader
from production_rag.ingestion.loaders.pdf import PDFLoader
from production_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
from production_rag.vectorstore.tenant_manager import TenantManager
from production_rag.vectorstore.weaviate_client import WeaviateClient

log = structlog.get_logger(__name__)


async def ingest_documents(
    ctx: dict[str, Any],
    *,
    arxiv_ids: list[str] | None = None,
    pdf_paths: list[str] | None = None,
    tenant_id: str = "default",
    chunking_strategy: str = "hierarchical",
    enable_raptor: bool = True,
) -> dict[str, Any]:
    """Arq job: load and ingest a set of ArXiv papers or local PDFs.

    Returns serialisable result dict for job status endpoint.
    """
    settings = get_settings()
    configure_logging()

    weaviate = ctx["weaviate"]
    tenant_manager = ctx["tenant_manager"]
    pipeline: IngestionPipeline = ctx["pipeline"]

    config = IngestionConfig(
        tenant_id=tenant_id,
        chunking_strategy=ChunkStrategy(chunking_strategy),
        enable_raptor=enable_raptor,
    )

    documents = []

    if arxiv_ids:
        loader = ArXivLoader(fetch_pdf=True)
        docs = await loader.load(",".join(arxiv_ids))
        documents.extend(docs)
        log.info("worker.arxiv_loaded", count=len(docs))

    if pdf_paths:
        pdf_loader = PDFLoader()
        for path in pdf_paths:
            docs = await pdf_loader.load(path)
            documents.extend(docs)

    if not documents:
        return {"status": "skipped", "reason": "no_documents_loaded"}

    result = await pipeline.ingest(documents, config)

    return {
        "status": "done" if not result.errors else "partial",
        "doc_ids": result.doc_ids,
        "total_chunks": result.total_chunks,
        "leaf_chunks": result.leaf_chunks,
        "summary_chunks": result.summary_chunks,
        "errors": result.errors,
        "duration_s": result.duration_s,
    }



async def startup(ctx: dict[str, Any]) -> None:
    """Initialise shared resources on worker start."""
    from production_rag.ingestion.embedder import get_default_embedder
    from production_rag.core.llm_client import get_llm_client

    configure_logging()
    log.info("worker.startup")

    weaviate = WeaviateClient()
    await weaviate.connect()

    tenant_manager = TenantManager(weaviate)
    embedder = get_default_embedder()
    llm = await get_llm_client()

    pipeline = IngestionPipeline(
        weaviate=weaviate,
        tenant_manager=tenant_manager,
        embedder=embedder,
        llm=llm,
    )

    ctx["weaviate"] = weaviate
    ctx["tenant_manager"] = tenant_manager
    ctx["pipeline"] = pipeline
    log.info("worker.startup.done")


async def shutdown(ctx: dict[str, Any]) -> None:
    await ctx["weaviate"].close()
    log.info("worker.shutdown")



class WorkerSettings:
    functions = [ingest_documents]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(
        os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )
    job_timeout = 3600   # 1 hour max per job
    max_jobs = 4
    keep_result = 3600   # keep result for 1 hour

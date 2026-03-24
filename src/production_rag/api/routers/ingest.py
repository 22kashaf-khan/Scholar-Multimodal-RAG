"""Ingest router — async document ingestion via Arq job queue.

POST /ingest → 202 Accepted + {job_id}
GET  /ingest/{job_id} → job status and result
"""

from __future__ import annotations

import uuid

from arq.connections import ArqRedis
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

router = APIRouter(prefix="/ingest", tags=["ingest"])


class IngestRequest(BaseModel):
    arxiv_ids: list[str] = Field(default_factory=list)
    pdf_paths: list[str] = Field(default_factory=list)
    tenant_id: str | None = None
    chunking_strategy: str = "hierarchical"
    enable_raptor: bool = True


class IngestResponse(BaseModel):
    job_id: str
    status: str = "queued"


@router.post("", response_model=IngestResponse, status_code=202)
async def enqueue_ingest(body: IngestRequest, request: Request) -> IngestResponse:
    """Enqueue an ingestion job.  Returns 202 + job_id immediately."""
    tenant_id = (
        getattr(request.state, "tenant_id", None)
        or body.tenant_id
        or "default"
    )
    redis: ArqRedis = request.app.state.arq_pool

    job = await redis.enqueue_job(
        "ingest_documents",
        arxiv_ids=body.arxiv_ids or None,
        pdf_paths=body.pdf_paths or None,
        tenant_id=tenant_id,
        chunking_strategy=body.chunking_strategy,
        enable_raptor=body.enable_raptor,
    )

    return IngestResponse(job_id=job.job_id)


@router.get("/{job_id}")
async def get_job_status(job_id: str, request: Request) -> dict:  # type: ignore[type-arg]
    """Poll the status of an ingestion job."""
    redis: ArqRedis = request.app.state.arq_pool

    try:
        job_result = await redis.get_job_result(job_id)
    except Exception:
        raise HTTPException(status_code=404, detail="Job not found")

    if job_result is None:
        return {"job_id": job_id, "status": "pending"}

    return {
        "job_id": job_id,
        "status": job_result.status,
        "result": job_result.result,
        "start_time": str(job_result.start_time) if job_result.start_time else None,
        "finish_time": str(job_result.finish_time) if job_result.finish_time else None,
    }

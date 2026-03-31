"""Chat router — SSE streaming RAG responses."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from production_rag.core.config import get_settings
from production_rag.generation.streaming import rag_sse_stream

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    tenant_id: str | None = Field(default=None)
    enable_crag: bool = True
    enable_self_rag: bool = True


def _get_chain(request: Request) -> object:
    return request.app.state.rag_chain


@router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
) -> EventSourceResponse:
    """Stream a RAG response as SSE events."""
    tenant_id = (
        getattr(request.state, "tenant_id", None)
        or body.tenant_id
        or "default"
    )

    rag_chain = request.app.state.rag_chain

    async def _event_generator():  # type: ignore[return]
        try:
            async for event in rag_chain.stream(  # type: ignore[attr-defined]
                query=body.query,
                tenant_id=tenant_id,
                enable_crag=body.enable_crag,
                enable_self_rag=body.enable_self_rag,
            ):
                yield event
        except Exception as e:
            from production_rag.generation.streaming import error_event
            yield error_event("pipeline_error", "Internal retrieval error").to_json()

    return EventSourceResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )

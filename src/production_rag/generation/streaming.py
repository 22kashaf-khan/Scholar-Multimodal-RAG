"""SSE streaming events for the RAG pipeline."""

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from dataclasses import asdict, dataclass
from typing import Any

import structlog

from production_rag.core.types import Citation, RAGResponse, RetrievalDiagnostics, RetrievedChunk

log = structlog.get_logger(__name__)


@dataclass
class SSEEvent:
    type: str
    data: dict[str, Any]

    def to_sse(self) -> str:
        """Format as SSE wire format."""
        payload = json.dumps({"type": self.type, "data": self.data})
        return f"data: {payload}\n\n"

    def to_json(self) -> str:
        """Return JSON payload only (no SSE framing) for EventSourceResponse."""
        return json.dumps({"type": self.type, "data": self.data})


def context_chunk_event(chunk: RetrievedChunk, index: int) -> SSEEvent:
    c = chunk.chunk
    return SSEEvent(
        type="context_chunk",
        data={
            "index": index,
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "title": c.title,
            "arxiv_id": c.arxiv_id,
            "page": c.page,
            "snippet": chunk.display_text[:300],
            "score": round(chunk.rerank_score or chunk.rrf_score, 4),
            "chunk_type": getattr(c, "chunk_type", "text"),
        },
    )


def token_event(token: str) -> SSEEvent:
    return SSEEvent(type="token", data={"token": token})


def citation_event(citation: Citation) -> SSEEvent:
    return SSEEvent(
        type="citation",
        data={
            "citation_id": citation.citation_id,
            "doc_id": citation.doc_id,
            "chunk_id": citation.chunk_id,
            "arxiv_id": citation.arxiv_id,
            "title": citation.title,
            "page": citation.page,
            "snippet": citation.snippet,
            "score": round(citation.score, 4),
        },
    )


def diagnostics_event(d: RetrievalDiagnostics) -> SSEEvent:
    return SSEEvent(
        type="diagnostics",
        data={
            "candidate_count": d.candidate_count,
            "post_rrf_count": d.post_rrf_count,
            "post_mmr_count": d.post_mmr_count,
            "reranked_count": d.reranked_count,
            "top_k_used": d.top_k_used,
            "adaptive_hops": d.adaptive_hops,
            "quality_score": round(d.quality_score, 4),
            "query_variants": d.query_variants_used,
        },
    )


def done_event() -> SSEEvent:
    return SSEEvent(type="done", data={})


def error_event(code: str, message: str) -> SSEEvent:
    return SSEEvent(type="error", data={"code": code, "message": message})


async def rag_sse_stream(
    query: str,
    chunks: list[RetrievedChunk],
    diagnostics: RetrievalDiagnostics,
    synthesizer: object,  # Synthesizer
    citations: list[Citation] | None = None,
) -> AsyncGenerator[str, None]:
    """Full SSE stream generator for a RAG response. Yields JSON payload strings."""
    for i, chunk in enumerate(chunks):
        yield context_chunk_event(chunk, i).to_json()


    full_answer = ""
    try:
        async for token in synthesizer.stream(query, chunks):  # type: ignore[attr-defined]
            full_answer += token
            yield token_event(token).to_json()
    except Exception as e:
        log.error("sse_stream.generation_failed", error=str(e))
        yield error_event("generation_failed", str(e)).to_json()
        return

    if citations is None:
        from production_rag.generation.synthesizer import _extract_citations
        source_map = {i + 1: c for i, c in enumerate(chunks)}
        citations = _extract_citations(full_answer, source_map)

    for citation in citations:
        yield citation_event(citation).to_json()

    yield diagnostics_event(diagnostics).to_json()

    yield done_event().to_json()

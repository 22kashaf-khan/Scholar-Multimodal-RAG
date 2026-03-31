"""Answer synthesizer with inline citation extraction."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog

from production_rag.core.llm_client import LLMClient
from production_rag.core.types import Citation, RAGResponse, RetrievalDiagnostics, RetrievedChunk

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """\
You are an expert scientific assistant.  Answer the user's question using ONLY the
provided source passages below.  Follow these rules strictly:

1. Cite your sources inline using [SOURCE N] notation.
2. If a claim is not supported by any source, write [UNSUPPORTED: <claim>].
3. Do NOT make up information beyond what the sources contain.
4. Be precise and technical; this audience is domain experts.
5. If the sources are insufficient to answer fully, say so explicitly.

TABLE HANDLING:
- Source passages marked with [TABLE]...[/TABLE] contain structured tabular data.
- When a question is about a table, extract and present the relevant rows, columns,
  and numeric values directly from the [TABLE] block — do not paraphrase or omit numbers.
- If the table uses markdown or space-aligned formatting, reproduce it as a markdown table.
- Always cite the [SOURCE N] that contains the table data.
"""

_CONTEXT_HEADER = "SOURCE PASSAGES:\n" + "─" * 60


def _build_context_block(chunks: list[RetrievedChunk]) -> tuple[str, dict[int, RetrievedChunk]]:
    """Build context string and return (context_str, {source_n → RetrievedChunk})."""
    lines: list[str] = [_CONTEXT_HEADER]
    source_map: dict[int, RetrievedChunk] = {}

    for i, chunk in enumerate(chunks, start=1):
        c = chunk.chunk
        header = (
            f"[SOURCE {i} | doc: {c.doc_id} | chunk: {c.chunk_id} "
            f"| page: {c.page} | arxiv: {c.arxiv_id}]"
        )
        text = chunk.display_text
        if getattr(c, "chunk_type", "text") == "table":
            lines.append(f"\n{header} [TABLE]\n{text}\n[/TABLE]")
        else:
            lines.append(f"\n{header}\n{text}")
        source_map[i] = chunk

    return "\n".join(lines), source_map


def _extract_citations(
    answer: str,
    source_map: dict[int, RetrievedChunk],
) -> list[Citation]:
    """Parse [SOURCE N] references from the answer into Citation objects."""
    import re

    cited_ids = set(map(int, re.findall(r"\[SOURCE (\d+)\]", answer)))
    citations: list[Citation] = []

    for n in sorted(cited_ids):
        chunk_ref = source_map.get(n)
        if chunk_ref is None:
            continue
        c = chunk_ref.chunk
        # Use first 200 chars of display_text as snippet
        snippet = chunk_ref.display_text[:200].strip()
        citations.append(
            Citation(
                citation_id=f"cite_{n}",
                doc_id=c.doc_id,
                chunk_id=c.chunk_id,
                page=c.page,
                snippet=snippet,
                score=chunk_ref.rerank_score or chunk_ref.rrf_score,
                arxiv_id=c.arxiv_id,
                title=c.title,
            )
        )

    return citations


class Synthesizer:
    """LLM-based answer synthesizer with citation extraction."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def synthesize(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        diagnostics: RetrievalDiagnostics,
    ) -> RAGResponse:
        """Generate a complete answer synchronously (non-streaming)."""
        import time

        context_str, source_map = _build_context_block(chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_str}\n\n{'─'*60}\nQUESTION: {query}"},
        ]

        t0 = time.perf_counter()
        answer, tokens = await self._llm.complete(messages, max_tokens=2048)
        latency_ms = (time.perf_counter() - t0) * 1000

        citations = _extract_citations(answer, source_map)

        log.info(
            "synthesizer.done",
            query_len=len(query),
            chunks=len(chunks),
            citations=len(citations),
            tokens=tokens,
            latency_ms=round(latency_ms),
        )

        return RAGResponse(
            answer=answer,
            citations=citations,
            diagnostics=diagnostics,
            latency_ms=latency_ms,
            tokens_used=tokens,
        )

    async def stream(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> AsyncGenerator[str, None]:
        """Stream raw answer tokens."""
        context_str, _ = _build_context_block(chunks)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"{context_str}\n\n{'─'*60}\nQUESTION: {query}"},
        ]
        async for token in self._llm.stream(messages, max_tokens=2048):
            yield token

    def build_context_block(
        self, chunks: list[RetrievedChunk]
    ) -> tuple[str, dict[int, RetrievedChunk]]:
        """Public accessor for context block (used by CRAG/Self-RAG)."""
        return _build_context_block(chunks)

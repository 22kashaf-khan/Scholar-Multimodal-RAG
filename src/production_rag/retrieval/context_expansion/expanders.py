"""Context expansion strategies.

After RRF + MMR selects a diverse candidate set, context expansion
replaces (or augments) the retrieved chunk text before synthesis to
reduce the "lost-in-the-middle" effect and preserve local coherence.

Two strategies:
1. ParentDocumentExpander — fetch the parent section chunk.
2. SentenceWindowExpander — fetch ±N adjacent chunks in the same document.
"""

from __future__ import annotations

import asyncio

import structlog

from production_rag.core.types import RetrievedChunk
from production_rag.vectorstore.weaviate_client import WeaviateClient

log = structlog.get_logger(__name__)


class ParentDocumentExpander:
    """Replace chunk text with its parent (section-level) chunk text.

    If the parent is not found (e.g. RAPTOR summary node), the original
    chunk text is preserved unchanged.
    """

    def __init__(self, client: WeaviateClient) -> None:
        self._client = client

    async def expand(
        self,
        candidates: list[RetrievedChunk],
        tenant_id: str,
    ) -> list[RetrievedChunk]:
        parent_ids = {
            c.chunk.parent_chunk_id
            for c in candidates
            if c.chunk.parent_chunk_id
        }
        if not parent_ids:
            return candidates

        # Fetch all parent chunks concurrently
        fetch_tasks = {
            pid: self._client.fetch_by_chunk_id(pid, tenant_id)
            for pid in parent_ids
        }
        fetched = await asyncio.gather(*fetch_tasks.values(), return_exceptions=True)
        parent_map = {
            pid: chunk
            for pid, chunk in zip(fetch_tasks.keys(), fetched, strict=True)
            if not isinstance(chunk, Exception) and chunk is not None
        }

        for candidate in candidates:
            pid = candidate.chunk.parent_chunk_id
            if pid and pid in parent_map:
                parent_chunk = parent_map[pid]
                candidate.expanded_text = parent_chunk.chunk_text
                log.debug(
                    "context_expansion.parent",
                    chunk_id=candidate.chunk.chunk_id,
                    parent_id=pid,
                    expanded_chars=len(candidate.expanded_text),
                )

        return candidates


class SentenceWindowExpander:
    """Expand chunk text by including ±window_size adjacent chunks.

    Adjacent chunks are fetched by chunk_index range within the same doc.
    The expanded text preserves reading order.
    """

    def __init__(self, client: WeaviateClient, window_size: int = 2) -> None:
        self._client = client
        self._window = window_size

    async def expand(
        self,
        candidates: list[RetrievedChunk],
        tenant_id: str,
    ) -> list[RetrievedChunk]:
        tasks = [self._expand_one(c, tenant_id) for c in candidates]
        expanded = await asyncio.gather(*tasks, return_exceptions=True)

        result: list[RetrievedChunk] = []
        for candidate, res in zip(candidates, expanded, strict=True):
            if isinstance(res, Exception):
                log.warning(
                    "context_expansion.window_failed",
                    chunk_id=candidate.chunk.chunk_id,
                    error=str(res),
                )
                result.append(candidate)
            else:
                result.append(res)
        return result

    async def _expand_one(
        self, candidate: RetrievedChunk, tenant_id: str
    ) -> RetrievedChunk:
        idx = candidate.chunk.chunk_index
        doc_id = candidate.chunk.doc_id
        i_min = max(0, idx - self._window)
        i_max = idx + self._window

        window_chunks = await self._client.fetch_by_chunk_index_range(
            doc_id=doc_id,
            index_min=i_min,
            index_max=i_max,
            tenant_id=tenant_id,
        )

        if window_chunks:
            candidate.expanded_text = "\n".join(c.chunk_text for c in window_chunks)
            log.debug(
                "context_expansion.window",
                chunk_id=candidate.chunk.chunk_id,
                window=f"[{i_min},{i_max}]",
                n_chunks=len(window_chunks),
            )
        return candidate

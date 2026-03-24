"""Semantic chunker.

Splits text at points of maximum semantic discontinuity — where
consecutive sentence embeddings have the lowest cosine similarity.
Produces topically coherent chunks at the cost of embedding inference during ingestion.
"""

from __future__ import annotations

import hashlib

import numpy as np
import structlog

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, Document
from production_rag.ingestion.chunkers.base import BaseChunker

log = structlog.get_logger(__name__)


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter; good enough for scientific text."""
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SemanticChunker(BaseChunker):
    """Embedding-similarity-based semantic chunker.

    Args:
        embedder: Any object with an `embed_texts(list[str]) -> list[list[float]]` method.
        breakpoint_percentile: Percentile of similarity score distribution
            below which a boundary is inserted (lower → more chunks).
        min_chunk_chars: Minimum characters per chunk before merging small pieces.
    """

    strategy = ChunkStrategy.SEMANTIC

    def __init__(
        self,
        embedder: object,
        breakpoint_percentile: float = 20.0,
        min_chunk_chars: int = 200,
    ) -> None:
        self._embedder = embedder
        self._breakpoint_percentile = breakpoint_percentile
        self._min_chunk_chars = min_chunk_chars

    def chunk(self, document: Document, tenant_id: str = "default") -> list[Chunk]:
        base = self._base_fields(document)
        sentences = _split_sentences(document.text)

        if len(sentences) < 2:
            # Degenerate: return single chunk
            chunk_id = hashlib.sha256(
                f"semantic:{document.doc_id}:0".encode()
            ).hexdigest()[:24]
            return [
                Chunk(
                    chunk_text=document.text,
                    chunk_id=chunk_id,
                    chunk_level=ChunkLevel.LEAF,
                    start_char=0,
                    end_char=len(document.text),
                    tenant_id=tenant_id,
                    **base,  # type: ignore[arg-type]
                )
            ]

        # Embed all sentences
        embeddings: list[list[float]] = self._embedder.embed_texts(sentences)  # type: ignore[attr-defined]
        emb_array = np.array(embeddings)

        # Compute pairwise consecutive similarity
        sims = [
            _cosine_sim(emb_array[i], emb_array[i + 1])
            for i in range(len(emb_array) - 1)
        ]

        # Breakpoints at low-similarity gaps
        threshold = float(np.percentile(sims, self._breakpoint_percentile))
        breakpoints: set[int] = {
            i + 1 for i, s in enumerate(sims) if s < threshold
        }
        breakpoints.add(len(sentences))  # sentinel

        chunks: list[Chunk] = []
        chunk_idx = 0
        group_start = 0
        cursor = 0

        for bp in sorted(breakpoints):
            group = sentences[group_start:bp]
            text = " ".join(group).strip()
            if len(text) < self._min_chunk_chars and chunks:
                # Merge tiny tail into previous chunk
                chunks[-1].chunk_text += " " + text
                chunks[-1].end_char = cursor + len(text)
                group_start = bp
                cursor += len(text) + 1
                continue

            start = document.text.find(sentences[group_start], cursor)
            if start == -1:
                start = cursor
            end = start + len(text)

            chunk_id = hashlib.sha256(
                f"semantic:{document.doc_id}:{chunk_idx}".encode()
            ).hexdigest()[:24]
            chunks.append(
                Chunk(
                    chunk_text=text,
                    chunk_id=chunk_id,
                    chunk_level=ChunkLevel.LEAF,
                    chunk_index=chunk_idx,
                    start_char=start,
                    end_char=end,
                    tenant_id=tenant_id,
                    **base,  # type: ignore[arg-type]
                )
            )
            chunk_idx += 1
            cursor = end
            group_start = bp

        log.debug(
            "semantic_chunker.done",
            doc_id=document.doc_id,
            sentences=len(sentences),
            chunks=len(chunks),
            threshold=round(threshold, 4),
        )
        return chunks

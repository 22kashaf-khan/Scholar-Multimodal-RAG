"""Chunker that converts Docling Serve output into typed Chunk objects.

Works in tandem with DoclingPDFLoader.  If the Document has
``metadata["docling_chunks"]``, it creates one Chunk per Docling chunk item,
tagging each with the appropriate ``chunk_type`` (text | table | figure).
Tables are kept whole — never split mid-row.

Falls back to HierarchicalChunker for documents without Docling metadata
(e.g. ArXiv docs loaded as raw text).
"""

from __future__ import annotations

import hashlib

import structlog

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, Document
from production_rag.ingestion.chunkers.base import BaseChunker

log = structlog.get_logger(__name__)


def _make_id(doc_id: str, index: int) -> str:
    return hashlib.sha256(f"docling:{doc_id}:{index}".encode()).hexdigest()[:24]


def _detect_type(doc_items: list[str], captions: list[str] | None) -> str:
    """Infer chunk type from Docling item references."""
    for ref in doc_items:
        if "#/tables/" in ref:
            return "table"
        if "#/pictures/" in ref:
            return "figure"
    # Chunks with captions but no table ref are usually figure captions
    if captions:
        return "figure"
    return "text"


class DoclingChunker(BaseChunker):
    """Produce typed Chunks from Docling Serve's hybrid chunker output.

    chunk_type values stored on each Chunk:
      - ``"text"``   — prose paragraph / section
      - ``"table"``  — Markdown-formatted table (kept as one chunk)
      - ``"figure"`` — figure caption text
    """

    strategy = ChunkStrategy.DOCLING

    def chunk(self, document: Document, tenant_id: str = "default") -> list[Chunk]:
        raw_chunks: list[dict] | None = document.metadata.get("docling_chunks")

        if not raw_chunks:
            log.warning(
                "docling_chunker.no_metadata",
                doc_id=document.doc_id,
                fallback="hierarchical",
            )
            from production_rag.ingestion.chunkers.hierarchical import HierarchicalChunker
            return HierarchicalChunker().chunk(document, tenant_id)

        base = self._base_fields(document)
        chunks: list[Chunk] = []
        skipped = 0

        for i, raw in enumerate(raw_chunks):
            text = (raw.get("text") or "").strip()
            if not text:
                skipped += 1
                continue

            doc_items: list[str] = raw.get("doc_items") or []
            captions: list[str] | None = raw.get("captions")
            chunk_type = _detect_type(doc_items, captions)

            page_numbers: list[int] | None = raw.get("page_numbers")
            page = page_numbers[0] if page_numbers else 0

            chunks.append(Chunk(
                chunk_text=text,
                chunk_id=_make_id(document.doc_id, i),
                chunk_type=chunk_type,
                chunk_level=ChunkLevel.LEAF,
                chunk_index=i,
                section_index=0,
                page=page,
                tenant_id=tenant_id,
                **base,
            ))

        text_count = sum(1 for c in chunks if c.chunk_type == "text")
        table_count = sum(1 for c in chunks if c.chunk_type == "table")
        figure_count = sum(1 for c in chunks if c.chunk_type == "figure")
        log.info(
            "docling_chunker.done",
            doc_id=document.doc_id,
            total=len(chunks),
            text=text_count,
            tables=table_count,
            figures=figure_count,
            skipped=skipped,
        )
        return chunks

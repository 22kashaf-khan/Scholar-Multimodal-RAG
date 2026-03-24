"""Hierarchical parent-child chunker using LlamaIndex's HierarchicalNodeParser.

Produces a 4-level tree:
  Level 0 (paper)     — full abstract / paper introduction
  Level 1 (section)   — major sections (~2000 chars)
  Level 2 (paragraph) — paragraphs (~500 chars, retrieved by default)
  Level 3 (sentence)  — individual sentences (~100 chars, fine-grained)
"""

from __future__ import annotations

import hashlib
from dataclasses import replace

import structlog

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, Document
from production_rag.ingestion.chunkers.base import BaseChunker

log = structlog.get_logger(__name__)

# Chunk sizes per level (in characters)
_LEVEL_SIZES = {
    ChunkLevel.PAPER: 4000,
    ChunkLevel.SECTION: 2000,
    ChunkLevel.LEAF: 500,   # paragraph — default retrieval target
}


def _make_id(strategy_prefix: str, doc_id: str, level: str, index: int) -> str:
    return hashlib.sha256(
        f"{strategy_prefix}:{doc_id}:{level}:{index}".encode()
    ).hexdigest()[:24]


def _split_at_size(text: str, size: int) -> list[tuple[int, int]]:
    """Return (start, end) character spans splitting text at ~size boundaries."""
    spans = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Try to break at paragraph/sentence boundary
        if end < len(text):
            for sep in ["\n\n", "\n", ". ", " "]:
                last_sep = text.rfind(sep, start, end)
                if last_sep > start:
                    end = last_sep + len(sep)
                    break
        spans.append((start, end))
        start = end
    return spans


class HierarchicalChunker(BaseChunker):
    """Build a 4-level parent-child chunk tree.

    Returns ALL levels interleaved — callers can filter by chunk_level
    to get retrieval targets (typically ChunkLevel.LEAF = paragraph).
    """

    strategy = ChunkStrategy.HIERARCHICAL

    def chunk(self, document: Document, tenant_id: str = "default") -> list[Chunk]:
        # Try LlamaIndex first; fall back to custom implementation
        try:
            return self._chunk_with_llamaindex(document, tenant_id)
        except ImportError:
            log.warning(
                "hierarchical_chunker.llamaindex_unavailable",
                fallback="custom_splitter",
            )
            return self._chunk_custom(document, tenant_id)

    def _chunk_with_llamaindex(
        self, document: Document, tenant_id: str
    ) -> list[Chunk]:
        from llama_index.core import Document as LIDoc
        from llama_index.core.node_parser import (
            HierarchicalNodeParser,
            get_leaf_nodes,
        )

        li_doc = LIDoc(
            text=document.text,
            metadata={
                "doc_id": document.doc_id,
                "title": document.title,
                "arxiv_id": document.arxiv_id,
            },
        )
        parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[_LEVEL_SIZES[ChunkLevel.PAPER],
                         _LEVEL_SIZES[ChunkLevel.SECTION],
                         _LEVEL_SIZES[ChunkLevel.LEAF]],
        )
        all_nodes = parser.get_nodes_from_documents([li_doc])

        level_map = {
            0: ChunkLevel.PAPER,
            1: ChunkLevel.SECTION,
            2: ChunkLevel.LEAF,
        }

        chunks: list[Chunk] = []
        base = self._base_fields(document)

        for node in all_nodes:
            depth = node.metadata.get("__depth__", 2)
            level = level_map.get(depth, ChunkLevel.LEAF)
            chunk_id = _make_id("hier", document.doc_id, level.value,
                                 hash(node.node_id) % 999999)
            parent_id = None
            if node.parent_node:
                parent_id = _make_id(
                    "hier", document.doc_id, level_map.get(depth - 1,
                     ChunkLevel.SECTION).value,
                    hash(node.parent_node.node_id) % 999999,
                )

            chunks.append(
                Chunk(
                    chunk_text=node.text,
                    chunk_id=chunk_id,
                    chunk_level=level,
                    parent_chunk_id=parent_id,
                    tenant_id=tenant_id,
                    **base,    # type: ignore[arg-type]
                )
            )

        self._assign_indexes(chunks, document)
        return chunks

    def _chunk_custom(self, document: Document, tenant_id: str) -> list[Chunk]:
        """Custom fallback hierarchical splitter (no LlamaIndex needed)."""
        base = self._base_fields(document)
        text = document.text
        chunks: list[Chunk] = []

        # Level 0: paper — one or two top-level chunks
        paper_spans = _split_at_size(text, _LEVEL_SIZES[ChunkLevel.PAPER])
        for pi, (ps, pe) in enumerate(paper_spans):
            paper_id = _make_id("hier", document.doc_id, "paper", pi)
            chunks.append(Chunk(
                chunk_text=text[ps:pe].strip(),
                chunk_id=paper_id,
                chunk_level=ChunkLevel.PAPER,
                start_char=ps, end_char=pe,
                tenant_id=tenant_id, **base,  # type: ignore[arg-type]
            ))

            # Level 1: sections within paper chunk
            section_spans = _split_at_size(text[ps:pe], _LEVEL_SIZES[ChunkLevel.SECTION])
            for si, (ss, se) in enumerate(section_spans):
                abs_ss, abs_se = ps + ss, ps + se
                section_id = _make_id("hier", document.doc_id, "section", pi * 100 + si)
                chunks.append(Chunk(
                    chunk_text=text[abs_ss:abs_se].strip(),
                    chunk_id=section_id,
                    chunk_level=ChunkLevel.SECTION,
                    parent_chunk_id=paper_id,
                    section_index=si,
                    start_char=abs_ss, end_char=abs_se,
                    tenant_id=tenant_id, **base,  # type: ignore[arg-type]
                ))

                # Level 2: paragraphs (leaf — retrieval target)
                leaf_spans = _split_at_size(text[abs_ss:abs_se], _LEVEL_SIZES[ChunkLevel.LEAF])
                for li, (ls, le) in enumerate(leaf_spans):
                    abs_ls, abs_le = abs_ss + ls, abs_ss + le
                    leaf_id = _make_id(
                        "hier", document.doc_id, "leaf",
                        pi * 10000 + si * 100 + li
                    )
                    chunks.append(Chunk(
                        chunk_text=text[abs_ls:abs_le].strip(),
                        chunk_id=leaf_id,
                        chunk_level=ChunkLevel.LEAF,
                        parent_chunk_id=section_id,
                        section_index=si,
                        start_char=abs_ls, end_char=abs_le,
                        tenant_id=tenant_id, **base,  # type: ignore[arg-type]
                    ))

        self._assign_indexes(chunks, document)
        return chunks

    @staticmethod
    def _assign_indexes(chunks: list[Chunk], document: Document) -> None:
        """Assign sequential chunk_index within each level for sentence-window."""
        from collections import defaultdict
        counters: dict[str, int] = defaultdict(int)
        for chunk in chunks:
            lvl = chunk.chunk_level.value
            chunk.chunk_index = counters[lvl]
            counters[lvl] += 1
            chunk.doc_id = document.doc_id

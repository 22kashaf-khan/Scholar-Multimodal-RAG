"""Shared domain types used across all modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkLevel(str, Enum):
    """RAPTOR hierarchy levels. Leaf = actual text; section/paper = summaries."""

    LEAF = "leaf"
    SECTION = "section"
    PAPER = "paper"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    LATE = "late"


class QueryComplexity(str, Enum):
    FACTOID = "factoid"
    MULTI_HOP = "multi_hop"
    SUMMARY = "summary"


class RetrieverType(str, Enum):
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"


@dataclass
class Chunk:
    """Uniform output contract for all chunking strategies."""

    chunk_text: str
    chunk_id: str
    doc_id: str

    # Hierarchy / parent linking
    parent_chunk_id: str | None = None
    chunk_level: ChunkLevel = ChunkLevel.LEAF
    chunk_index: int = 0
    section_index: int = 0

    # Position
    start_char: int = 0
    end_char: int = 0
    page: int = 0

    # Paper metadata
    arxiv_id: str = ""
    paper_doi: str = ""
    publication_year: int = 0
    title: str = ""
    authors: list[str] = field(default_factory=list)

    # Multi-tenancy
    tenant_id: str = "default"

    # Runtime — populated by embedder, not persisted directly in this object
    embedding: list[float] = field(default_factory=list)
    title_embedding: list[float] = field(default_factory=list)


@dataclass
class Document:
    """Raw document loaded from any source before chunking."""

    text: str
    doc_id: str
    title: str = ""
    source_uri: str = ""
    arxiv_id: str = ""
    paper_doi: str = ""
    publication_year: int = 0
    authors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A chunk returned from a retriever with score and provenance."""

    chunk: Chunk
    score: float
    retriever_type: RetrieverType
    query_variant: str  # "original" | "multi_query_0" | "hyde" | "step_back"
    rank: int

    # Post-fusion / rerank
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    # Context expansion may swap chunk_text for parent/window text
    expanded_text: str = ""

    @property
    def display_text(self) -> str:
        """Return expanded text if available, else original chunk text."""
        return self.expanded_text or self.chunk.chunk_text


@dataclass
class Citation:
    citation_id: str
    doc_id: str
    chunk_id: str
    page: int
    snippet: str
    score: float
    arxiv_id: str = ""
    title: str = ""


@dataclass
class RetrievalDiagnostics:
    candidate_count: int
    post_rrf_count: int
    post_mmr_count: int
    reranked_count: int
    top_k_used: int
    adaptive_hops: int = 0
    query_variants_used: list[str] = field(default_factory=list)


@dataclass
class RAGResponse:
    """Final structured response returned by the RAG pipeline."""

    answer: str
    citations: list[Citation]
    diagnostics: RetrievalDiagnostics
    latency_ms: float = 0.0
    model_used: str = ""
    tokens_used: int = 0

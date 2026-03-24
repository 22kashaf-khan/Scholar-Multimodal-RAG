"""Chunker factory — selects and instantiates a chunking strategy by config."""

from __future__ import annotations

from production_rag.core.types import ChunkStrategy
from production_rag.ingestion.chunkers.base import BaseChunker
from production_rag.ingestion.chunkers.fixed import FixedChunker
from production_rag.ingestion.chunkers.recursive import RecursiveChunker
from production_rag.ingestion.chunkers.hierarchical import HierarchicalChunker
from production_rag.ingestion.chunkers.semantic import SemanticChunker
from production_rag.ingestion.chunkers.late import LateChunker


def get_chunker(
    strategy: ChunkStrategy | str,
    embedder: object | None = None,
    **kwargs: object,
) -> BaseChunker:
    """Return a configured chunker for the given strategy.

    Args:
        strategy: ChunkStrategy enum or its string value.
        embedder: Required for `semantic` strategy; ignored for others.
        **kwargs: Forwarded to the chunker constructor.
    """
    s = ChunkStrategy(strategy) if isinstance(strategy, str) else strategy

    if s == ChunkStrategy.FIXED:
        return FixedChunker(**kwargs)  # type: ignore[arg-type]
    if s == ChunkStrategy.RECURSIVE:
        return RecursiveChunker(**kwargs)  # type: ignore[arg-type]
    if s == ChunkStrategy.HIERARCHICAL:
        return HierarchicalChunker(**kwargs)  # type: ignore[arg-type]
    if s == ChunkStrategy.SEMANTIC:
        if embedder is None:
            raise ValueError("SemanticChunker requires an `embedder` argument.")
        return SemanticChunker(embedder=embedder, **kwargs)  # type: ignore[arg-type]
    if s == ChunkStrategy.LATE:
        return LateChunker(**kwargs)  # type: ignore[arg-type]

    raise ValueError(f"Unknown chunking strategy: {strategy}")

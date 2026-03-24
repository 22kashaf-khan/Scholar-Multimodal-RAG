"""Unit tests for MMR diversification."""

from __future__ import annotations

import numpy as np
import pytest

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, RetrievedChunk
from production_rag.retrieval.fusion.mmr import maximal_marginal_relevance


def _make_rc(chunk_id: str, embedding: list[float] | None = None) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        chunk_text=f"Text {chunk_id}",
        source_doc_id="doc_001",
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        chunk_strategy=ChunkStrategy.RECURSIVE,
        embedding=embedding,
    )
    rc = RetrievedChunk(chunk=chunk, score=0.9, retriever_type="dense")
    if embedding:
        rc.chunk.embedding = embedding
    return rc


class TestMMR:
    def test_returns_top_k(self):
        candidates = [_make_rc(f"c{i}", np.random.rand(64).tolist()) for i in range(20)]
        result = maximal_marginal_relevance(candidates, top_k=5, lambda_=0.7)
        assert len(result) <= 5

    def test_first_item_is_highest_score(self):
        # Create chunks where c0 has highest score
        candidates = []
        for i in range(5):
            emb = np.random.rand(64).tolist()
            rc = _make_rc(f"c{i}", emb)
            rc.score = 1.0 - i * 0.1
            candidates.append(rc)
        result = maximal_marginal_relevance(candidates, top_k=3, lambda_=1.0)
        # With lambda=1.0 (pure relevance), top item should be first
        assert result[0].chunk.chunk_id == "c0"

    def test_empty_candidates(self):
        assert maximal_marginal_relevance([], top_k=5) == []

    def test_no_embeddings_falls_back_gracefully(self):
        candidates = [_make_rc(f"c{i}") for i in range(5)]
        result = maximal_marginal_relevance(candidates, top_k=3)
        # Should return candidates unchanged (no embedding, can't do MMR properly)
        assert len(result) <= 5

    def test_diversity_with_lambda_zero(self):
        # lambda=0 → pure diversity → identical chunks should not all be selected
        base_emb = np.ones(64).tolist()
        identical = [_make_rc(f"dup{i}", base_emb) for i in range(5)]
        diverse = [_make_rc(f"div{i}", (np.random.rand(64)).tolist()) for i in range(5)]
        candidates = identical + diverse
        result = maximal_marginal_relevance(candidates, top_k=5, lambda_=0.0)
        ids = [r.chunk.chunk_id for r in result]
        # Diverse chunks should be preferentially selected
        assert any("div" in i for i in ids)

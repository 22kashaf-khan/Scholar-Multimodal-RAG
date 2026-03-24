"""Unit tests for RRF fusion."""

from __future__ import annotations

import pytest

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, RetrievedChunk
from production_rag.retrieval.fusion.rrf import reciprocal_rank_fusion


def _make_rc(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=chunk_id,
        chunk_text=f"Text for {chunk_id}",
        source_doc_id="doc_001",
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        chunk_strategy=ChunkStrategy.RECURSIVE,
    )
    return RetrievedChunk(chunk=chunk, score=score, retriever_type="dense")


class TestRRF:
    def test_basic_fusion_deduplicates(self):
        list_a = [_make_rc("c1"), _make_rc("c2"), _make_rc("c3")]
        list_b = [_make_rc("c2"), _make_rc("c4"), _make_rc("c1")]
        result = reciprocal_rank_fusion([list_a, list_b], k=60, top_n=10)
        ids = [r.chunk.chunk_id for r in result]
        assert len(ids) == len(set(ids)), "Deduplication failed"

    def test_chunk_appearing_in_all_lists_ranks_high(self):
        shared = "shared"
        list_a = [_make_rc(shared), _make_rc("a1"), _make_rc("a2")]
        list_b = [_make_rc(shared), _make_rc("b1"), _make_rc("b2")]
        list_c = [_make_rc(shared), _make_rc("c1"), _make_rc("c2")]
        result = reciprocal_rank_fusion([list_a, list_b, list_c], k=60, top_n=5)
        assert result[0].chunk.chunk_id == shared

    def test_empty_lists_returns_empty(self):
        assert reciprocal_rank_fusion([], k=60) == []
        assert reciprocal_rank_fusion([[]], k=60) == []

    def test_top_n_respected(self):
        lists = [[_make_rc(f"c{i}") for i in range(20)]]
        result = reciprocal_rank_fusion(lists, k=60, top_n=5)
        assert len(result) <= 5

    def test_rrf_score_assigned(self):
        lists = [[_make_rc("c1"), _make_rc("c2")]]
        result = reciprocal_rank_fusion(lists, k=60)
        for r in result:
            assert r.rrf_score is not None
            assert r.rrf_score > 0

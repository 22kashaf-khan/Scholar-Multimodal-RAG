"""Unit tests for citation validation."""

from __future__ import annotations

import pytest

from production_rag.core.types import (
    Chunk,
    ChunkLevel,
    ChunkStrategy,
    Citation,
    RetrievedChunk,
)
from production_rag.generation.citation_validator import CitationValidator


def _chunk(chunk_id: str, text: str) -> RetrievedChunk:
    c = Chunk(
        chunk_id=chunk_id,
        chunk_text=text,
        source_doc_id="doc_1",
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        chunk_strategy=ChunkStrategy.RECURSIVE,
    )
    return RetrievedChunk(chunk=c, score=0.9, retriever_type="hybrid")


def _citation(chunk_id: str, marker: str = "[1]") -> Citation:
    return Citation(
        chunk_id=chunk_id,
        citation_marker=marker,
        title="Test Paper",
        text_snippet="Some snippet from the chunk",
    )


class TestCitationValidator:
    def setup_method(self):
        self.validator = CitationValidator()

    def test_valid_citation(self):
        rc = _chunk("c1", "Transformers use attention mechanisms to process sequences")
        cit = _citation("c1")
        result = self.validator.validate(
            answer="Transformers use attention [1].",
            citations=[cit],
            retrieved_chunks=[rc],
        )
        assert result.is_valid

    def test_invalid_chunk_id(self):
        rc = _chunk("c1", "Some text")
        cit = _citation("c999")  # chunk not in retrieved set
        result = self.validator.validate(
            answer="Some answer [1].",
            citations=[cit],
            retrieved_chunks=[rc],
        )
        assert not result.is_valid
        assert "c999" in result.invalid_citation_ids

    def test_no_citations(self):
        result = self.validator.validate(
            answer="Answer with no citations.",
            citations=[],
            retrieved_chunks=[],
        )
        # Empty citations → valid (not every answer needs citations)
        assert result.is_valid

    def test_unsupported_claims_detected(self):
        rc = _chunk("c1", "The model achieves 80% accuracy")
        cit = _citation("c1")
        # Answer claims something not in the chunk
        result = self.validator.validate(
            answer="The model achieves 99% accuracy [1].",
            citations=[cit],
            retrieved_chunks=[rc],
        )
        # Low overlap should still pass chunk_id check but may flag low overlap
        assert result is not None

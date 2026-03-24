"""Unit tests for query transforms."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from production_rag.query.transforms.transforms import (
    HyDETransform,
    MultiQueryTransform,
    QueryTransformOrchestrator,
    StepBackTransform,
)


@pytest.fixture
def mock_llm():
    llm = AsyncMock()
    llm.generate_multi_queries = AsyncMock(
        return_value=["What is attention?", "Explain attention mechanism", "How does self-attention work?"]
    )
    llm.generate_hyde_document = AsyncMock(
        return_value="A hypothetical document about neural attention mechanisms."
    )
    llm.generate_step_back_query = AsyncMock(
        return_value="What are neural network architectures?"
    )
    return llm


@pytest.fixture
def mock_embedder():
    emb = AsyncMock()
    emb.embed_query = AsyncMock(return_value=[0.1] * 1024)
    return emb


class TestMultiQueryTransform:
    @pytest.mark.asyncio
    async def test_returns_correct_count(self, mock_llm, mock_embedder):
        transform = MultiQueryTransform(mock_llm, mock_embedder, n_variants=3)
        variants = await transform.transform("What is attention?", [0.1] * 1024)
        assert len(variants) == 3

    @pytest.mark.asyncio
    async def test_variants_have_embeddings(self, mock_llm, mock_embedder):
        transform = MultiQueryTransform(mock_llm, mock_embedder, n_variants=3)
        variants = await transform.transform("What is attention?", [0.1] * 1024)
        for v in variants:
            assert v.query_vector is not None
            assert len(v.query_vector) > 0


class TestHyDETransform:
    @pytest.mark.asyncio
    async def test_creates_hyde_variant(self, mock_llm, mock_embedder):
        transform = HyDETransform(mock_llm, mock_embedder)
        variants = await transform.transform("What is attention?", [0.1] * 1024)
        assert len(variants) == 1
        assert variants[0].transform_type == "hyde"


class TestStepBackTransform:
    @pytest.mark.asyncio
    async def test_creates_stepback_variant(self, mock_llm, mock_embedder):
        transform = StepBackTransform(mock_llm, mock_embedder)
        variants = await transform.transform("What is attention?", [0.1] * 1024)
        assert len(variants) == 1
        assert variants[0].transform_type == "step_back"


class TestQueryTransformOrchestrator:
    @pytest.mark.asyncio
    async def test_expand_returns_all_transforms(self, mock_llm, mock_embedder):
        orch = QueryTransformOrchestrator(mock_llm, mock_embedder)
        variants = await orch.expand("What is attention?", [0.1] * 1024)
        types = {v.transform_type for v in variants}
        assert "multi_query" in types
        assert "hyde" in types
        assert "step_back" in types

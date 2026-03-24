"""Shared pytest fixtures for unit and integration tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from production_rag.core.types import (
    Chunk,
    ChunkLevel,
    ChunkStrategy,
    RetrievedChunk,
)


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="chunk_001",
        chunk_text="Attention mechanisms allow neural networks to focus on relevant parts of input.",
        source_doc_id="doc_001",
        source="arxiv",
        arxiv_id="2312.10997",
        title="Attention Is All You Need",
        authors=["Vaswani et al."],
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        chunk_strategy=ChunkStrategy.RECURSIVE,
        page=1,
    )


@pytest.fixture
def sample_chunk_with_embedding(sample_chunk: Chunk) -> Chunk:
    import numpy as np
    sample_chunk.embedding = np.random.rand(1024).tolist()
    return sample_chunk


@pytest.fixture
def sample_retrieved_chunk(sample_chunk_with_embedding: Chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=sample_chunk_with_embedding,
        score=0.87,
        retriever_type="hybrid",
        rrf_score=0.032,
        rerank_score=0.91,
    )


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    client.complete = AsyncMock(return_value="Mocked LLM response")
    client.complete_json = AsyncMock(return_value={"answer": "mock"})
    client.stream = AsyncMock(return_value=iter(["token1", "token2"]))
    return client


@pytest.fixture
def mock_embedder():
    import numpy as np

    embedder = AsyncMock()
    embedder.embed_query = AsyncMock(return_value=np.random.rand(1024).tolist())
    embedder.embed_documents = AsyncMock(
        side_effect=lambda texts: [np.random.rand(1024).tolist() for _ in texts]
    )
    return embedder


@pytest.fixture
def mock_weaviate_client():
    client = AsyncMock()
    client.search_hybrid = AsyncMock(return_value=[])
    client.search_dense = AsyncMock(return_value=[])
    client.search_bm25 = AsyncMock(return_value=[])
    client.upsert_chunks = AsyncMock()
    client.create_schema = AsyncMock()
    return client

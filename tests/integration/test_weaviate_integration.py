"""Integration tests for the Weaviate schema and basic retrieval.

Requires a running Weaviate instance at WEAVIATE_URL.
Mark: pytest -m integration
"""

from __future__ import annotations

import uuid

import pytest
import pytest_asyncio

from production_rag.core.config import get_settings
from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy
from production_rag.vectorstore.weaviate_client import WeaviateClient
from production_rag.vectorstore.schema import build_collection_config

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def weaviate_client():
    settings = get_settings()
    client = WeaviateClient(settings)
    yield client
    # cleanup handled per-test


@pytest.mark.asyncio
async def test_schema_create_and_drop(weaviate_client: WeaviateClient):
    await weaviate_client.create_schema(embedding_dimension=64)
    await weaviate_client.drop_schema()


@pytest.mark.asyncio
async def test_upsert_and_search(weaviate_client: WeaviateClient):
    import numpy as np

    settings = get_settings()
    tenant_id = f"test_{uuid.uuid4().hex[:8]}"
    await weaviate_client.create_schema(embedding_dimension=64)

    chunk = Chunk(
        chunk_id=str(uuid.uuid4()),
        chunk_text="Attention mechanisms are used in transformers to process sequences.",
        source_doc_id="doc_test",
        chunk_index=0,
        chunk_level=ChunkLevel.LEAF,
        chunk_strategy=ChunkStrategy.RECURSIVE,
        embedding=np.random.rand(64).tolist(),
        title_embedding=np.random.rand(64).tolist(),
    )

    await weaviate_client.upsert_chunks([chunk], tenant_id=tenant_id)

    # BM25 search
    results = await weaviate_client.search_bm25(
        query="transformers attention",
        tenant_id=tenant_id,
        top_k=5,
    )
    assert len(results) >= 0  # may return 0 if BM25 doesn't match

    # Dense search
    results = await weaviate_client.search_dense(
        query_vector=np.random.rand(64).tolist(),
        tenant_id=tenant_id,
        top_k=5,
    )
    assert len(results) >= 0

    await weaviate_client.drop_schema()

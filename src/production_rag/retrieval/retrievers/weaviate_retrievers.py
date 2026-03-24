"""Dense, BM25, and Hybrid Weaviate retrievers.

All three wrap the WeaviateClient's low-level search methods and
expose the uniform BaseRetriever interface consumed by EnsembleRetriever.
"""

from __future__ import annotations

from production_rag.core.config import get_settings
from production_rag.core.types import RetrievedChunk, RetrieverType
from production_rag.retrieval.retrievers.base import BaseRetriever
from production_rag.vectorstore.weaviate_client import WeaviateClient


class DenseRetriever(BaseRetriever):
    """Near-vector (dense) retrieval using the semantic_vector named vector."""

    def __init__(
        self,
        client: WeaviateClient,
        named_vector: str = "semantic_vector",
    ) -> None:
        self._client = client
        self._named_vector = named_vector

    async def retrieve(
        self,
        query: str,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        query_variant: str = "original",
    ) -> list[RetrievedChunk]:
        results = await self._client.search_dense(
            query_vector=query_vector,
            top_k=top_k,
            tenant_id=tenant_id,
            named_vector=self._named_vector,
        )
        for r in results:
            r.query_variant = query_variant
        return results


class BM25Retriever(BaseRetriever):
    """BM25 keyword retrieval."""

    def __init__(
        self,
        client: WeaviateClient,
        query_properties: list[str] | None = None,
    ) -> None:
        self._client = client
        self._query_properties = query_properties or ["chunk_text", "title"]

    async def retrieve(
        self,
        query: str,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        query_variant: str = "original",
    ) -> list[RetrievedChunk]:
        results = await self._client.search_bm25(
            query=query,
            top_k=top_k,
            tenant_id=tenant_id,
            query_properties=self._query_properties,
        )
        for r in results:
            r.query_variant = query_variant
            r.retriever_type = RetrieverType.BM25
        return results


class HybridRetriever(BaseRetriever):
    """Weaviate native hybrid (dense + BM25, relative-score fusion).

    alpha=0.0 → pure BM25, alpha=1.0 → pure dense.
    """

    def __init__(
        self,
        client: WeaviateClient,
        alpha: float | None = None,
        named_vector: str = "semantic_vector",
    ) -> None:
        self._client = client
        self._alpha = alpha if alpha is not None else get_settings().retrieval_hybrid_alpha
        self._named_vector = named_vector

    async def retrieve(
        self,
        query: str,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        query_variant: str = "original",
    ) -> list[RetrievedChunk]:
        results = await self._client.search_hybrid(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            tenant_id=tenant_id,
            alpha=self._alpha,
            named_vector=self._named_vector,
        )
        for r in results:
            r.query_variant = query_variant
            r.retriever_type = RetrieverType.HYBRID
        return results

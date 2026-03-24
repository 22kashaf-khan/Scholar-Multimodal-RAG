"""Base retriever interface.

All retrievers implement retrieve() which accepts a query + config
and returns a ranked list of RetrievedChunk objects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from production_rag.core.types import RetrievedChunk


class BaseRetriever(ABC):
    """Async retriever interface."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        query_vector: list[float],
        tenant_id: str,
        top_k: int,
        query_variant: str = "original",
    ) -> list[RetrievedChunk]:
        """Retrieve top_k candidates for the given query from a Weaviate tenant."""
        ...

"""Query transforms: Multi-Query, HyDE, Step-Back.

Each transform takes a query string and returns one or more transformed
query strings (plus optionally their embeddings for HyDE).

All transforms:
- Use the fast LLM model (configured in Settings.llm_fast_model).
- Cache results by SHA-256 hash of the input query (in-process LRU cache).
- Are async-safe for concurrent calls.
"""

from __future__ import annotations

import asyncio
import hashlib
from functools import lru_cache

import structlog

from production_rag.core.llm_client import LLMClient
from production_rag.ingestion.embedder import BaseEmbedder

log = structlog.get_logger(__name__)


def _query_hash(prefix: str, query: str) -> str:
    return hashlib.sha256(f"{prefix}:{query}".encode()).hexdigest()[:16]


class MultiQueryTransform:
    """Generate N paraphrases of the query for retrieval diversity.

    Returns: list of (query_text, embedding) tuples.
    The original query is NOT included — callers combine separately.
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: BaseEmbedder,
        n_queries: int = 3,
    ) -> None:
        self._llm = llm
        self._embedder = embedder
        self._n = n_queries
        self._cache: dict[str, list[tuple[str, list[float]]]] = {}

    async def transform(
        self, query: str
    ) -> list[tuple[str, list[float]]]:
        """Return [(query_text, embedding), ...] for N paraphrases."""
        key = _query_hash("mq", query)
        if key in self._cache:
            return self._cache[key]

        queries = await self._llm.generate_multi_queries(query, self._n)
        if not queries:
            return []

        embeddings = await self._embedder.aembed_texts(queries)
        result = list(zip(queries, embeddings, strict=True))
        self._cache[key] = result
        log.debug("multi_query.done", n=len(result))
        return result


class HyDETransform:
    """Hypothetical Document Embeddings.

    Generates a hypothetical paper abstract → embeds it → dense search.
    Returns: (hyde_text, hyde_embedding).
    """

    def __init__(self, llm: LLMClient, embedder: BaseEmbedder) -> None:
        self._llm = llm
        self._embedder = embedder
        self._cache: dict[str, tuple[str, list[float]]] = {}

    async def transform(self, query: str) -> tuple[str, list[float]]:
        key = _query_hash("hyde", query)
        if key in self._cache:
            return self._cache[key]

        hyde_text = await self._llm.generate_hyde_document(query)
        embedding = await self._embedder.aembed_texts([hyde_text])
        result = (hyde_text, embedding[0])
        self._cache[key] = result
        log.debug("hyde.done", hyde_len=len(hyde_text))
        return result


class StepBackTransform:
    """Step-Back prompting: abstract query to higher-level concept.

    Returns: (abstracted_query, embedding).
    """

    def __init__(self, llm: LLMClient, embedder: BaseEmbedder) -> None:
        self._llm = llm
        self._embedder = embedder
        self._cache: dict[str, tuple[str, list[float]]] = {}

    async def transform(self, query: str) -> tuple[str, list[float]]:
        key = _query_hash("stepback", query)
        if key in self._cache:
            return self._cache[key]

        abstract_query = await self._llm.generate_step_back_query(query)
        embedding = await self._embedder.aembed_texts([abstract_query])
        result = (abstract_query, embedding[0])
        self._cache[key] = result
        log.debug("step_back.done", abstracted=abstract_query)
        return result


class QueryVariant:
    """A single (query_text, query_vector, variant_name) triple."""

    __slots__ = ("query", "vector", "variant_name")

    def __init__(self, query: str, vector: list[float], variant_name: str) -> None:
        self.query = query
        self.vector = vector
        self.variant_name = variant_name


class QueryTransformOrchestrator:
    """Run all query transforms in parallel and collect QueryVariant objects.

    Always includes the original query as variant "original".
    """

    def __init__(
        self,
        llm: LLMClient,
        embedder: BaseEmbedder,
        enable_multi_query: bool = True,
        enable_hyde: bool = True,
        enable_step_back: bool = True,
        n_multi_queries: int = 3,
    ) -> None:
        self._embedder = embedder
        self._multi = MultiQueryTransform(llm, embedder, n_multi_queries) if enable_multi_query else None
        self._hyde = HyDETransform(llm, embedder) if enable_hyde else None
        self._step_back = StepBackTransform(llm, embedder) if enable_step_back else None

    async def expand(self, query: str) -> list[QueryVariant]:
        """Return all query variants including original."""
        # Original query embedding
        original_vector = await self._embedder.aembed_query(query)
        variants: list[QueryVariant] = [
            QueryVariant(query, original_vector, "original")
        ]

        # Run all transforms concurrently
        tasks = []
        if self._multi:
            tasks.append(("multi", self._multi.transform(query)))
        if self._hyde:
            tasks.append(("hyde", self._hyde.transform(query)))
        if self._step_back:
            tasks.append(("step_back", self._step_back.transform(query)))

        if tasks:
            names, coros = zip(*tasks, strict=True)
            results = await asyncio.gather(*coros, return_exceptions=True)

            for name, result in zip(names, results, strict=True):
                if isinstance(result, Exception):
                    log.warning("query_transform.failed", transform=name, error=str(result))
                    continue

                if name == "multi":
                    for i, (q, v) in enumerate(result):  # type: ignore[arg-type]
                        variants.append(QueryVariant(q, v, f"multi_query_{i}"))
                else:
                    q, v = result  # type: ignore[misc]
                    variants.append(QueryVariant(q, v, name))

        log.debug("query_expand.done", n_variants=len(variants))
        return variants

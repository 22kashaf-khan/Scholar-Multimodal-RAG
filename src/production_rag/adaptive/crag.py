"""CRAG — Corrective Retrieval Augmented Generation.

If retrieval quality falls below a threshold, reformulates the query
and re-retrieves up to max_hops times.
"""

from __future__ import annotations

import structlog

from production_rag.adaptive.quality_estimator import RetrievalQualityEstimator
from production_rag.core.config import Settings, get_settings
from production_rag.core.llm_client import LLMClient
from production_rag.core.types import RetrievalDiagnostics, RetrievedChunk
from production_rag.ingestion.embedder import BaseEmbedder

log = structlog.get_logger(__name__)


class CRAGLoop:
    """Corrective retrieval loop.

    Args:
        ensemble_retriever: The EnsembleRetriever to call on re-query.
        llm: LLM client for query reformulation.
        embedder: To embed the reformulated query for quality estimation.
        quality_threshold: Minimum quality_score to accept retrieval.
        max_hops: Maximum number of corrective re-query attempts.
    """

    def __init__(
        self,
        ensemble_retriever: object,   # EnsembleRetriever (avoids circular import)
        llm: LLMClient,
        embedder: BaseEmbedder,
        settings: Settings | None = None,
    ) -> None:
        self._ensemble = ensemble_retriever
        self._llm = llm
        self._embedder = embedder
        s = settings or get_settings()
        self._threshold = s.crag_quality_threshold
        self._max_hops = s.crag_max_hops
        self._estimator = RetrievalQualityEstimator()

    async def run(
        self,
        query: str,
        initial_chunks: list[RetrievedChunk],
        initial_diagnostics: RetrievalDiagnostics,
        tenant_id: str,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics, int]:
        """Run CRAG loop.

        Returns:
            (final_chunks, final_diagnostics, hops_taken)
        """
        query_vector = await self._embedder.aembed_query(query)
        quality = self._estimator.score(initial_chunks, query_vector)

        log.info(
            "crag.initial_quality",
            query=query[:80],
            quality=round(quality, 3),
            threshold=self._threshold,
        )

        if quality >= self._threshold:
            return initial_chunks, initial_diagnostics, 0

        chunks = initial_chunks
        diagnostics = initial_diagnostics
        current_query = query
        hops = 0

        while quality < self._threshold and hops < self._max_hops:
            hops += 1
            feedback = (
                f"The retrieved passages had low relevance (quality={quality:.2f}). "
                "They did not contain specific enough information about the question."
            )

            current_query = await self._llm.reformulate_query(current_query, feedback)
            log.info(
                "crag.reformulated",
                hop=hops,
                new_query=current_query[:120],
            )

            from production_rag.retrieval.retrievers.ensemble import EnsembleConfig

            chunks, diagnostics = await self._ensemble.retrieve(  # type: ignore[attr-defined]
                current_query,
                EnsembleConfig(tenant_id=tenant_id),
            )
            diagnostics.adaptive_hops = hops

            query_vector = await self._embedder.aembed_query(current_query)
            quality = self._estimator.score(chunks, query_vector)

            log.info(
                "crag.hop_quality",
                hop=hops,
                quality=round(quality, 3),
            )

        return chunks, diagnostics, hops

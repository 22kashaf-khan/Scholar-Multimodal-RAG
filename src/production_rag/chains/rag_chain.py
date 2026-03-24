"""Full production RAG chain: ensemble retrieval + CRAG + synthesis + Self-RAG."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import structlog

from production_rag.adaptive.crag import CRAGLoop
from production_rag.adaptive.self_rag import SelfRAGCritic
from production_rag.core.config import Settings, get_settings
from production_rag.core.llm_client import LLMClient
from production_rag.core.types import RAGResponse
from production_rag.generation.citation_validator import CitationValidator
from production_rag.generation.streaming import rag_sse_stream
from production_rag.generation.synthesizer import Synthesizer
from production_rag.ingestion.embedder import BaseEmbedder
from production_rag.query.router import QueryRouter
from production_rag.retrieval.rerankers.rerankers import get_reranker
from production_rag.retrieval.retrievers.ensemble import EnsembleConfig, EnsembleRetriever
from production_rag.vectorstore.weaviate_client import WeaviateClient

log = structlog.get_logger(__name__)


class RAGChain:
    """Full production RAG chain with CRAG + Self-RAG."""

    def __init__(
        self,
        weaviate_client: WeaviateClient,
        embedder: BaseEmbedder,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        s = settings or get_settings()
        self._settings = s
        self._llm = llm

        self._ensemble = EnsembleRetriever(
            weaviate_client=weaviate_client,
            embedder=embedder,
            llm=llm,
            reranker=get_reranker(s),
            settings=s,
        )
        self._crag = CRAGLoop(
            ensemble_retriever=self._ensemble,
            llm=llm,
            embedder=embedder,
            settings=s,
        )
        self._synthesizer = Synthesizer(llm)
        self._self_rag = SelfRAGCritic(llm) if s.self_rag_enabled else None
        self._validator = CitationValidator()
        self._router = QueryRouter(llm)

    async def invoke(
        self,
        query: str,
        tenant_id: str = "default",
        enable_crag: bool = True,
        enable_self_rag: bool = True,
    ) -> RAGResponse:
        """Full pipeline, synchronous return (no streaming)."""
        routing = await self._router.route(query)

        cfg = EnsembleConfig(
            tenant_id=tenant_id,
            top_k_candidates=self._settings.retrieval_top_k_candidates,
            rrf_k=self._settings.retrieval_rrf_k,
            mmr_lambda=self._settings.retrieval_mmr_lambda,
            rerank_top_n=self._settings.retrieval_rerank_top_n,
            final_top_k=self._settings.retrieval_final_top_k,
            use_parent_expansion=routing.use_parent_expansion,
        )
        chunks, diagnostics = await self._ensemble.retrieve(query, cfg)

        if enable_crag:
            chunks, diagnostics, hops = await self._crag.run(
                query=query,
                initial_chunks=chunks,
                initial_diagnostics=diagnostics,
                tenant_id=tenant_id,
            )
            diagnostics.adaptive_hops = hops

        response = await self._synthesizer.synthesize(query, chunks, diagnostics)

        validation = self._validator.validate(response.answer, response.citations, chunks)
        if not validation.is_valid:
            log.warning(
                "rag_chain.citation_validation_failed",
                invalid=validation.invalid_citation_ids,
            )


        if enable_self_rag and self._self_rag:
            response = await self._self_rag.critique_and_refine(query, response, chunks)

        return response

    async def stream(
        self,
        query: str,
        tenant_id: str = "default",
        enable_crag: bool = True,
        enable_self_rag: bool = True,
    ) -> AsyncGenerator[str, None]:
        """Full pipeline with SSE streaming.

        Context chunks and tokens are streamed as they are generated.
        CRAG runs synchronously before streaming begins (retrieval must complete
        before generation starts).
        """
        routing = await self._router.route(query)

        cfg = EnsembleConfig(
            tenant_id=tenant_id,
            top_k_candidates=self._settings.retrieval_top_k_candidates,
            rrf_k=self._settings.retrieval_rrf_k,
            mmr_lambda=self._settings.retrieval_mmr_lambda,
            rerank_top_n=self._settings.retrieval_rerank_top_n,
            final_top_k=self._settings.retrieval_final_top_k,
            use_parent_expansion=routing.use_parent_expansion,
        )
        chunks, diagnostics = await self._ensemble.retrieve(query, cfg)

        if enable_crag:
            chunks, diagnostics, hops = await self._crag.run(
                query=query,
                initial_chunks=chunks,
                initial_diagnostics=diagnostics,
                tenant_id=tenant_id,
            )
            diagnostics.adaptive_hops = hops

        async for event in rag_sse_stream(
            query=query,
            chunks=chunks,
            diagnostics=diagnostics,
            synthesizer=self._synthesizer,
        ):
            yield event

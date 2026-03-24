"""Ensemble retriever: parallel fan-out across all query variants and retriever types.

Pipeline: query variants × (Dense, BM25, Hybrid) → RRF → MMR → expansion → rerank.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import structlog

from production_rag.core.config import Settings, get_settings
from production_rag.core.types import RetrievalDiagnostics, RetrievedChunk
from production_rag.ingestion.embedder import BaseEmbedder
from production_rag.query.transforms.transforms import QueryTransformOrchestrator
from production_rag.retrieval.context_expansion.expanders import (
    ParentDocumentExpander,
    SentenceWindowExpander,
)
from production_rag.retrieval.fusion.mmr import maximal_marginal_relevance
from production_rag.retrieval.fusion.rrf import reciprocal_rank_fusion
from production_rag.retrieval.rerankers.rerankers import BaseReranker, get_reranker
from production_rag.retrieval.retrievers.base import BaseRetriever
from production_rag.retrieval.retrievers.weaviate_retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
)
from production_rag.vectorstore.weaviate_client import WeaviateClient

log = structlog.get_logger(__name__)


@dataclass
class EnsembleConfig:
    tenant_id: str
    top_k_candidates: int = 200     # per retriever call
    rrf_k: int = 60
    mmr_lambda: float = 0.7
    mmr_top_n: int = 100            # candidates entering MMR
    rerank_top_n: int = 50          # candidates sent to reranker
    final_top_k: int = 12           # chunks injected to synthesis
    use_parent_expansion: bool = True
    use_sentence_window: bool = False  # mutually exclusive with parent for clarity


class EnsembleRetriever:
    """Full parallel retrieval ensemble with RRF, MMR, expansion, rerank."""

    def __init__(
        self,
        weaviate_client: WeaviateClient,
        embedder: BaseEmbedder,
        llm: object,  # LLMClient
        reranker: BaseReranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        s = settings or get_settings()
        self._settings = s
        self._embedder = embedder
        self._llm = llm

        self._retrievers: list[BaseRetriever] = [
            DenseRetriever(weaviate_client),
            BM25Retriever(weaviate_client),
            HybridRetriever(weaviate_client, alpha=s.retrieval_hybrid_alpha),
        ]

        self._parent_expander = ParentDocumentExpander(weaviate_client)
        self._window_expander = SentenceWindowExpander(weaviate_client)

        self._reranker = reranker or get_reranker(s)

        from production_rag.core.llm_client import LLMClient
        self._transform_orch = QueryTransformOrchestrator(
            llm=llm,  # type: ignore[arg-type]
            embedder=embedder,
        )

    async def retrieve(
        self,
        query: str,
        config: EnsembleConfig | None = None,
    ) -> tuple[list[RetrievedChunk], RetrievalDiagnostics]:
        """Full retrieval pipeline. Returns (chunks, diagnostics)."""
        t0 = time.perf_counter()
        cfg = config or EnsembleConfig(
            tenant_id="default",
            top_k_candidates=self._settings.retrieval_top_k_candidates,
            rrf_k=self._settings.retrieval_rrf_k,
            mmr_lambda=self._settings.retrieval_mmr_lambda,
            rerank_top_n=self._settings.retrieval_rerank_top_n,
            final_top_k=self._settings.retrieval_final_top_k,
        )

        variants = await self._transform_orch.expand(query)
        log.debug("ensemble.variants", n=len(variants))

        tasks = []
        for variant in variants:
            for retriever in self._retrievers:
                tasks.append(
                    retriever.retrieve(
                        query=variant.query,
                        query_vector=variant.vector,
                        tenant_id=cfg.tenant_id,
                        top_k=cfg.top_k_candidates,
                        query_variant=variant.variant_name,
                    )
                )

        rank_lists_raw = await asyncio.gather(*tasks, return_exceptions=True)
        rank_lists: list[list[RetrievedChunk]] = []
        for rl in rank_lists_raw:
            if isinstance(rl, Exception):
                log.warning("ensemble.retriever_failed", error=str(rl))
            else:
                rank_lists.append(rl)

        total_candidates = sum(len(rl) for rl in rank_lists)
        log.debug("ensemble.rank_lists", n=len(rank_lists), total_candidates=total_candidates)

        fused = reciprocal_rank_fusion(rank_lists, k=cfg.rrf_k)

        mmr_input = fused[: cfg.mmr_top_n]
        diverse = maximal_marginal_relevance(
            mmr_input, top_k=cfg.rerank_top_n, lambda_=cfg.mmr_lambda
        )

        if cfg.use_parent_expansion:
            diverse = await self._parent_expander.expand(diverse, cfg.tenant_id)
        elif cfg.use_sentence_window:
            diverse = await self._window_expander.expand(diverse, cfg.tenant_id)

        reranked = await self._reranker.rerank(
            query=query,
            candidates=diverse,
            top_k=cfg.final_top_k,
        )

        latency = time.perf_counter() - t0
        diagnostics = RetrievalDiagnostics(
            candidate_count=total_candidates,
            post_rrf_count=len(fused),
            post_mmr_count=len(diverse),
            reranked_count=len(reranked),
            top_k_used=cfg.final_top_k,
            query_variants_used=[v.variant_name for v in variants],
        )

        log.info(
            "ensemble.done",
            total_candidates=total_candidates,
            post_rrf=len(fused),
            post_mmr=len(diverse),
            final=len(reranked),
            latency_ms=round(latency * 1000),
        )
        return reranked, diagnostics

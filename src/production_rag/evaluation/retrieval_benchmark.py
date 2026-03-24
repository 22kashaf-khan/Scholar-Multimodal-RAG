"""Retrieval configuration comparison benchmark.

Progressively adds retrieval components (dense → BM25 → hybrid → RRF → MMR → rerank)
and measures recall@5, recall@10, MRR, nDCG@10, and latency_ms at each stage.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class BenchmarkResult:
    stage: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    avg_latency_ms: float
    n_queries: int


def _recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for r in retrieved[:k] if r in relevant) / max(len(relevant), 1)


def _mrr(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, 1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def _ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    gains = np.array([1.0 if r in relevant else 0.0 for r in retrieved[:k]])
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))
    ideal = np.ones(min(len(relevant), k))
    idcg = float(np.sum(ideal / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


async def _run_stage(
    stage_name: str,
    retrieval_fn,  # async callable(query) -> list[RetrievedChunk]
    queries: list[dict],
) -> BenchmarkResult:
    recalls5, recalls10, mrrs, ndcgs, latencies = [], [], [], [], []
    for q in queries:
        relevant: set[str] = set(q.get("relevant_chunk_ids", []))
        t0 = time.perf_counter()
        chunks = await retrieval_fn(q["question"])
        latency = (time.perf_counter() - t0) * 1000.0
        ids = [c.chunk.chunk_id for c in chunks]
        recalls5.append(_recall_at_k(ids, relevant, 5))
        recalls10.append(_recall_at_k(ids, relevant, 10))
        mrrs.append(_mrr(ids, relevant))
        ndcgs.append(_ndcg(ids, relevant, 10))
        latencies.append(latency)

    return BenchmarkResult(
        stage=stage_name,
        recall_at_5=float(np.mean(recalls5)) if recalls5 else 0.0,
        recall_at_10=float(np.mean(recalls10)) if recalls10 else 0.0,
        mrr=float(np.mean(mrrs)) if mrrs else 0.0,
        ndcg_at_10=float(np.mean(ndcgs)) if ndcgs else 0.0,
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        n_queries=len(queries),
    )


def _print_table(results: list[BenchmarkResult]) -> None:
    header = f"{'Stage':<30} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@10':>8} {'ms':>8}"
    print("\n" + header)
    print("─" * len(header))
    for r in results:
        print(
            f"{r.stage:<30} {r.recall_at_5:>6.4f} {r.recall_at_10:>6.4f} "
            f"{r.mrr:>6.4f} {r.ndcg_at_10:>8.4f} {r.avg_latency_ms:>8.1f}"
        )


async def _async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant-id", default="eval")
    parser.add_argument("--queries-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval_results/retrieval_benchmark.json"))
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    from production_rag.core.config import get_settings
    from production_rag.core.llm_client import get_llm_client
    from production_rag.core.logging import setup_logging
    from production_rag.core.types import RetrievedChunk
    from production_rag.ingestion.embedder import get_embedder
    from production_rag.retrieval.fusion.mmr import maximal_marginal_relevance
    from production_rag.retrieval.fusion.rrf import reciprocal_rank_fusion
    from production_rag.retrieval.rerankers.rerankers import get_reranker
    from production_rag.retrieval.retrievers.weaviate_retrievers import (
        BM25Retriever,
        DenseRetriever,
        HybridRetriever,
    )
    from production_rag.vectorstore.weaviate_client import get_weaviate_client

    setup_logging()
    settings = get_settings()
    weaviate = await get_weaviate_client()
    embedder = get_embedder(settings)
    llm = await get_llm_client()  # noqa: F841 — unused in benchmark, kept for parity
    reranker = get_reranker(settings)
    tenant = args.tenant_id
    K = args.top_k

    dense_r = DenseRetriever(weaviate)
    bm25_r = BM25Retriever(weaviate)
    hybrid_r = HybridRetriever(weaviate)

    if args.queries_file and args.queries_file.exists():
        queries = [json.loads(l) for l in args.queries_file.read_text().splitlines() if l.strip()]
    else:
        queries = [{"question": "What are transformer attention mechanisms?", "relevant_chunk_ids": []}]
        log.warning("No queries file; metrics will be 0 without relevant_chunk_ids")

    # Stage 1: Dense-only
    async def dense_only(query: str) -> list[RetrievedChunk]:
        vec = await embedder.aembed_query(query)
        return await dense_r.retrieve(query, query_vector=vec, tenant_id=tenant, top_k=K)

    # Stage 2: BM25-only
    async def bm25_only(query: str) -> list[RetrievedChunk]:
        return await bm25_r.retrieve(query, query_vector=None, tenant_id=tenant, top_k=K)

    # Stage 3: Hybrid
    async def hybrid_only(query: str) -> list[RetrievedChunk]:
        vec = await embedder.aembed_query(query)
        return await hybrid_r.retrieve(query, query_vector=vec, tenant_id=tenant, top_k=K)

    # Stage 4: Hybrid + RRF
    async def hybrid_rrf(query: str) -> list[RetrievedChunk]:
        vec = await embedder.aembed_query(query)
        d = await dense_r.retrieve(query, query_vector=vec, tenant_id=tenant, top_k=K * 3)
        b = await bm25_r.retrieve(query, query_vector=None, tenant_id=tenant, top_k=K * 3)
        h = await hybrid_r.retrieve(query, query_vector=vec, tenant_id=tenant, top_k=K * 3)
        return reciprocal_rank_fusion([d, b, h], k=60, top_n=K)

    # Stage 5: Hybrid + RRF + MMR
    async def hybrid_rrf_mmr(query: str) -> list[RetrievedChunk]:
        fused = await hybrid_rrf(query)
        return maximal_marginal_relevance(fused, top_k=K, lambda_=0.7)

    # Stage 6: Hybrid + RRF + MMR + Rerank
    async def hybrid_rrf_mmr_rerank(query: str) -> list[RetrievedChunk]:
        diversified = await hybrid_rrf_mmr(query)
        reranked = await reranker.rerank(query, diversified, top_k=K)
        return reranked

    stages = [
        ("1_dense_only", dense_only),
        ("2_bm25_only", bm25_only),
        ("3_hybrid", hybrid_only),
        ("4_hybrid_rrf", hybrid_rrf),
        ("5_hybrid_rrf_mmr", hybrid_rrf_mmr),
        ("6_hybrid_rrf_mmr_rerank", hybrid_rrf_mmr_rerank),
    ]

    results: list[BenchmarkResult] = []
    for name, fn in stages:
        print(f"▶  Stage: {name}")
        r = await _run_stage(name, fn, queries)
        results.append(r)

    _print_table(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\n✔  Results → {args.output}")


if __name__ == "__main__":
    asyncio.run(_async_main())

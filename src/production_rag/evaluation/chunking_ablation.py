"""Chunking strategy ablation study.

Runs ingestion + retrieval for each strategy and measures recall@K, MRR, nDCG@10,
faithfulness, and latency_ms per strategy.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from production_rag.core.config import get_settings
from production_rag.ingestion.chunkers.factory import get_chunker, ChunkStrategy

log = structlog.get_logger(__name__)

STRATEGIES = [
    ChunkStrategy.FIXED,
    ChunkStrategy.RECURSIVE,
    ChunkStrategy.SEMANTIC,
    ChunkStrategy.HIERARCHICAL,
    ChunkStrategy.LATE,
]


@dataclass
class StrategyResult:
    strategy: str
    recall_at_5: float
    recall_at_10: float
    mrr: float
    ndcg_at_10: float
    faithfulness: float | None
    avg_latency_ms: float
    n_queries: int



def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    hits = sum(1 for rid in retrieved_ids[:k] if rid in relevant_ids)
    return hits / max(len(relevant_ids), 1)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    gains = np.array(
        [1.0 if rid in relevant_ids else 0.0 for rid in retrieved_ids[:k]]
    )
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal_gains = np.ones(min(len(relevant_ids), k))
    ideal_discounts = 1.0 / np.log2(np.arange(2, len(ideal_gains) + 2))
    idcg = float(np.sum(ideal_gains * ideal_discounts))

    return dcg / idcg if idcg > 0 else 0.0



async def evaluate_strategy(
    strategy: ChunkStrategy,
    documents: list[Any],
    queries: list[dict[str, Any]],
    tenant_suffix: str,
) -> StrategyResult:
    """Ingest docs with given chunking strategy, retrieves for each query."""
    from production_rag.core.llm_client import get_llm_client
    from production_rag.ingestion.embedder import get_embedder
    from production_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
    from production_rag.retrieval.retrievers.ensemble import EnsembleConfig, EnsembleRetriever
    from production_rag.retrieval.rerankers.rerankers import get_reranker
    from production_rag.vectorstore.weaviate_client import get_weaviate_client

    settings = get_settings()
    weaviate = await get_weaviate_client()
    embedder = get_embedder(settings)
    llm = await get_llm_client()
    tenant_id = f"ablation_{strategy.value}_{tenant_suffix}"

    pipeline = IngestionPipeline(
        weaviate_client=weaviate,
        embedder=embedder,
        settings=settings,
    )
    config = IngestionConfig(
        tenant_id=tenant_id,
        chunk_strategy=strategy,
        raptor_enabled=False,  # isolate chunking effect
    )
    await pipeline.ingest(documents, config)


    ensemble = EnsembleRetriever(
        weaviate_client=weaviate,
        embedder=embedder,
        llm=llm,
        reranker=get_reranker(settings),
        settings=settings,
    )
    cfg = EnsembleConfig(
        tenant_id=tenant_id,
        top_k_candidates=50,
        rrf_k=60,
        mmr_lambda=0.7,
        rerank_top_n=20,
        final_top_k=10,
        use_parent_expansion=False,
    )

    recalls_5, recalls_10, mrrs, ndcgs, latencies = [], [], [], [], []

    for q in queries:
        relevant: set[str] = set(q.get("relevant_chunk_ids", []))
        t0 = time.perf_counter()
        chunks, _ = await ensemble.retrieve(q["question"], cfg)
        latency = (time.perf_counter() - t0) * 1000.0

        retrieved_ids = [c.chunk.chunk_id for c in chunks]
        recalls_5.append(recall_at_k(retrieved_ids, relevant, 5))
        recalls_10.append(recall_at_k(retrieved_ids, relevant, 10))
        mrrs.append(mrr(retrieved_ids, relevant))
        ndcgs.append(ndcg_at_k(retrieved_ids, relevant, 10))
        latencies.append(latency)

    return StrategyResult(
        strategy=strategy.value,
        recall_at_5=float(np.mean(recalls_5)) if recalls_5 else 0.0,
        recall_at_10=float(np.mean(recalls_10)) if recalls_10 else 0.0,
        mrr=float(np.mean(mrrs)) if mrrs else 0.0,
        ndcg_at_10=float(np.mean(ndcgs)) if ndcgs else 0.0,
        faithfulness=None,  # optionally populated downstream
        avg_latency_ms=float(np.mean(latencies)) if latencies else 0.0,
        n_queries=len(queries),
    )



def _print_table(results: list[StrategyResult]) -> None:
    header = f"{'Strategy':<15} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'nDCG@10':>8} {'Latency(ms)':>12}"
    print("\n" + header)
    print("─" * len(header))
    for r in sorted(results, key=lambda x: x.mrr, reverse=True):
        print(
            f"{r.strategy:<15} {r.recall_at_5:>6.4f} {r.recall_at_10:>6.4f} "
            f"{r.mrr:>6.4f} {r.ndcg_at_10:>8.4f} {r.avg_latency_ms:>12.1f}"
        )


async def _async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arxiv-ids", nargs="+", default=["2312.10997"])
    parser.add_argument("--queries-file", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("eval_results/chunking_ablation.json"))
    parser.add_argument("--strategies", nargs="+", default=[s.value for s in STRATEGIES])
    args = parser.parse_args()

    from production_rag.core.logging import setup_logging
    from production_rag.ingestion.loaders.arxiv_loader import ArXivLoader

    setup_logging()

    # Load documents
    loader = ArXivLoader()
    documents = await loader.load(args.arxiv_ids)
    print(f"Loaded {len(documents)} documents from ArXiv")

    # Load queries
    if args.queries_file and args.queries_file.exists():
        queries = [json.loads(l) for l in args.queries_file.read_text().splitlines() if l.strip()]
    else:
        # Minimal synthetic queries if no file provided
        queries = [
            {"question": f"What is the main contribution of document {i}?", "relevant_chunk_ids": []}
            for i, _ in enumerate(documents)
        ]
        log.warning("No queries file provided; metrics will be 0 without relevant_chunk_ids")

    selected = [ChunkStrategy(v) for v in args.strategies if v in [s.value for s in STRATEGIES]]
    results: list[StrategyResult] = []
    import uuid
    run_id = uuid.uuid4().hex[:8]

    for strategy in selected:
        print(f"\n▶  Evaluating strategy: {strategy.value}")
        result = await evaluate_strategy(strategy, documents, queries, run_id)
        results.append(result)
        print(f"   R@5={result.recall_at_5:.4f}  MRR={result.mrr:.4f}  Latency={result.avg_latency_ms:.0f}ms")

    _print_table(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([asdict(r) for r in results], indent=2))
    print(f"\n✔  Results saved → {args.output}")


if __name__ == "__main__":
    asyncio.run(_async_main())

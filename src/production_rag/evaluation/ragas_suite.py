"""RAGAS evaluation suite for the production RAG pipeline.

Evaluates the full RAG chain on QASPER and SciQ datasets.
Results are written to `eval_results/ragas_<timestamp>.json` for CI regression tracking.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

RESULTS_DIR = Path("eval_results")
PASS_THRESHOLDS = {
    "faithfulness": 0.85,
    "answer_relevancy": 0.80,
    "context_recall": 0.80,
    "context_precision": 0.70,
}



def load_qasper(n: int = 200) -> list[dict[str, Any]]:
    """Load QASPER dev split.  Returns list of {question, answer, contexts}."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    ds = load_dataset("allenai/qasper", split="validation", trust_remote_code=True)
    samples: list[dict[str, Any]] = []
    for row in ds:
        for qa in row.get("qas", {}).get("question", []):
            if len(samples) >= n:
                break
            answer_list = row["qas"]["answers"][len(samples) % max(1, len(row["qas"]["question"]))]
            answers = answer_list.get("answer", [])
            if not answers:
                continue
            free_text = answers[0].get("free_form_answer", "")
            if not free_text:
                continue
            # Flatten full text as ground-truth context
            full_text = " ".join(
                " ".join(p.get("paragraphs", []))
                for p in row.get("full_text", {}).get("section_name", [])
            )[:8000]
            samples.append({
                "question": qa,
                "ground_truth": free_text,
                "reference_contexts": [full_text],
            })
        if len(samples) >= n:
            break
    return samples[:n]


def load_sciq(n: int = 200) -> list[dict[str, Any]]:
    """Load SciQ test split. Returns list of {question, answer, contexts}."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("pip install datasets") from exc

    ds = load_dataset("sciq", split="test", trust_remote_code=True)
    samples = []
    for row in ds:
        if len(samples) >= n:
            break
        samples.append({
            "question": row["question"],
            "ground_truth": row["correct_answer"],
            "reference_contexts": [row["support"]],
        })
    return samples[:n]



async def _run_pipeline_on_sample(
    chain: Any,  # RAGChain
    sample: dict[str, Any],
    tenant_id: str = "eval",
) -> dict[str, Any]:
    """Run the RAG chain on one sample and return RAGAS-formatted dict."""
    try:
        response = await chain.invoke(
            query=sample["question"],
            tenant_id=tenant_id,
            enable_crag=True,
            enable_self_rag=False,  # disable Self-RAG for speed in eval
        )
        return {
            "question": sample["question"],
            "answer": response.answer,
            "contexts": [c.display_text for c in response.chunks],
            "ground_truth": sample["ground_truth"],
            "reference_contexts": sample.get("reference_contexts", []),
        }
    except Exception:
        log.exception("ragas_suite.sample_failed", question=sample["question"][:80])
        return {
            "question": sample["question"],
            "answer": "",
            "contexts": [],
            "ground_truth": sample["ground_truth"],
            "reference_contexts": sample.get("reference_contexts", []),
        }


async def build_ragas_dataset(
    chain: Any,
    samples: list[dict[str, Any]],
    concurrency: int = 4,
) -> list[dict[str, Any]]:
    """Run all samples through the pipeline with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)

    async def _run_with_sem(sample: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _run_pipeline_on_sample(chain, sample)

    tasks = [_run_with_sem(s) for s in samples]
    results = await asyncio.gather(*tasks)
    return list(results)



def run_ragas_metrics(
    ragas_samples: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute RAGAS metrics.  Returns {metric_name: score} dict."""
    try:
        from ragas import evaluate, RunConfig  # type: ignore[import-untyped]
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
            NoiseSensitivity,
        )
    except ImportError as exc:
        raise ImportError("pip install ragas") from exc

    dataset = EvaluationDataset(
        samples=[
            SingleTurnSample(
                user_input=s["question"],
                response=s["answer"],
                retrieved_contexts=s["contexts"],
                reference=s["ground_truth"],
                reference_contexts=s.get("reference_contexts", []),
            )
            for s in ragas_samples
            if s["answer"]  # skip failed calls
        ]
    )

    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            NoiseSensitivity(),
        ],
        run_config=RunConfig(max_workers=4),
    )

    scores: dict[str, float] = {k: float(v) for k, v in result.items()}
    return scores



def check_pass_gate(scores: dict[str, float]) -> bool:
    """Return True if all monitored metrics are above threshold."""
    passed = True
    for metric, threshold in PASS_THRESHOLDS.items():
        score = scores.get(metric, 0.0)
        ok = score >= threshold
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {metric}: {score:.4f} (threshold {threshold:.2f})")
        if not ok:
            passed = False
    return passed



def save_results(
    scores: dict[str, float],
    dataset_name: str,
    n_samples: int,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"ragas_{dataset_name}_{ts}.json"
    out.write_text(
        json.dumps(
            {
                "timestamp": ts,
                "dataset": dataset_name,
                "n_samples": n_samples,
                "scores": scores,
                "thresholds": PASS_THRESHOLDS,
            },
            indent=2,
        )
    )
    log.info("ragas_suite.results_saved", path=str(out))
    return out



def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAGAS evaluation suite")
    p.add_argument("--dataset", choices=["qasper", "sciq"], default="qasper")
    p.add_argument("--n", type=int, default=200, help="Number of eval samples")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--tenant-id", default="eval")
    p.add_argument("--fail-fast", action="store_true", help="Exit 1 on CI gate failure")
    return p.parse_args()


async def _async_main() -> None:
    args = _parse_args()

    # lazy import to avoid circular deps at module level
    from production_rag.core.config import get_settings
    from production_rag.core.llm_client import get_llm_client
    from production_rag.core.logging import setup_logging
    from production_rag.ingestion.embedder import get_embedder
    from production_rag.vectorstore.weaviate_client import get_weaviate_client
    from production_rag.chains.rag_chain import RAGChain

    setup_logging()
    settings = get_settings()
    weaviate = await get_weaviate_client()
    embedder = get_embedder(settings)
    llm = await get_llm_client()
    chain = RAGChain(weaviate, embedder, llm, settings)

    print(f"\n▶  RAGAS evaluation | dataset={args.dataset} n={args.n}")

    loader = load_qasper if args.dataset == "qasper" else load_sciq
    samples = loader(args.n)
    print(f"   Loaded {len(samples)} samples")

    ragas_samples = await build_ragas_dataset(chain, samples, args.concurrency)
    scores = run_ragas_metrics(ragas_samples)

    print("\n── Scores ──────────────────────────────────────────────────")
    for k, v in scores.items():
        print(f"  {k}: {v:.4f}")

    print("\n── CI Gate ─────────────────────────────────────────────────")
    passed = check_pass_gate(scores)

    path = save_results(scores, args.dataset, len(samples))
    print(f"\n✔  Results saved → {path}")

    if args.fail_fast and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(_async_main())

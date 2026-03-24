"""Retrieval quality estimator for CRAG gating.

Returns a scalar quality_score ∈ [0, 1] combining cosine similarity,
score entropy, and coverage. Scores below threshold trigger re-query.
"""

from __future__ import annotations

import math

import numpy as np

from production_rag.core.types import RetrievedChunk


def _cosine_sim(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def _score_entropy(scores: list[float]) -> float:
    """Return normalised Shannon entropy of score distribution.
    Higher = more uniform (good). Lower = dominated by one chunk (bad quality signal).
    """
    arr = np.array(scores, dtype=np.float64)
    arr = arr - arr.min()
    total = arr.sum()
    if total == 0:
        return 0.0
    probs = arr / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = math.log(len(scores)) if len(scores) > 1 else 1.0
    return entropy / max_entropy if max_entropy > 0 else 0.0


class RetrievalQualityEstimator:
    """Estimate retrieval quality for CRAG gating.

    Args:
        min_score_threshold: Minimum rerank/rrf score considered useful.
        entropy_weight: Weight assigned to entropy component (0–1).
        sim_weight: Weight assigned to mean-similarity component.
        coverage_weight: Weight assigned to coverage component.
    """

    def __init__(
        self,
        min_score_threshold: float = 0.05,
        entropy_weight: float = 0.2,
        sim_weight: float = 0.5,
        coverage_weight: float = 0.3,
    ) -> None:
        self._min_threshold = min_score_threshold
        self._w_entropy = entropy_weight
        self._w_sim = sim_weight
        self._w_coverage = coverage_weight

    def score(
        self,
        chunks: list[RetrievedChunk],
        query_vector: list[float],
    ) -> float:
        """Return quality_score ∈ [0, 1].

        0 → poor retrieval (trigger CRAG re-query)
        1 → excellent retrieval
        """
        if not chunks:
            return 0.0


        chunk_sims = [
            _cosine_sim(query_vector, c.chunk.embedding)
            for c in chunks
            if c.chunk.embedding
        ]
        mean_sim = float(np.mean(chunk_sims)) if chunk_sims else 0.0

        scores = [c.rerank_score or c.rrf_score for c in chunks]
        entropy = _score_entropy(scores)

        above_threshold = sum(1 for s in scores if s > self._min_threshold)
        coverage = above_threshold / len(chunks)

        quality = (
            self._w_sim * mean_sim
            + self._w_entropy * entropy
            + self._w_coverage * coverage
        )

        return float(np.clip(quality, 0.0, 1.0))

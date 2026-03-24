r"""Maximal Marginal Relevance (MMR) diversification.

MMR criterion: arg max_{d in C \ S}  [λ · Rel(d,q) − (1−λ) · max_{s in S} Sim(d,s)]
where Rel uses normalised rrf_score and Sim is cosine similarity of embeddings.
"""

from __future__ import annotations

import numpy as np

from production_rag.core.types import RetrievedChunk


def _cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute n×n cosine similarity matrix for a set of L2-normalised embeddings."""
    # Normalise rows
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-10, norms)
    normalised = embeddings / norms
    return normalised @ normalised.T  # type: ignore[return-value]


def maximal_marginal_relevance(
    candidates: list[RetrievedChunk],
    top_k: int,
    lambda_: float = 0.7,
) -> list[RetrievedChunk]:
    """Apply MMR to a candidate pool and return top_k diverse results.

    Candidates MUST have their `chunk.embedding` populated.  If embeddings
    are missing the function falls back to returning the top_k by rrf_score.

    Args:
        candidates: Fused candidate pool (typically from RRF), score-sorted.
        top_k: Number of chunks to select.
        lambda_: Relevance weight. 0.7 = 70% relevance, 30% novelty.

    Returns:
        List of top_k RetrievedChunks selected by MMR, re-ranked by selection order.
    """
    if not candidates:
        return []

    # Filter to those with embeddings
    valid = [c for c in candidates if c.chunk.embedding]
    if not valid:
        return candidates[:top_k]

    emb_matrix = np.array([c.chunk.embedding for c in valid], dtype=np.float32)
    sim_matrix = _cosine_matrix(emb_matrix)

    scores = np.array([c.rrf_score for c in valid], dtype=np.float32)
    score_range = scores.max() - scores.min()
    if score_range > 0:
        rel_scores = (scores - scores.min()) / score_range
    else:
        rel_scores = np.ones(len(valid), dtype=np.float32)

    n = len(valid)
    k = min(top_k, n)
    selected_indices: list[int] = []
    remaining = set(range(n))

    for _ in range(k):
        if not remaining:
            break

        if not selected_indices:
            best = max(remaining, key=lambda i: rel_scores[i])
        else:
            best = max(
                remaining,
                key=lambda i: (
                    lambda_ * rel_scores[i]
                    - (1.0 - lambda_) * sim_matrix[i, selected_indices].max()
                ),
            )

        selected_indices.append(best)
        remaining.discard(best)

    result = [valid[i] for i in selected_indices]
    for i, c in enumerate(result):
        c.rank = i
    return result

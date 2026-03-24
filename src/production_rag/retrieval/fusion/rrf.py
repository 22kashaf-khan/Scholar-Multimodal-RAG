"""Reciprocal Rank Fusion (RRF).

RRF(d) = Σ 1 / (k + r_i(d))  where r_i(d) is rank in list i.
Robust to incomparable score scales; no calibration required.
"""

from __future__ import annotations

from collections import defaultdict

from production_rag.core.types import RetrievedChunk


def reciprocal_rank_fusion(
    rank_lists: list[list[RetrievedChunk]],
    k: int = 60,
    top_n: int | None = None,
) -> list[RetrievedChunk]:
    """Fuse multiple rank lists into one using RRF.

    Args:
        rank_lists: Each list is already sorted by score (best first).
        k: RRF smoothing constant. k=60 is the standard default.
        top_n: If set, return only the top_n candidates.

    Returns:
        Sorted list of unique RetrievedChunks (best RRF score first),
        with `rrf_score` populated.
    """
    rrf_scores: dict[str, float] = defaultdict(float)
    best_chunk: dict[str, RetrievedChunk] = {}
    provenance: dict[str, list[str]] = defaultdict(list)

    for rank_list in rank_lists:
        for rank, chunk in enumerate(rank_list, start=1):
            cid = chunk.chunk.chunk_id
            rrf_scores[cid] += 1.0 / (k + rank)
            provenance[cid].append(f"{chunk.retriever_type.value}:{chunk.query_variant}")

            if cid not in best_chunk or chunk.score > best_chunk[cid].score:
                best_chunk[cid] = chunk

    fused: list[RetrievedChunk] = []
    for cid, rrf_score in rrf_scores.items():
        candidate = best_chunk[cid]
        candidate.rrf_score = rrf_score
        # Embed provenance in query_variant field (multi-source)
        candidate.query_variant = ";".join(set(provenance[cid]))
        fused.append(candidate)

    fused.sort(key=lambda c: c.rrf_score, reverse=True)

    if top_n is not None:
        fused = fused[:top_n]

    for i, c in enumerate(fused):
        c.rank = i

    return fused

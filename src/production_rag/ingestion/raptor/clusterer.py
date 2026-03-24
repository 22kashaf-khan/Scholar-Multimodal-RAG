"""RAPTOR clusterer: UMAP dimensionality reduction + GMM soft clustering."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import numpy as np
import structlog

log = structlog.get_logger(__name__)


@dataclass
class ClusterResult:
    assignments: dict[int, list[int]]       # cluster_id → list of chunk indices
    n_clusters: int
    global_cluster_id: int = -1             # -1 if global cluster not created


class RAPTORClusterer:
    """UMAP + GMM clusterer for RAPTOR tree construction.

    Args:
        n_components: UMAP output dimensions.
        n_neighbors: UMAP n_neighbors parameter.
        max_cluster_size: Target max chunks per cluster; used to pick k.
        threshold: Minimum GMM posterior probability for assignment.
    """

    def __init__(
        self,
        n_components: int = 10,
        n_neighbors: int = 15,
        max_cluster_size: int = 10,
        threshold: float = 0.5,
    ) -> None:
        self._n_components = n_components
        self._n_neighbors = n_neighbors
        self._max_cluster_size = max_cluster_size
        self._threshold = threshold

    async def cluster(
        self, embeddings: list[list[float]]
    ) -> ClusterResult:
        """Cluster embeddings asynchronously (compute-heavy, runs in thread pool)."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._cluster_sync, embeddings)

    def _cluster_sync(self, embeddings: list[list[float]]) -> ClusterResult:
        try:
            import umap  # type: ignore[import-untyped]
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import normalize
        except ImportError as exc:
            raise ImportError(
                "RAPTOR requires: pip install umap-learn scikit-learn"
            ) from exc

        n = len(embeddings)
        if n == 0:
            return ClusterResult(assignments={0: []}, n_clusters=0)

        # If corpus is tiny, single global cluster
        if n <= self._max_cluster_size:
            log.debug("raptor.clusterer.single_cluster", n=n)
            return ClusterResult(
                assignments={0: list(range(n))},
                n_clusters=1,
                global_cluster_id=0,
            )

        emb_array = normalize(np.array(embeddings), norm="l2")


        n_neighbors = min(self._n_neighbors, n - 1)
        reducer = umap.UMAP(
            n_components=min(self._n_components, n - 2),
            n_neighbors=n_neighbors,
            metric="cosine",
            min_dist=0.0,
            random_state=42,
        )
        reduced = reducer.fit_transform(emb_array)


        k = max(1, n // self._max_cluster_size)
        k = min(k, min(n // 2, 50))  # hard cap to avoid degenerate models

        best_gmm = None
        best_bic = float("inf")
        for n_components in range(max(1, k - 2), k + 3):
            if n_components >= n:
                break
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type="full",
                random_state=42,
                max_iter=200,
            )
            gmm.fit(reduced)
            bic = gmm.bic(reduced)
            if bic < best_bic:
                best_bic = bic
                best_gmm = gmm

        if best_gmm is None:
            return ClusterResult(
                assignments={0: list(range(n))}, n_clusters=1, global_cluster_id=0
            )

        probs = best_gmm.predict_proba(reduced)
        assignments: dict[int, list[int]] = {
            c: [] for c in range(best_gmm.n_components)
        }
        for idx, row in enumerate(probs):
            best_cluster = int(np.argmax(row))
            assignments[best_cluster].append(idx)
            for c, p in enumerate(row):
                if c != best_cluster and p >= self._threshold:
                    assignments[c].append(idx)

        # Remove empty clusters
        assignments = {c: v for c, v in assignments.items() if v}

        log.info(
            "raptor.clusterer.done",
            n_chunks=n,
            n_clusters=len(assignments),
            bic=round(best_bic, 2),
        )
        return ClusterResult(
            assignments=assignments,
            n_clusters=len(assignments),
        )

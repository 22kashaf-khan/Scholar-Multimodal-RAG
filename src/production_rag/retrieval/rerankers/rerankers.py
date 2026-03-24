"""Reranker implementations.

Stage 2 reranking: takes the MMR-diversified candidate pool,
sends top-N to the reranker, returns top-K for synthesis.

Implementations:
- CohereReranker: Cohere Rerank API (Phase 1, recommended default).
- BGEReranker: Self-hosted BGE cross-encoder (Phase 2, for high-volume / privacy).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

import structlog

from production_rag.core.config import Settings, get_settings
from production_rag.core.types import RetrievedChunk

log = structlog.get_logger(__name__)


class BaseReranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        ...


class CohereReranker(BaseReranker):
    """Rerank using the Cohere Rerank v3 API.

    Sends `display_text` (expanded or original) as the document.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: object = None

    def _get_client(self) -> object:
        if self._client is None:
            import cohere  # type: ignore[import-untyped]
            self._client = cohere.AsyncClient(
                api_key=self._settings.cohere_api_key.get_secret_value()
            )
        return self._client

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        if not candidates:
            return []

        import cohere  # type: ignore[import-untyped]
        client = self._get_client()
        documents = [c.display_text for c in candidates]

        try:
            response = await client.rerank(  # type: ignore[union-attr]
                model=self._settings.cohere_rerank_model,
                query=query,
                documents=documents,
                top_n=min(top_k, len(documents)),
                return_documents=False,
            )
        except Exception as e:
            log.error("reranker.cohere.failed", error=str(e))
            # Fallback: return candidates sorted by rrf_score
            return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_k]

        reranked: list[RetrievedChunk] = []
        for i, result in enumerate(response.results):
            candidate = candidates[result.index]
            candidate.rerank_score = result.relevance_score
            candidate.rank = i
            reranked.append(candidate)

        log.debug(
            "reranker.cohere.done",
            query_len=len(query),
            input=len(candidates),
            output=len(reranked),
        )
        return reranked


class BGEReranker(BaseReranker):
    """Self-hosted BGE cross-encoder for high-volume / privacy use cases.

    Uses sentence-transformers CrossEncoder locally.
    Falls back to RRF score ordering if model is unavailable.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
    ) -> None:
        self._model_name = model_name
        self._model: object = None

    def _get_model(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
                self._model = CrossEncoder(self._model_name, max_length=512)
                log.info("reranker.bge.loaded", model=self._model_name)
            except Exception as e:
                log.error("reranker.bge.load_failed", error=str(e))
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        model = self._get_model()
        if model is None:
            return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_k]

        pairs = [[query, c.display_text] for c in candidates]

        loop = asyncio.get_event_loop()
        try:
            scores = await loop.run_in_executor(
                None,
                lambda: model.predict(pairs),  # type: ignore[union-attr]
            )
        except Exception as e:
            log.error("reranker.bge.predict_failed", error=str(e))
            return sorted(candidates, key=lambda c: c.rrf_score, reverse=True)[:top_k]

        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)

        reranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)[:top_k]
        for i, c in enumerate(reranked):
            c.rank = i

        return reranked


def get_reranker(settings: Settings | None = None) -> BaseReranker:
    s = settings or get_settings()
    if s.reranker_provider == "bge":
        return BGEReranker(model_name=s.bge_reranker_model)
    return CohereReranker(s)

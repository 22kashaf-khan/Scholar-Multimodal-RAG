"""Provider-agnostic embedding abstraction.

Supports three backends:
- sentence_transformers: local, best for privacy / offline use
- openai: text-embedding-3-large, best quality via API
- jina: jina-embeddings-v3, long-context with late-chunking support

All embedders implement the same interface:
  embed_texts(texts: list[str]) -> list[list[float]]
  embed_query(text: str) -> list[float]

`embed_query` may apply a query-specific instruction prefix (BGE/Jina models
often improve with "Represent this sentence: " or similar).
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog

from production_rag.core.config import Settings, get_settings

log = structlog.get_logger(__name__)


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts (for ingestion)."""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string (may apply query instruction prefix)."""
        ...

    async def aembed_texts(self, texts: list[str]) -> list[list[float]]:
        """Async wrapper — runs sync embed_texts in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_texts, texts)

    async def aembed_query(self, text: str) -> list[float]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.embed_query, text)

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformersEmbedder(BaseEmbedder):
    """Embedding via sentence-transformers (local, no API key required).

    For BGE models, apply the recommended query instruction prefix.
    """

    QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages:"

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        self._model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            log.info("embedder.loaded", model=self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        return self._get_model().get_sentence_embedding_dimension()  # type: ignore[no-any-return]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        model = self._get_model()
        embs: np.ndarray = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embs.tolist()

    def embed_query(self, text: str) -> list[float]:
        # BGE models benefit from prefixing queries
        if "bge" in self._model_name.lower():
            text = f"{self.QUERY_INSTRUCTION} {text}"
        model = self._get_model()
        emb: np.ndarray = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.tolist()  # type: ignore[return-value]


class OpenAIEmbedder(BaseEmbedder):
    """Embedding via OpenAI API (text-embedding-3-large or similar)."""

    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: str = "",
        dimensions: int = 1024,  # text-embedding-3-large supports truncation
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._dimensions = dimensions
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._api_key or None)
        return self._client

    @property
    def dimension(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        client = self._get_client()
        # OpenAI API max 2048 texts per batch; handle in chunks
        result: list[list[float]] = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            result.extend([d.embedding for d in response.data])
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class JinaEmbedder(BaseEmbedder):
    """Jina embeddings via sentence-transformers (long-context)."""

    def __init__(
        self,
        model_name: str = "jinaai/jina-embeddings-v3",
        dimensions: int = 1024,
    ) -> None:
        self._model_name = model_name
        self._dimensions = dimensions
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._model_name, trust_remote_code=True
            )
        return self._model

    @property
    def dimension(self) -> int:
        return self._dimensions

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embs: np.ndarray = self._get_model().encode(
            texts,
            task="retrieval.passage",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()

    def embed_query(self, text: str) -> list[float]:
        embs: np.ndarray = self._get_model().encode(
            [text],
            task="retrieval.query",
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs[0].tolist()


def get_embedder(settings: Settings | None = None) -> BaseEmbedder:
    """Return a configured embedder based on settings."""
    s = settings or get_settings()
    provider = s.embedding_provider
    model = s.embedding_model
    dim = s.embedding_dimension

    if provider == "openai":
        key = s.openai_api_key.get_secret_value()
        return OpenAIEmbedder(model=model, api_key=key, dimensions=dim)
    if provider == "jina":
        return JinaEmbedder(model_name=model, dimensions=dim)
    # Default: sentence_transformers
    return SentenceTransformersEmbedder(model_name=model)


_embedder_instance: BaseEmbedder | None = None


def get_default_embedder() -> BaseEmbedder:
    """Return a module-level singleton embedder."""
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = get_embedder()
    return _embedder_instance

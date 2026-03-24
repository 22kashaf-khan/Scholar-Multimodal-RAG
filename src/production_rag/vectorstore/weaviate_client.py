"""Weaviate v4 async client wrapper.

Exposes schema management (create / verify / drop) and
the CRUD + search operations used by retrievers and the ingestion pipeline.

Thread-safety: Weaviate client v4 is not thread-safe; callers must use
asyncio and avoid sharing a single client across threads.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import structlog
import weaviate
import weaviate.classes as wvc
from weaviate.classes.query import Filter, MetadataQuery
from weaviate.exceptions import UnexpectedStatusCodeError

from production_rag.core.config import Settings, get_settings
from production_rag.core.types import Chunk, RetrievedChunk, RetrieverType
from production_rag.vectorstore.schema import COLLECTION_NAME, build_collection_config

log = structlog.get_logger(__name__)


def _chunk_to_properties(chunk: Chunk) -> dict[str, Any]:
    """Convert a Chunk dataclass into Weaviate property dict."""
    return {
        "chunk_text": chunk.chunk_text,
        "title": chunk.title,
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "parent_chunk_id": chunk.parent_chunk_id or "",
        "chunk_level": chunk.chunk_level.value,
        "chunk_index": chunk.chunk_index,
        "section_index": chunk.section_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
        "page": chunk.page,
        "arxiv_id": chunk.arxiv_id,
        "paper_doi": chunk.paper_doi,
        "publication_year": chunk.publication_year,
        "authors": chunk.authors,
        "tenant_id": chunk.tenant_id,
    }


def _prop_to_chunk(props: dict[str, Any]) -> Chunk:
    """Reconstruct a Chunk from Weaviate returned properties."""
    from production_rag.core.types import ChunkLevel

    return Chunk(
        chunk_text=props.get("chunk_text", ""),
        chunk_id=props.get("chunk_id", ""),
        doc_id=props.get("doc_id", ""),
        parent_chunk_id=props.get("parent_chunk_id") or None,
        chunk_level=ChunkLevel(props.get("chunk_level", "leaf")),
        chunk_index=props.get("chunk_index", 0),
        section_index=props.get("section_index", 0),
        start_char=props.get("start_char", 0),
        end_char=props.get("end_char", 0),
        page=props.get("page", 0),
        arxiv_id=props.get("arxiv_id", ""),
        paper_doi=props.get("paper_doi", ""),
        publication_year=props.get("publication_year", 0),
        title=props.get("title", ""),
        authors=props.get("authors", []),
        tenant_id=props.get("tenant_id", "default"),
    )


class WeaviateClient:
    """Thin async wrapper around the Weaviate v4 synchronous client.

    Weaviate v4 does not have a native async client; we run blocking
    operations in a thread-pool executor to stay non-blocking in FastAPI.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client: weaviate.WeaviateClient | None = None

    def _get_client(self) -> weaviate.WeaviateClient:
        if self._client is None or not self._client.is_connected():
            raise RuntimeError("WeaviateClient not connected. Call connect() first.")
        return self._client

    async def connect(self) -> None:
        """Establish connection to Weaviate."""
        s = self._settings
        api_key = s.weaviate_api_key.get_secret_value()

        def _connect() -> weaviate.WeaviateClient:
            auth = weaviate.auth.AuthApiKey(api_key) if api_key else None
            client = weaviate.connect_to_local(
                host=s.weaviate_url.replace("http://", "").split(":")[0],
                port=int(s.weaviate_url.split(":")[-1]),
                auth_credentials=auth,
            )
            return client

        loop = asyncio.get_event_loop()
        self._client = await loop.run_in_executor(None, _connect)
        log.info("weaviate.connected", url=s.weaviate_url)

    async def close(self) -> None:
        if self._client and self._client.is_connected():
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._client.close)
            log.info("weaviate.disconnected")


    async def create_schema(self, embedding_dimension: int = 1024) -> None:
        """Create the ScientificChunk collection. Idempotent."""
        client = self._get_client()
        cfg = build_collection_config(embedding_dimension)

        def _create() -> None:
            if client.collections.exists(COLLECTION_NAME):
                log.info("weaviate.schema.exists", collection=COLLECTION_NAME)
                return
            client.collections.create(**cfg)
            log.info("weaviate.schema.created", collection=COLLECTION_NAME)

        await asyncio.get_event_loop().run_in_executor(None, _create)

    async def drop_schema(self) -> None:
        """Drop the ScientificChunk collection. DESTRUCTIVE — use in tests only."""
        client = self._get_client()

        def _drop() -> None:
            if client.collections.exists(COLLECTION_NAME):
                client.collections.delete(COLLECTION_NAME)
                log.warning("weaviate.schema.dropped", collection=COLLECTION_NAME)

        await asyncio.get_event_loop().run_in_executor(None, _drop)


    async def upsert_chunks(self, chunks: list[Chunk], tenant_id: str) -> None:
        """Batch-upsert chunks into Weaviate under the given tenant."""
        client = self._get_client()

        def _upsert() -> None:
            collection = client.collections.get(COLLECTION_NAME)
            with collection.batch.fixed_size(batch_size=100) as batch:
                for chunk in chunks:
                    vectors = {}
                    if chunk.embedding:
                        vectors["semantic_vector"] = chunk.embedding
                    if chunk.title_embedding:
                        vectors["title_vector"] = chunk.title_embedding

                    batch.add_object(
                        properties=_chunk_to_properties(chunk),
                        vector=vectors if vectors else None,
                        uuid=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id)),
                        tenant=tenant_id,
                    )

            if hasattr(batch, "failed_objects") and batch.failed_objects:
                log.error(
                    "weaviate.upsert.failures",
                    count=len(batch.failed_objects),
                    tenant=tenant_id,
                )

        await asyncio.get_event_loop().run_in_executor(None, _upsert)
        log.info("weaviate.upsert.done", count=len(chunks), tenant=tenant_id)


    async def search_dense(
        self,
        query_vector: list[float],
        top_k: int,
        tenant_id: str,
        named_vector: str = "semantic_vector",
        filters: Filter | None = None,
    ) -> list[RetrievedChunk]:
        client = self._get_client()

        def _search() -> list[Any]:
            collection = client.collections.get(COLLECTION_NAME)
            result = collection.query.near_vector(
                near_vector=query_vector,
                limit=top_k,
                target_vector=named_vector,
                filters=filters,
                return_metadata=MetadataQuery(distance=True, score=True),
                tenant=tenant_id,
            )
            return result.objects

        objects = await asyncio.get_event_loop().run_in_executor(None, _search)
        return [
            RetrievedChunk(
                chunk=_prop_to_chunk(o.properties),
                score=1.0 - (o.metadata.distance or 1.0),
                retriever_type=RetrieverType.DENSE,
                query_variant="original",
                rank=i,
            )
            for i, o in enumerate(objects)
        ]


    async def search_bm25(
        self,
        query: str,
        top_k: int,
        tenant_id: str,
        query_properties: list[str] | None = None,
        filters: Filter | None = None,
    ) -> list[RetrievedChunk]:
        client = self._get_client()
        props = query_properties or ["chunk_text", "title"]

        def _search() -> list[Any]:
            collection = client.collections.get(COLLECTION_NAME)
            result = collection.query.bm25(
                query=query,
                query_properties=props,
                limit=top_k,
                filters=filters,
                return_metadata=MetadataQuery(score=True),
                tenant=tenant_id,
            )
            return result.objects

        objects = await asyncio.get_event_loop().run_in_executor(None, _search)
        return [
            RetrievedChunk(
                chunk=_prop_to_chunk(o.properties),
                score=o.metadata.score or 0.0,
                retriever_type=RetrieverType.BM25,
                query_variant="original",
                rank=i,
            )
            for i, o in enumerate(objects)
        ]


    async def search_hybrid(
        self,
        query: str,
        query_vector: list[float],
        top_k: int,
        tenant_id: str,
        alpha: float = 0.5,
        named_vector: str = "semantic_vector",
        filters: Filter | None = None,
    ) -> list[RetrievedChunk]:
        """Weaviate native hybrid: dense + BM25, fused with relative-score fusion."""
        client = self._get_client()

        def _search() -> list[Any]:
            collection = client.collections.get(COLLECTION_NAME)
            result = collection.query.hybrid(
                query=query,
                vector=query_vector,
                alpha=alpha,
                limit=top_k,
                target_vector=named_vector,
                fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE,
                filters=filters,
                return_metadata=MetadataQuery(score=True),
                tenant=tenant_id,
            )
            return result.objects

        objects = await asyncio.get_event_loop().run_in_executor(None, _search)
        return [
            RetrievedChunk(
                chunk=_prop_to_chunk(o.properties),
                score=o.metadata.score or 0.0,
                retriever_type=RetrieverType.HYBRID,
                query_variant="original",
                rank=i,
            )
            for i, o in enumerate(objects)
        ]


    async def fetch_by_chunk_id(
        self, chunk_id: str, tenant_id: str
    ) -> Chunk | None:
        """Fetch a single chunk by its logical chunk_id (for context expansion)."""
        client = self._get_client()

        def _fetch() -> list[Any]:
            collection = client.collections.get(COLLECTION_NAME)
            result = collection.query.fetch_objects(
                filters=Filter.by_property("chunk_id").equal(chunk_id),
                limit=1,
                tenant=tenant_id,
            )
            return result.objects

        objects = await asyncio.get_event_loop().run_in_executor(None, _fetch)
        if not objects:
            return None
        return _prop_to_chunk(objects[0].properties)

    async def fetch_by_chunk_index_range(
        self,
        doc_id: str,
        index_min: int,
        index_max: int,
        tenant_id: str,
    ) -> list[Chunk]:
        """Fetch chunks by positional index range (sentence-window expansion)."""
        client = self._get_client()

        def _fetch() -> list[Any]:
            collection = client.collections.get(COLLECTION_NAME)
            result = collection.query.fetch_objects(
                filters=(
                    Filter.by_property("doc_id").equal(doc_id)
                    & Filter.by_property("chunk_index").greater_or_equal(index_min)
                    & Filter.by_property("chunk_index").less_or_equal(index_max)
                ),
                limit=index_max - index_min + 1,
                tenant=tenant_id,
            )
            return result.objects

        objects = await asyncio.get_event_loop().run_in_executor(None, _fetch)
        chunks = [_prop_to_chunk(o.properties) for o in objects]
        return sorted(chunks, key=lambda c: c.chunk_index)



_client_instance: WeaviateClient | None = None
_client_lock = asyncio.Lock()


async def get_weaviate_client() -> WeaviateClient:
    global _client_instance
    if _client_instance is None:
        async with _client_lock:
            if _client_instance is None:
                _client_instance = WeaviateClient()
                await _client_instance.connect()
    return _client_instance


@asynccontextmanager
async def weaviate_client_ctx() -> AsyncIterator[WeaviateClient]:
    """Context manager for one-off scripts / tests."""
    client = WeaviateClient()
    await client.connect()
    try:
        yield client
    finally:
        await client.close()

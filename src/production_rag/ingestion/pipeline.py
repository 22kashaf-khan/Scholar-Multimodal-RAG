"""Ingestion pipeline: load → chunk → embed → RAPTOR → upsert."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import structlog

from production_rag.core.config import Settings, get_settings
from production_rag.core.llm_client import LLMClient, get_llm_client
from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, Document
from production_rag.ingestion.chunkers.factory import get_chunker
from production_rag.ingestion.embedder import BaseEmbedder, get_default_embedder
from production_rag.ingestion.raptor.clusterer import RAPTORClusterer
from production_rag.ingestion.raptor.tree_builder import RAPTORTreeBuilder
from production_rag.vectorstore.tenant_manager import TenantManager
from production_rag.vectorstore.weaviate_client import WeaviateClient

log = structlog.get_logger(__name__)


@dataclass
class IngestionConfig:
    tenant_id: str
    chunking_strategy: ChunkStrategy = ChunkStrategy.HIERARCHICAL
    enable_raptor: bool = True
    chunk_size: int = 500
    chunk_overlap: int = 100


@dataclass
class IngestionResult:
    doc_ids: list[str] = field(default_factory=list)
    total_chunks: int = 0
    leaf_chunks: int = 0
    summary_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    duration_s: float = 0.0


class IngestionPipeline:
    """Orchestrates the full document ingestion pipeline."""

    def __init__(
        self,
        weaviate: WeaviateClient,
        tenant_manager: TenantManager,
        embedder: BaseEmbedder | None = None,
        llm: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._weaviate = weaviate
        self._tenant_manager = tenant_manager
        self._embedder = embedder or get_default_embedder()
        self._llm: LLMClient | None = llm
        self._settings = settings or get_settings()

    async def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = await get_llm_client()
        return self._llm

    async def ingest(
        self,
        documents: list[Document],
        config: IngestionConfig,
    ) -> IngestionResult:
        """Ingest a list of pre-loaded documents."""
        start = time.perf_counter()
        result = IngestionResult()

        await self._tenant_manager.create(config.tenant_id)

        for doc in documents:
            try:
                chunks = await self._process_document(doc, config)
                await self._weaviate.upsert_chunks(chunks, config.tenant_id)
                result.doc_ids.append(doc.doc_id)
                leaves = sum(1 for c in chunks if c.chunk_level == ChunkLevel.LEAF)
                summaries = len(chunks) - leaves
                result.leaf_chunks += leaves
                result.summary_chunks += summaries
                result.total_chunks += len(chunks)
                log.info(
                    "ingestion.doc_done",
                    doc_id=doc.doc_id,
                    chunks=len(chunks),
                    leaves=leaves,
                    summaries=summaries,
                )
            except Exception as e:
                log.error("ingestion.doc_failed", doc_id=doc.doc_id, error=str(e))
                result.errors.append(f"{doc.doc_id}: {e}")

        result.duration_s = time.perf_counter() - start
        log.info(
            "ingestion.done",
            docs=len(documents),
            total_chunks=result.total_chunks,
            errors=len(result.errors),
            duration_s=round(result.duration_s, 2),
        )
        return result

    async def _process_document(
        self, doc: Document, config: IngestionConfig
    ) -> list[Chunk]:
        chunker = get_chunker(
            config.chunking_strategy,
            embedder=self._embedder,
            chunk_size=config.chunk_size,
            overlap=config.chunk_overlap,
        )
        chunks = chunker.chunk(doc, tenant_id=config.tenant_id)
        log.debug("ingestion.chunked", doc_id=doc.doc_id, n=len(chunks))

        leaf_chunks = [c for c in chunks if c.chunk_level == ChunkLevel.LEAF]
        leaf_texts = [c.chunk_text for c in leaf_chunks]

        # Skip re-embedding if late chunker already computed embeddings
        needs_embed = [i for i, c in enumerate(leaf_chunks) if not c.embedding]
        if needs_embed:
            batch_texts = [leaf_texts[i] for i in needs_embed]
            embeddings = await self._embedder.aembed_texts(batch_texts)
            titles = [leaf_chunks[i].title for i in needs_embed]
            title_embeddings = await self._embedder.aembed_texts(
                [f"Title: {t}" if t else "unknown" for t in titles]
            )
            for j, (i, emb, temb) in enumerate(
                zip(needs_embed, embeddings, title_embeddings, strict=True)
            ):
                leaf_chunks[i].embedding = emb
                leaf_chunks[i].title_embedding = temb

        # For non-leaf levels coming from hierarchical chunker, embed them too
        non_leaf = [c for c in chunks if c.chunk_level != ChunkLevel.LEAF]
        if non_leaf:
            nl_texts = [c.chunk_text for c in non_leaf]
            nl_embs = await self._embedder.aembed_texts(nl_texts)
            for c, emb in zip(non_leaf, nl_embs, strict=True):
                c.embedding = emb


        if config.enable_raptor and self._settings.raptor_enabled:
            llm = await self._get_llm()
            clusterer = RAPTORClusterer(
                n_components=self._settings.raptor_umap_n_components,
                n_neighbors=self._settings.raptor_umap_n_neighbors,
                max_cluster_size=self._settings.raptor_max_cluster_size,
            )
            builder = RAPTORTreeBuilder(
                embedder=self._embedder,
                llm=llm,
                clusterer=clusterer,
            )
            # Pass only leaf chunks to RAPTOR (hierarchical chunks already have parents)
            all_chunks = await builder.build(leaf_chunks)
            id_set = {c.chunk_id for c in all_chunks}
            for c in non_leaf:
                if c.chunk_id not in id_set:
                    all_chunks.append(c)
                    id_set.add(c.chunk_id)
            return all_chunks

        return chunks

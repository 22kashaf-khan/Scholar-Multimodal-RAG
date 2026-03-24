"""RAPTOR tree builder: clusters leaf chunks and builds a 3-level summary hierarchy."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import structlog

from production_rag.core.llm_client import LLMClient
from production_rag.core.types import Chunk, ChunkLevel
from production_rag.ingestion.embedder import BaseEmbedder
from production_rag.ingestion.raptor.clusterer import RAPTORClusterer
from production_rag.ingestion.raptor.summarizer import RAPTORSummarizer

log = structlog.get_logger(__name__)


def _summary_chunk_id(doc_id: str, level: str, cluster_id: int) -> str:
    return hashlib.sha256(
        f"raptor:{doc_id}:{level}:{cluster_id}".encode()
    ).hexdigest()[:24]


class RAPTORTreeBuilder:
    """Build a 3-level RAPTOR tree from a flat list of leaf chunks.

    Args:
        embedder: Used to embed summary nodes for indexing.
        llm: LLM client for summarization.
        clusterer: RAPTOR clusterer instance.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        llm: LLMClient,
        clusterer: RAPTORClusterer | None = None,
    ) -> None:
        self._embedder = embedder
        self._summarizer = RAPTORSummarizer(llm)
        self._clusterer = clusterer or RAPTORClusterer()

    async def build(
        self,
        leaf_chunks: list[Chunk],
    ) -> list[Chunk]:
        """Build the RAPTOR tree and return ALL chunks (leaf + summary nodes).

        All returned chunks have embeddings populated.
        """
        if not leaf_chunks:
            return []

        doc_id = leaf_chunks[0].doc_id
        log.info("raptor.build.start", doc_id=doc_id, n_leaves=len(leaf_chunks))


        section_chunks, leaf_with_parents = await self._build_level(
            chunks=leaf_chunks,
            level_name="section",
            child_level=ChunkLevel.LEAF,
            parent_level=ChunkLevel.SECTION,
            level_hint="section",
        )


        if len(section_chunks) > 1:
            paper_chunks, section_with_parents = await self._build_level(
                chunks=section_chunks,
                level_name="paper",
                child_level=ChunkLevel.SECTION,
                parent_level=ChunkLevel.PAPER,
                level_hint="paper",
            )
        else:
            # Only 1 section → promote directly as paper node
            paper_chunks = [
                replace(
                    section_chunks[0],
                    chunk_level=ChunkLevel.PAPER,
                    chunk_id=_summary_chunk_id(doc_id, "paper", 0),
                    parent_chunk_id=None,
                )
            ] if section_chunks else []
            section_with_parents = section_chunks

        all_chunks = leaf_with_parents + section_with_parents + paper_chunks
        log.info(
            "raptor.build.done",
            doc_id=doc_id,
            leaves=len(leaf_with_parents),
            sections=len(section_with_parents),
            papers=len(paper_chunks),
            total=len(all_chunks),
        )
        return all_chunks

    async def _build_level(
        self,
        chunks: list[Chunk],
        level_name: str,
        child_level: ChunkLevel,
        parent_level: ChunkLevel,
        level_hint: str,
    ) -> tuple[list[Chunk], list[Chunk]]:
        """Cluster chunks → summarize → embed summaries.

        Returns:
            (parent_nodes, updated_children_with_parent_ids)
        """
        doc_id = chunks[0].doc_id

        # Get child embeddings (re-use stored embedding if available)
        child_embs = [c.embedding for c in chunks]
        missing = [i for i, e in enumerate(child_embs) if not e]
        if missing:
            texts = [chunks[i].chunk_text for i in missing]
            new_embs = await self._embedder.aembed_texts(texts)
            for i, emb in zip(missing, new_embs, strict=True):
                child_embs[i] = emb
                chunks[i].embedding = emb

        # Cluster
        result = await self._clusterer.cluster(child_embs)

        cluster_texts = {
            cid: [chunks[i].chunk_text for i in idxs]
            for cid, idxs in result.assignments.items()
        }
        summaries = await self._summarizer.summarize_all_clusters(
            cluster_texts, level_hint=level_hint
        )

        parent_chunks: list[Chunk] = []
        parent_id_map: dict[int, str] = {}  # cluster_id → parent chunk_id

        for cid, summary_text in summaries.items():
            pid = _summary_chunk_id(doc_id, level_name, cid)
            parent_id_map[cid] = pid

            # Embed summary
            summary_emb = await self._embedder.aembed_query(summary_text)
            # Use first child's title embedding as proxy
            title_emb = await self._embedder.aembed_query(
                f"Title: {chunks[0].title}" if chunks[0].title else summary_text
            )

            ref_chunk = chunks[result.assignments[cid][0]]
            parent_chunks.append(
                Chunk(
                    chunk_text=summary_text,
                    chunk_id=pid,
                    doc_id=doc_id,
                    chunk_level=parent_level,
                    parent_chunk_id=None,  # set in next level
                    arxiv_id=ref_chunk.arxiv_id,
                    paper_doi=ref_chunk.paper_doi,
                    publication_year=ref_chunk.publication_year,
                    title=ref_chunk.title,
                    authors=ref_chunk.authors,
                    tenant_id=ref_chunk.tenant_id,
                    embedding=summary_emb,
                    title_embedding=title_emb,
                )
            )

        updated_children: list[Chunk] = []
        for cid, idxs in result.assignments.items():
            parent_id = parent_id_map.get(cid)
            for i in idxs:
                updated = replace(chunks[i], parent_chunk_id=parent_id)
                updated_children.append(updated)

        # Deduplicate children (soft-assigned chunks may appear twice)
        seen: set[str] = set()
        deduped: list[Chunk] = []
        for c in updated_children:
            if c.chunk_id not in seen:
                seen.add(c.chunk_id)
                deduped.append(c)

        return parent_chunks, deduped

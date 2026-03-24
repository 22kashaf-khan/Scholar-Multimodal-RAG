"""Abstract base for all chunking strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod

from production_rag.core.types import Chunk, ChunkStrategy, Document


class BaseChunker(ABC):
    strategy: ChunkStrategy

    @abstractmethod
    def chunk(self, document: Document, tenant_id: str = "default") -> list[Chunk]:
        """Split document into Chunks.

        Must populate:
        - chunk_text, chunk_id (deterministic), doc_id
        - parent_chunk_id (if hierarchical), chunk_level
        - chunk_index, section_index
        - start_char, end_char, page
        - All paper metadata fields copied from document
        """
        ...

    def _base_fields(self, document: Document) -> dict:  # type: ignore[type-arg]
        return {
            "doc_id": document.doc_id,
            "arxiv_id": document.arxiv_id,
            "paper_doi": document.paper_doi,
            "publication_year": document.publication_year,
            "title": document.title,
            "authors": document.authors,
        }

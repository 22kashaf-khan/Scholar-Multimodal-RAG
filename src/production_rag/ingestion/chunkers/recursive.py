"""Recursive character/token splitter chunker.

Uses LangChain's RecursiveCharacterTextSplitter which respects
paragraph → sentence → word boundaries before falling back to characters.
Strong production baseline for scientific prose.
"""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, Document
from production_rag.ingestion.chunkers.base import BaseChunker


class RecursiveChunker(BaseChunker):
    strategy = ChunkStrategy.RECURSIVE

    def __init__(self, chunk_size: int = 600, overlap: int = 100) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

    def chunk(self, document: Document, tenant_id: str = "default") -> list[Chunk]:
        base = self._base_fields(document)
        raw_chunks = self._splitter.split_text(document.text)
        chunks: list[Chunk] = []
        cursor = 0

        for idx, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            # Find character position in original text for provenance
            start = document.text.find(text, cursor)
            if start == -1:
                start = cursor
            end = start + len(text)
            cursor = start + 1

            chunk_id = hashlib.sha256(
                f"recursive:{document.doc_id}:{idx}".encode()
            ).hexdigest()[:24]

            chunks.append(
                Chunk(
                    chunk_text=text,
                    chunk_id=chunk_id,
                    chunk_level=ChunkLevel.LEAF,
                    chunk_index=idx,
                    start_char=start,
                    end_char=end,
                    tenant_id=tenant_id,
                    **base,  # type: ignore[arg-type]
                )
            )

        return chunks

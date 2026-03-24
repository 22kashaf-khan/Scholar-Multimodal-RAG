"""Fixed-size chunker.

Simple baseline: split on character count with optional overlap.
No semantic boundaries — useful as ablation baseline only.
"""

from __future__ import annotations

import hashlib

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, Document
from production_rag.ingestion.chunkers.base import BaseChunker


class FixedChunker(BaseChunker):
    strategy = ChunkStrategy.FIXED

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        self._size = chunk_size
        self._overlap = overlap

    def chunk(self, document: Document, tenant_id: str = "default") -> list[Chunk]:
        text = document.text
        base = self._base_fields(document)
        chunks: list[Chunk] = []
        step = self._size - self._overlap
        i = 0
        idx = 0

        while i < len(text):
            end = min(i + self._size, len(text))
            chunk_text = text[i:end].strip()
            if chunk_text:
                chunk_id = hashlib.sha256(
                    f"fixed:{document.doc_id}:{i}".encode()
                ).hexdigest()[:24]
                chunks.append(
                    Chunk(
                        chunk_text=chunk_text,
                        chunk_id=chunk_id,
                        chunk_level=ChunkLevel.LEAF,
                        chunk_index=idx,
                        start_char=i,
                        end_char=end,
                        tenant_id=tenant_id,
                        **base,  # type: ignore[arg-type]
                    )
                )
                idx += 1
            i += step

        return chunks

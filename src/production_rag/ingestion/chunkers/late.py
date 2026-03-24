"""Late chunking strategy (Jina-style).

Encodes the full document first, then pools token embeddings per chunk span,
so each chunk vector retains global document context.
Falls back to mean-pooled embeddings if a long-context model is unavailable.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import structlog

from production_rag.core.types import Chunk, ChunkLevel, ChunkStrategy, Document
from production_rag.ingestion.chunkers.base import BaseChunker
from production_rag.ingestion.chunkers.recursive import RecursiveChunker

log = structlog.get_logger(__name__)

_FALLBACK_CHUNK_SIZE = 600
_FALLBACK_OVERLAP = 100


class LateChunker(BaseChunker):
    """Long-context late-chunking strategy.

    Args:
        model_name_or_path: HuggingFace model path/name.  Must support
            `encode` with `output_value="token_embeddings"`.
        chunk_size: Approximate character size of each chunk span.
        max_seq_len: Maximum token sequence the model supports.
    """

    strategy = ChunkStrategy.LATE

    def __init__(
        self,
        model_name_or_path: str = "jinaai/jina-embeddings-v3",
        chunk_size: int = 600,
        max_seq_len: int = 8192,
    ) -> None:
        self._model_name = model_name_or_path
        self._chunk_size = chunk_size
        self._max_seq_len = max_seq_len
        self._model: Any = None
        self._tokenizer: Any = None
        self._fallback = RecursiveChunker(chunk_size, 100)

    def _load_model(self) -> bool:
        """Lazy load sentence-transformers model. Returns True on success."""
        if self._model is not None:
            return True
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._model_name, trust_remote_code=True
            )
            log.info("late_chunker.model_loaded", model=self._model_name)
            return True
        except Exception as e:
            log.warning(
                "late_chunker.model_load_failed",
                model=self._model_name,
                error=str(e),
                fallback="recursive",
            )
            return False

    def chunk(self, document: Document, tenant_id: str = "default") -> list[Chunk]:
        if not self._load_model():
            # Graceful degradation: use recursive chunking
            return self._fallback.chunk(document, tenant_id)

        # 1. Get boundary positions via recursive splitter (char positions)
        boundary_chunks = self._fallback.chunk(document, tenant_id)
        if not boundary_chunks:
            return []

        # 2. Encode full document → token embeddings
        try:
            token_embs = self._model.encode(  # type: ignore[union-attr]
                document.text[: self._max_seq_len * 4],  # rough char limit
                output_value="token_embeddings",
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            # token_embs shape: (seq_len, hidden_dim)
        except Exception as e:
            log.warning("late_chunker.encode_failed", error=str(e))
            return self._fallback.chunk(document, tenant_id)

        # 3. Tokenize to get char→token mapping
        try:
            tokenizer = self._model.tokenizer  # type: ignore[union-attr]
            encoding = tokenizer(
                document.text[: self._max_seq_len * 4],
                return_offsets_mapping=True,
                truncation=True,
                max_length=self._max_seq_len,
            )
            offset_mapping: list[tuple[int, int]] = encoding["offset_mapping"]
        except Exception as e:
            log.warning("late_chunker.tokenize_failed", error=str(e))
            return self._fallback.chunk(document, tenant_id)

        char_to_token = _build_char_to_token_map(offset_mapping)

        # 4. Pool token embeddings per chunk span
        final_chunks: list[Chunk] = []
        for bc in boundary_chunks:
            tok_start = char_to_token.get(bc.start_char, 0)
            tok_end = char_to_token.get(min(bc.end_char - 1, len(offset_mapping) - 1), tok_start + 1)
            span_embs = token_embs[tok_start : tok_end + 1]

            if len(span_embs) == 0:
                pooled = np.zeros(token_embs.shape[-1])
            else:
                pooled = span_embs.mean(axis=0)

            chunk_id = hashlib.sha256(
                f"late:{document.doc_id}:{bc.chunk_index}".encode()
            ).hexdigest()[:24]

            final_chunks.append(
                Chunk(
                    chunk_text=bc.chunk_text,
                    chunk_id=chunk_id,
                    doc_id=bc.doc_id,
                    chunk_level=ChunkLevel.LEAF,
                    chunk_index=bc.chunk_index,
                    start_char=bc.start_char,
                    end_char=bc.end_char,
                    arxiv_id=bc.arxiv_id,
                    paper_doi=bc.paper_doi,
                    publication_year=bc.publication_year,
                    title=bc.title,
                    authors=bc.authors,
                    tenant_id=tenant_id,
                    # Late-chunking provides pre-computed semantic embeddings
                    embedding=pooled.tolist(),
                )
            )

        log.debug(
            "late_chunker.done",
            doc_id=document.doc_id,
            chunks=len(final_chunks),
        )
        return final_chunks


def _build_char_to_token_map(
    offset_mapping: list[tuple[int, int]],
) -> dict[int, int]:
    """Map character positions to token indices."""
    mapping: dict[int, int] = {}
    for tok_idx, (char_start, char_end) in enumerate(offset_mapping):
        for c in range(char_start, char_end):
            mapping[c] = tok_idx
    return mapping

"""PDF loader using PyMuPDF (fitz).

Extracts text page-by-page, normalises whitespace, and returns a single
Document per PDF file with per-page metadata available in doc.metadata.
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import structlog

from production_rag.core.types import Document
from production_rag.ingestion.loaders.base import BaseLoader

log = structlog.get_logger(__name__)


class PDFLoader(BaseLoader):
    """Load a PDF file from a local path using PyMuPDF.

    Returns one Document whose text is the full concatenated content,
    plus page boundaries stored in metadata for downstream use.
    """

    def __init__(self, min_page_chars: int = 50) -> None:
        self._min_page_chars = min_page_chars

    async def load(self, source: str) -> list[Document]:
        """Load PDF from file path. Source is an absolute or relative path."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load_sync, source)

    def _load_sync(self, path: str) -> list[Document]:
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError("PyMuPDF required: pip install pymupdf") from exc

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        doc = fitz.open(path)
        pages: list[dict[str, Any]] = []
        full_text_parts: list[str] = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")  # type: ignore[call-overload]
            text = self._clean(text)
            if len(text) < self._min_page_chars:
                continue  # skip near-empty pages
            pages.append({"page": page_num + 1, "text": text, "start_char": sum(len(t) for t in full_text_parts)})
            full_text_parts.append(text)

        doc.close()
        full_text = "\n\n".join(full_text_parts)
        doc_id = f"pdf:{hashlib.sha256(full_text[:500].encode()).hexdigest()[:12]}"

        log.info("pdf.loaded", path=path, pages=len(pages), chars=len(full_text))
        return [
            Document(
                text=full_text,
                doc_id=doc_id,
                title=p.stem,
                source_uri=str(p.resolve()),
                metadata={"pages": pages},
            )
        ]

    @staticmethod
    def _clean(text: str) -> str:
        import re
        # Normalise whitespace; remove form-feeds
        text = re.sub(r"\f", "\n", text)
        text = re.sub(r" +", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

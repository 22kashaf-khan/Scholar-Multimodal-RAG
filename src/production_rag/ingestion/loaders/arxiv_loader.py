"""ArXiv paper loader via the arxiv-python library."""

from __future__ import annotations

import asyncio
import hashlib
import tempfile
from pathlib import Path

import arxiv
import structlog

from production_rag.core.types import Document
from production_rag.ingestion.loaders.base import BaseLoader
from production_rag.ingestion.loaders.pdf import PDFLoader

log = structlog.get_logger(__name__)


class ArXivLoader(BaseLoader):
    """Load papers from ArXiv by ID or search query.

    Args:
        fetch_pdf: Whether to download the full PDF and extract text.
                   If False, only the abstract is used.
        pdf_loader: PDFLoader instance; one is created if not provided.
    """

    def __init__(
        self,
        fetch_pdf: bool = True,
        pdf_loader: PDFLoader | None = None,
    ) -> None:
        self._fetch_pdf = fetch_pdf
        self._pdf_loader = pdf_loader or PDFLoader()
        self._client = arxiv.Client(
            page_size=10,
            delay_seconds=3.0,
            num_retries=3,
        )

    async def load(self, source: str) -> list[Document]:
        """Load a paper by ArXiv ID (e.g. '2005.11401') or comma-separated IDs."""
        ids = [s.strip() for s in source.split(",") if s.strip()]
        tasks = [self._load_single(aid) for aid in ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        docs: list[Document] = []
        for aid, result in zip(ids, results, strict=True):
            if isinstance(result, Exception):
                log.error("arxiv.load_failed", arxiv_id=aid, error=str(result))
            else:
                docs.extend(result)
        return docs

    async def _load_single(self, arxiv_id: str) -> list[Document]:
        loop = asyncio.get_event_loop()

        def _fetch() -> list[arxiv.Result]:
            search = arxiv.Search(id_list=[arxiv_id])
            return list(self._client.results(search))

        results = await loop.run_in_executor(None, _fetch)
        if not results:
            raise ValueError(f"ArXiv paper not found: {arxiv_id}")

        paper = results[0]
        doc_id = f"arxiv:{arxiv_id}"
        authors = [str(a) for a in paper.authors]

        if self._fetch_pdf:
            return await self._load_pdf(paper, doc_id, authors, arxiv_id)

        return [
            Document(
                text=paper.summary,
                doc_id=doc_id,
                title=paper.title,
                source_uri=paper.entry_id,
                arxiv_id=arxiv_id,
                publication_year=paper.published.year if paper.published else 0,
                authors=authors,
            )
        ]

    async def _load_pdf(
        self,
        paper: arxiv.Result,
        doc_id: str,
        authors: list[str],
        arxiv_id: str,
    ) -> list[Document]:
        loop = asyncio.get_event_loop()

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / f"{arxiv_id.replace('/', '_')}.pdf"

            def _download() -> None:
                paper.download_pdf(dirpath=tmpdir, filename=pdf_path.name)

            await loop.run_in_executor(None, _download)
            log.info("arxiv.pdf_downloaded", arxiv_id=arxiv_id, path=str(pdf_path))

            docs = await self._pdf_loader.load(str(pdf_path))
            # Enrich with ArXiv metadata
            for doc in docs:
                doc.doc_id = doc_id
                doc.title = paper.title
                doc.arxiv_id = arxiv_id
                doc.source_uri = paper.entry_id
                doc.authors = authors
                doc.publication_year = (
                    paper.published.year if paper.published else 0
                )
            return docs

"""PDF loader using Docling Serve for structured extraction.

Calls the running docling-serve container's /v1/chunk/hybrid/file endpoint,
which returns typed chunks (text, table, figure) with table content formatted
as Markdown.  Falls back to PyMuPDF PDFLoader if the service is unavailable.
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from pathlib import Path

import structlog

from production_rag.core.types import Document
from production_rag.ingestion.loaders.base import BaseLoader

log = structlog.get_logger(__name__)

_DOCLING_URL = os.getenv("DOCLING_URL", "http://docling-serve:5001")


class DoclingPDFLoader(BaseLoader):
    """Load a PDF via Docling Serve, preserving table and figure structure.

    The loaded Document has ``metadata["docling_chunks"]`` populated with the
    raw list of chunk dicts from the API.  DoclingChunker reads that list.

    Falls back to PDFLoader (PyMuPDF) if docling-serve is unreachable.
    """

    def __init__(self, docling_url: str | None = None, timeout: int = 300) -> None:
        self._url = (docling_url or _DOCLING_URL).rstrip("/")
        self._timeout = timeout

    async def load(self, source: str) -> list[Document]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._load_sync, source)

    def _load_sync(self, path: str) -> list[Document]:
        try:
            return self._load_with_docling(path)
        except Exception as exc:
            log.warning(
                "docling_pdf_loader.fallback",
                path=path,
                error=str(exc),
            )
            from production_rag.ingestion.loaders.pdf import PDFLoader
            return PDFLoader()._load_sync(path)  # type: ignore[attr-defined]

    def _load_with_docling(self, path: str) -> list[Document]:
        try:
            import httpx
            import time
        except ImportError as exc:
            raise ImportError("httpx required: pip install httpx") from exc

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        with httpx.Client(timeout=30) as client:
            # Submit async job
            with open(path, "rb") as f:
                resp = client.post(
                    f"{self._url}/v1/chunk/hybrid/file/async",
                    files={"files": (p.name, f, "application/pdf")},
                )
            resp.raise_for_status()
            task_id = resp.json().get("task_id")
            if not task_id:
                raise ValueError(f"No task_id in response: {resp.text[:200]}")

            log.info("docling_pdf_loader.submitted", task_id=task_id, path=path)

            # Poll until done
            deadline = time.time() + self._timeout
            poll_interval = 5
            while time.time() < deadline:
                time.sleep(poll_interval)
                poll_resp = client.get(f"{self._url}/v1/status/poll/{task_id}", timeout=10)
                poll_resp.raise_for_status()
                status_data = poll_resp.json()
                task_status = status_data.get("task_status", "")
                log.debug("docling_pdf_loader.poll", task_id=task_id, status=task_status)

                if task_status == "success":
                    result_resp = client.get(f"{self._url}/v1/result/{task_id}", timeout=30)
                    result_resp.raise_for_status()
                    data = result_resp.json()
                    break
                elif task_status in ("failure", "revoked"):
                    raise RuntimeError(f"Docling task {task_id} failed: {task_status}")
                # pending / started — keep polling
            else:
                raise TimeoutError(f"Docling task {task_id} timed out after {self._timeout}s")

        raw_chunks: list[dict] = data.get("chunks", [])
        if not raw_chunks:
            raise ValueError("docling-serve returned no chunks")

        # Derive a stable doc_id from the first chunk texts
        seed = " ".join(c.get("text", "") for c in raw_chunks[:5])
        doc_id = f"pdf:{hashlib.sha256(seed[:500].encode()).hexdigest()[:12]}"

        table_count = sum(
            1 for c in raw_chunks
            if any("#/tables/" in ref for ref in c.get("doc_items", []))
        )
        log.info(
            "docling_pdf_loader.done",
            path=path,
            total_chunks=len(raw_chunks),
            table_chunks=table_count,
        )

        return [
            Document(
                text="",  # chunker reads docling_chunks; text unused
                doc_id=doc_id,
                title=p.stem,
                source_uri=str(p.resolve()),
                metadata={"docling_chunks": raw_chunks},
            )
        ]

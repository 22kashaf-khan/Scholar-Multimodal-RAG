"""
End-to-end test: ingest ESG_Performance_Data_2024.pdf, then ask two questions.

Requires THREE terminals to be running first (see README or instructions below).

Usage:
    python run_e2e_test.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx

API = "http://localhost:8000"
TENANT = "esg_demo"
PDF_PATH = str(Path("ESG_Performance_Data_2024.pdf").resolve())

QUESTIONS = [
    "What are the key ESG performance highlights for 2024?",
    "What are the carbon emission targets and progress mentioned in the report?",
]


async def wait_for_api(timeout: int = 40) -> bool:
    for _ in range(timeout):
        try:
            async with httpx.AsyncClient() as client:
                r = await client.get(f"{API}/health", timeout=3)
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(1)
    return False


async def ingest_pdf() -> str:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(
            f"{API}/ingest",
            json={
                "pdf_paths": [PDF_PATH],
                "tenant_id": TENANT,
                "chunking_strategy": "recursive",
                "enable_raptor": False,   # faster for demo
            },
        )
        r.raise_for_status()
        data = r.json()
        job_id = data.get("job_id", "")
        print(f"[INGEST] Job queued: {job_id}")
        return job_id


async def poll_job(job_id: str, timeout: int = 180) -> bool:
    print(f"[INGEST] Waiting for worker to process job", end="", flush=True)
    for _ in range(timeout // 2):
        await asyncio.sleep(2)
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.get(f"{API}/ingest/{job_id}")
            if r.status_code == 200:
                status = r.json().get("status", "")
                print(".", end="", flush=True)
                if status in ("complete", "success"):
                    print(" done!")
                    return True
                if status == "failed":
                    print(f"\n[ERROR] Job failed: {r.json()}")
                    return False
        except Exception:
            pass
    print("\n[WARN] Job polling timed out — it may still be running in the worker.")
    return True   # proceed to queries anyway


async def ask(question: str) -> None:
    print(f"\n{'─'*60}")
    print(f"Q: {question}")
    print(f"{'─'*60}")
    print("A: ", end="", flush=True)

    citations: list[dict] = []
    diag: dict = {}

    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST",
            f"{API}/chat",
            json={
                "query": question,
                "tenant_id": TENANT,
                "enable_crag": True,
                "enable_self_rag": False,
            },
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if not raw:
                    continue
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                if etype == "token":
                    tok = event.get("data", {}).get("token", "")
                    print(tok, end="", flush=True)
                elif etype == "citation":
                    citations.append(event.get("data", {}))
                elif etype == "diagnostics":
                    diag = event.get("data", {})
                elif etype == "error":
                    print(f"\n[ERROR] {event.get('data', {}).get('message', '')}")

    print("\n")
    if citations:
        print(f"  Sources cited ({len(citations)}):")
        for c in citations[:4]:
            print(f"    [{c.get('citation_id','')}] {c.get('title','N/A')}  (page {c.get('page','?')})")
    if diag:
        print(f"  Pipeline stats: candidates={diag.get('candidate_count',0)} "
              f"| post-rerank={diag.get('post_rerank_count',0)} "
              f"| tokens={diag.get('total_tokens',0)}")


async def main() -> None:
    print("=" * 60)
    print("  Production RAG — End-to-End Test")
    print("=" * 60)

    print("\n[1/3] Checking API at http://localhost:8000 ...")
    if not await wait_for_api():
        print("\n[ERROR] API not responding. Start it first:")
        print("        uvicorn production_rag.api.main:app --reload --port 8000")
        sys.exit(1)
    print("       ✓ API is up")

    print(f"\n[2/3] Ingesting PDF: {PDF_PATH}")
    print(f"      Tenant: {TENANT}")
    try:
        job_id = await ingest_pdf()
        await poll_job(job_id)
    except httpx.HTTPStatusError as e:
        print(f"\n[ERROR] Ingest API error {e.response.status_code}: {e.response.text}")
        sys.exit(1)

    print("\n[3/3] Running RAG queries ...")
    for q in QUESTIONS:
        await ask(q)

    print("\n" + "=" * 60)
    print("  ✓ Test complete!")
    print("  Open http://localhost:8501 for the full Streamlit UI")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())

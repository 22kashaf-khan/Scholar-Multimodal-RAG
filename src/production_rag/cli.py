"""Typer CLI for the production RAG pipeline.

Commands:
  schema create   — create Weaviate collection schema
  schema drop     — drop collection (destructive!)
  ingest arxiv    — ingest papers by ArXiv IDs
  ingest pdf      — ingest local PDF files
  tenant create   — create a new tenant
  tenant list     — list all tenants
  eval ragas      — run RAGAS evaluation suite
  eval chunking   — run chunking ablation study
  eval retrieval  — run retrieval benchmark
  serve           — start FastAPI server (dev mode)

Usage:
    python -m production_rag.cli schema create
    python -m production_rag.cli ingest arxiv 2312.10997 2310.01558
    python -m production_rag.cli eval ragas --dataset qasper --n 200
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="production-rag",
    help="Expert-level production RAG pipeline CLI",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

schema_app = typer.Typer(help="Weaviate schema management", no_args_is_help=True)
ingest_app = typer.Typer(help="Document ingestion", no_args_is_help=True)
tenant_app = typer.Typer(help="Tenant management", no_args_is_help=True)
eval_app = typer.Typer(help="Evaluation pipelines", no_args_is_help=True)

app.add_typer(schema_app, name="schema")
app.add_typer(ingest_app, name="ingest")
app.add_typer(tenant_app, name="tenant")
app.add_typer(eval_app, name="eval")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(coro):
    """Run an async coroutine from sync CLI context."""
    return asyncio.run(coro)


def _bootstrap():
    """Lazily import and initialise shared resources."""
    from production_rag.core.config import get_settings
    from production_rag.core.llm_client import LLMClient
    from production_rag.core.logging import setup_logging
    from production_rag.ingestion.embedder import get_embedder
    from production_rag.vectorstore.weaviate_client import WeaviateClient

    setup_logging(json_logs=False)
    settings = get_settings()
    # Construct objects synchronously; WeaviateClient connects via run_in_executor
    # so the underlying TCP socket is not bound to any asyncio event loop.
    weaviate = WeaviateClient(settings)
    asyncio.run(weaviate.connect())
    embedder = get_embedder(settings)
    llm = LLMClient(settings)
    return settings, weaviate, embedder, llm


# ── Schema commands ───────────────────────────────────────────────────────────

@schema_app.command("create")
def schema_create(
    embedding_dim: int = typer.Option(1024, help="Embedding dimension"),
) -> None:
    """Create the Weaviate ScientificChunk collection schema."""
    settings, weaviate, _, _ = _bootstrap()
    _run(weaviate.create_schema(embedding_dim))
    typer.echo(typer.style("✔ Schema created", fg=typer.colors.GREEN))


@schema_app.command("drop")
def schema_drop(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
) -> None:
    """[bold red]DESTRUCTIVE[/bold red] — drop the entire collection."""
    if not yes:
        confirmed = typer.confirm("This will delete ALL data. Continue?")
        if not confirmed:
            raise typer.Abort()
    _, weaviate, _, _ = _bootstrap()
    _run(weaviate.drop_schema())
    typer.echo(typer.style("✔ Schema dropped", fg=typer.colors.YELLOW))


# ── Ingest commands ───────────────────────────────────────────────────────────

@ingest_app.command("arxiv")
def ingest_arxiv(
    arxiv_ids: list[str] = typer.Argument(..., help="ArXiv paper IDs, e.g. 2312.10997"),
    tenant_id: str = typer.Option("default", help="Target tenant"),
    chunk_strategy: str = typer.Option("recursive", help="Chunking strategy"),
    raptor: bool = typer.Option(True, help="Build RAPTOR tree"),
) -> None:
    """Ingest ArXiv papers by their IDs."""
    from production_rag.ingestion.loaders.arxiv_loader import ArXivLoader
    from production_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
    from production_rag.ingestion.chunkers.factory import ChunkStrategy

    settings, weaviate, embedder, _ = _bootstrap()

    async def _ingest():
        loader = ArXivLoader()
        docs = await loader.load(arxiv_ids)
        typer.echo(f"Loaded {len(docs)} document(s)")
        pipeline = IngestionPipeline(weaviate, embedder, settings)
        cfg = IngestionConfig(
            tenant_id=tenant_id,
            chunk_strategy=ChunkStrategy(chunk_strategy),
            raptor_enabled=raptor,
        )
        stats = await pipeline.ingest(docs, cfg)
        typer.echo(typer.style(f"✔ Ingested {stats['chunks_upserted']} chunks", fg=typer.colors.GREEN))

    _run(_ingest())


@ingest_app.command("pdf")
def ingest_pdf(
    paths: list[Path] = typer.Argument(..., help="PDF file paths"),
    tenant_id: str = typer.Option("default", help="Target tenant"),
    chunk_strategy: str = typer.Option("recursive", help="Chunking strategy"),
    raptor: bool = typer.Option(True, help="Build RAPTOR tree"),
) -> None:
    """Ingest local PDF files."""
    from production_rag.ingestion.loaders.pdf import PDFLoader
    from production_rag.ingestion.pipeline import IngestionConfig, IngestionPipeline
    from production_rag.ingestion.chunkers.factory import ChunkStrategy

    settings, weaviate, embedder, _ = _bootstrap()

    async def _ingest():
        loader = PDFLoader()
        docs = []
        for p in paths:
            docs.extend(await loader.load(str(p)))
        typer.echo(f"Loaded {len(docs)} document(s)")
        pipeline = IngestionPipeline(weaviate, embedder, settings)
        cfg = IngestionConfig(
            tenant_id=tenant_id,
            chunk_strategy=ChunkStrategy(chunk_strategy),
            raptor_enabled=raptor,
        )
        stats = await pipeline.ingest(docs, cfg)
        typer.echo(typer.style(f"✔ Ingested {stats['chunks_upserted']} chunks", fg=typer.colors.GREEN))

    _run(_ingest())


# ── Tenant commands ───────────────────────────────────────────────────────────

@tenant_app.command("create")
def tenant_create(
    tenant_id: str = typer.Argument(..., help="Tenant identifier"),
) -> None:
    """Create a new Weaviate tenant."""
    _, weaviate, _, _ = _bootstrap()
    from production_rag.vectorstore.tenant_manager import TenantManager
    mgr = TenantManager(weaviate)
    _run(mgr.create(tenant_id))
    typer.echo(typer.style(f"✔ Tenant '{tenant_id}' created", fg=typer.colors.GREEN))


@tenant_app.command("list")
def tenant_list() -> None:
    """List all Weaviate tenants."""
    _, weaviate, _, _ = _bootstrap()
    from production_rag.vectorstore.tenant_manager import TenantManager
    mgr = TenantManager(weaviate)
    tenants = _run(mgr.list_tenants())
    if not tenants:
        typer.echo("No tenants found")
        return
    for t in tenants:
        typer.echo(f"  {t['name']:30s}  {t['activity_status']}")


# ── Eval commands ─────────────────────────────────────────────────────────────

@eval_app.command("ragas")
def eval_ragas(
    dataset: str = typer.Option("qasper", help="qasper or sciq"),
    n: int = typer.Option(200, help="Number of samples"),
    fail_fast: bool = typer.Option(False, help="Exit 1 on CI gate failure"),
) -> None:
    """Run the RAGAS evaluation suite."""
    from production_rag.evaluation.ragas_suite import (
        build_ragas_dataset,
        check_pass_gate,
        load_qasper,
        load_sciq,
        run_ragas_metrics,
        save_results,
    )
    from production_rag.chains.rag_chain import RAGChain

    settings, weaviate, embedder, llm = _bootstrap()
    chain = RAGChain(weaviate, embedder, llm, settings)

    loader = load_qasper if dataset == "qasper" else load_sciq
    samples = loader(n)
    typer.echo(f"Loaded {len(samples)} samples")

    ragas_samples = _run(build_ragas_dataset(chain, samples))
    scores = run_ragas_metrics(ragas_samples)
    for k, v in scores.items():
        typer.echo(f"  {k}: {v:.4f}")

    passed = check_pass_gate(scores)
    save_results(scores, dataset, len(samples))

    if fail_fast and not passed:
        sys.exit(1)


@eval_app.command("chunking")
def eval_chunking(
    arxiv_ids: list[str] = typer.Argument(default=["2312.10997"]),
    output: Path = typer.Option(Path("eval_results/chunking_ablation.json")),
) -> None:
    """Run chunking strategy ablation study."""
    import subprocess
    args = ["python", "-m", "production_rag.evaluation.chunking_ablation",
            "--output", str(output)] + list(arxiv_ids)
    subprocess.run(args, check=True)


@eval_app.command("retrieval")
def eval_retrieval(
    tenant_id: str = typer.Option("eval"),
    output: Path = typer.Option(Path("eval_results/retrieval_benchmark.json")),
) -> None:
    """Run retrieval configuration benchmark."""
    import subprocess
    subprocess.run([
        "python", "-m", "production_rag.evaluation.retrieval_benchmark",
        "--tenant-id", tenant_id,
        "--output", str(output),
    ], check=True)


# ── Serve command ─────────────────────────────────────────────────────────────

@app.command("serve")
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, help="Auto-reload (dev only)"),
    workers: int = typer.Option(1, help="Uvicorn worker count"),
) -> None:
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        typer.echo("pip install uvicorn[standard]", err=True)
        raise typer.Exit(1)

    uvicorn.run(
        "production_rag.api.main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers,
        log_level="info",
    )


if __name__ == "__main__":
    app()

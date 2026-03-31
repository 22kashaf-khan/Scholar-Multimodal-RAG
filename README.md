# ScholarRAG — Production-Grade RAG for Scientific Papers

> A fully containerised Retrieval-Augmented Generation system built for scientific and technical documents — with support for structured table extraction, adaptive retrieval, streaming answers, and inline citations.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Weaviate](https://img.shields.io/badge/Vector%20Store-Weaviate-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## What is ScholarRAG?

ScholarRAG is an end-to-end RAG pipeline that goes far beyond a simple "chunk-and-embed" approach. It is designed for **domain experts who need precise, well-cited answers from dense scientific literature** — research papers, technical reports, and data-heavy PDFs.

It handles the hard parts of production RAG: table-aware document parsing, multi-query retrieval fusion, reranking, adaptive self-correction, streaming responses with citations, and a clean Streamlit UI — all running locally with Docker.

The pipeline supports **multimodal LLMs** — models that can reason over both text and visual content (tables, figures, charts) extracted from documents, enabling richer answers from visually complex scientific papers.

---

## Key Features

### 🔍 Retrieval
- **Hybrid search** — dense (BGE embeddings) + sparse (BM25) via Weaviate, fused with Reciprocal Rank Fusion (RRF)
- **Multi-query expansion** — generates 3 paraphrase variants per query to improve recall
- **HyDE** — Hypothetical Document Embeddings for better semantic alignment
- **MMR (Maximal Marginal Relevance)** — reduces redundancy in retrieved chunks
- **BGE Cross-Encoder reranker** — re-scores candidates for precision (self-hosted, no API key needed)
- **Context window expansion** — fetches neighbouring chunks around top hits

### 📄 Document Ingestion
| Strategy | Description |
|---|---|
| `recursive` | LangChain recursive character splitting (default) |
| `fixed` | Fixed-size chunks with overlap |
| `semantic` | Embedding-based semantic boundary detection |
| `hierarchical` | Parent-child chunk tree for context-aware retrieval |
| `late` | Late chunking — embed full document, slice embeddings |
| `docling` | **Structured extraction via Docling** — preserves tables as Markdown, identifies figures |

- **ArXiv loader** — ingest papers directly by ArXiv ID
- **PDF loader** — upload and ingest any PDF
- **RAPTOR** — recursive abstractive summarisation tree for long documents (optional)
- **Async ingestion** — jobs queued in Redis via Arq, processed in a background worker

### 🧠 Adaptive RAG
- **CRAG (Corrective RAG)** — detects low-quality retrieval and re-retrieves with a refined query
- **Self-RAG** — critiques the generated answer, flags unsupported claims, and regenerates if needed

### ✍️ Generation
- **Streaming SSE responses** — token-by-token rendering in the UI via Server-Sent Events
- **Inline citations** — every claim is cited using `[SOURCE N]` notation
- **Table-aware prompting** — table chunks are wrapped with `[TABLE]...[/TABLE]` markers so the LLM extracts exact rows and numbers
- **Provider-agnostic** via [LiteLLM](https://github.com/BerriAI/litellm) — supports Groq, OpenAI, Anthropic, Google Gemini, Ollama
- **Multimodal LLM support** — route to vision-capable models (e.g. `llama-3.2-11b-vision`, `gpt-4o`, `gemini-2.0-flash`) to reason over figures and chart images extracted from PDFs

### 📊 Observability
- Structured JSON logging via `structlog`
- OpenTelemetry tracing (OTLP export)
- Prometheus metrics endpoint
- Grafana dashboard (full profile)
- LangSmith tracing support

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Streamlit UI (:8501)                     │
│          Upload PDF / ArXiv ID │ Chat │ Citations │ Diagnostics │
└─────────────────────┬───────────────────────────────────────────┘
                      │ SSE stream
┌─────────────────────▼───────────────────────────────────────────┐
│                      FastAPI (:8000)                            │
│   POST /chat  ──▶  RAG Chain                                    │
│   POST /ingest ──▶  Arq Job Queue (Redis)                       │
└──────────┬──────────────────────────────┬───────────────────────┘
           │                              │
┌──────────▼──────────┐       ┌───────────▼──────────────────────┐
│     RAG Chain       │       │     Background Worker (Arq)      │
│                     │       │                                  │
│  Query Transforms   │       │  Loader (PDF / ArXiv / Docling)  │
│  ├─ Multi-Query     │       │  Chunker (6 strategies)          │
│  └─ HyDE            │       │  Embedder (BGE bge-large-en-v1.5)│
│                     │       │  RAPTOR (optional)               │
│  Retrieval          │       │  Weaviate Upsert                 │
│  ├─ Dense + Sparse  │       └──────────────────────────────────┘
│  ├─ RRF Fusion      │
│  ├─ MMR             │       ┌──────────────────────────────────┐
│  └─ BGE Reranker    │       │  Weaviate (:8080)                │
│                     │       │  Multi-tenant vector store       │
│  Adaptive           │       │  HNSW + BM25 hybrid index        │
│  ├─ CRAG            │       └──────────────────────────────────┘
│  └─ Self-RAG        │
│                     │       ┌──────────────────────────────────┐
│  Synthesizer        │       │  Docling Serve (:5001)           │
│  └─ Streaming SSE   │       │  Structured PDF/table extraction │
└─────────────────────┘       └──────────────────────────────────┘
```

---

## Quickstart

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (with Compose V2)
- An LLM API key — any **one** of:
  - [Groq](https://console.groq.com) — free tier, fast inference
  - [Google AI Studio](https://aistudio.google.com) — free tier (`GOOGLE_API_KEY`)
  - [OpenAI](https://platform.openai.com) or [Anthropic](https://console.anthropic.com)
  - Or run **fully local** with [Ollama](https://ollama.ai) (no API key needed)

### 1. Clone & configure

```bash
git clone https://github.com/your-username/scholar-rag.git
cd scholar-rag
cp .env.example .env
```

Open `.env` and set your LLM key — for example with Groq (free, no credit card):

```env
GROQ_API_KEY=your_key_here
LLM_DEFAULT_PROVIDER=groq
LLM_DEFAULT_MODEL=groq/llama-3.3-70b-versatile
LLM_FAST_MODEL=groq/llama-3.1-8b-instant
```

### 2. Start everything

```powershell
# Windows (PowerShell)
.\start.ps1
```

```bash
# macOS / Linux
docker compose --profile dev up --build
```

Wait ~60 seconds for all services to become healthy, then open **http://localhost:8501**.

### 3. Ingest a paper

**Option A — ArXiv ID** (sidebar → "Ingest ArXiv Papers"):
```
2312.10997
```

**Option B — Upload PDF** (sidebar → "Upload PDF"):
- Choose a PDF, select chunking strategy, click **Upload & Ingest PDF**
- Use strategy **`docling`** for papers with tables to get structured extraction

### 4. Ask questions

```
What is the main contribution of this paper?
Explain Table 1 in detail.
What baselines were compared and what were the results?
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Vector store | [Weaviate 1.27](https://weaviate.io) — hybrid HNSW + BM25, multi-tenant |
| Embeddings | [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) (1024-dim, self-hosted) |
| Reranker | [BGE cross-encoder/ms-marco-MiniLM-L-6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (self-hosted) |
| LLM gateway | [LiteLLM](https://github.com/BerriAI/litellm) — provider-agnostic, supports multimodal LLMs |
| Table extraction | [Docling Serve](https://github.com/docling-project/docling-serve) |
| API | FastAPI + SSE streaming + JWT auth + rate limiting |
| Job queue | [Arq](https://arq-docs.helpmanual.io) + Redis |
| UI | Streamlit |
| Evaluation | [RAGAS](https://github.com/explodinggradients/ragas) |
| Observability | OpenTelemetry, Prometheus, Grafana, LangSmith |

---

## Project Structure

```
.
├── src/production_rag/
│   ├── api/            # FastAPI routers, middleware (auth, rate limiting)
│   ├── adaptive/       # CRAG, Self-RAG, quality estimator
│   ├── chains/         # RAG chain orchestration
│   ├── core/           # Config, types, LLM client
│   ├── evaluation/     # RAGAS evaluation suite, chunking ablation
│   ├── generation/     # Synthesizer, citation extractor, SSE streaming
│   ├── ingestion/      # Loaders, chunkers, embedder, RAPTOR, Arq worker
│   ├── observability/  # Tracing, metrics, LangSmith
│   ├── query/          # Query transforms (multi-query, HyDE)
│   ├── retrieval/      # Retrievers, RRF/MMR fusion, rerankers
│   └── vectorstore/    # Weaviate client, schema, tenant manager
├── ui/
│   └── streamlit_app.py
├── tests/              # Unit + integration tests (pytest)
├── deploy/             # Prometheus, Grafana, OTel configs
├── notebooks/          # Exploration notebooks
├── docker-compose.yml
├── Dockerfile
├── start.ps1           # One-command start (Windows)
└── deploy.ps1          # Re-apply code changes to running containers
```

---

## API Reference

The full API is documented interactively at **http://localhost:8000/docs**.

### `POST /chat`
Streams a RAG response as Server-Sent Events.

```json
{
  "query": "What is the main finding of this paper?",
  "tenant_id": "default",
  "enable_crag": true,
  "enable_self_rag": true
}
```

**SSE event types:**

| Type | Description |
|---|---|
| `context_chunk` | Each retrieved chunk (score, snippet, source, chunk_type) |
| `token` | Streamed answer token |
| `citation` | Citation object (source, page, snippet) |
| `diagnostics` | Pipeline stats (quality score, hop count, chunk counts) |
| `error` | Pipeline error message |

### `POST /ingest`
Enqueues a background ingestion job.

```json
{
  "pdf_paths": ["/uploads/paper.pdf"],
  "arxiv_ids": ["2312.10997"],
  "tenant_id": "default",
  "chunking_strategy": "docling",
  "enable_raptor": false
}
```

Returns `{ "job_id": "..." }` immediately; the worker processes asynchronously.

---

## Chunking Strategies

| Strategy | Best For |
|---|---|
| `recursive` | General text, most papers |
| `fixed` | Uniform chunk size experiments |
| `semantic` | Thematic boundary detection |
| `hierarchical` | Multi-hop questions requiring parent context |
| `late` | Long-context models that support late chunking |
| `docling` | **Papers with tables, figures, and structured layout** |

The `docling` strategy uses [Docling Serve](https://github.com/docling-project/docling-serve) to parse the PDF structurally — tables are extracted as Markdown and tagged so the LLM can reason over exact rows and columns rather than garbled plain text.

---

## Multi-tenancy

Each corpus is isolated as a Weaviate **tenant**. Set `tenant_id` on every `/chat` and `/ingest` request to route to the correct document store. The default tenant is `"default"`.

This allows multiple independent document collections (e.g. separate research topics, teams, or users) to share the same infrastructure.

---

## Evaluation

Run the RAGAS evaluation suite against ingested documents:

```bash
python -m production_rag.evaluation.ragas_suite --tenant default
```

Run chunking strategy ablation to compare strategies on your corpus:

```bash
python -m production_rag.evaluation.chunking_ablation
```

---

## Configuration Reference

All settings are environment variables. Copy `.env.example` to `.env`.

| Variable | Description | Default |
|---|---|---|
| `LLM_DEFAULT_PROVIDER` | `groq` · `openai` · `anthropic` · `google` · `ollama` | `google` |
| `LLM_DEFAULT_MODEL` | Model name in LiteLLM format | `gemini-2.0-flash` |
| `LLM_FAST_MODEL` | Faster/cheaper model for lightweight tasks | `gemini-2.0-flash-lite` |
| `EMBEDDING_MODEL` | HuggingFace model ID | `BAAI/bge-large-en-v1.5` |
| `RERANKER_PROVIDER` | `bge` (self-hosted) or `cohere` | `bge` |
| `WEAVIATE_URL` | Weaviate endpoint | `http://weaviate:8080` |
| `REDIS_URL` | Redis connection string | `redis://redis:6379/0` |
| `DOCLING_URL` | Docling Serve endpoint | `http://docling-serve:5001` |

---

## License

MIT — see [LICENSE](LICENSE).

---

## Acknowledgements

Built with [Weaviate](https://weaviate.io), [LiteLLM](https://github.com/BerriAI/litellm), [Docling](https://github.com/docling-project/docling), [RAGAS](https://github.com/explodinggradients/ragas), [Arq](https://github.com/samuelcolvin/arq), and [Streamlit](https://streamlit.io).
ui/                    # Streamlit app
tests/                 # unit/ integration/ e2e/
notebooks/             # Evaluation and comparison notebooks
helm/                  # Helm chart for K8s deployment
.github/workflows/     # CI/CD pipelines
```

## Evaluation

```bash
# Run full RAGAS eval suite
python -m production_rag.evaluation.ragas_suite --dataset qasper

# Chunking ablation study
python -m production_rag.evaluation.chunking_ablation

# Retrieval benchmark
python -m production_rag.evaluation.retrieval_benchmark
```

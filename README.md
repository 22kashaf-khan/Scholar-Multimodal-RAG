# production-rag-pipeline

A production-grade Retrieval-Augmented Generation system targeting scientific papers.

## Architecture

```
Query
  └─▶ Query Transforms (parallel)
        ├─ Multi-Query (3 paraphrases)
        ├─ HyDE (hypothetical abstract → embed)
        └─ Step-Back (concept abstraction)
              │
              ▼
  EnsembleRetriever (~12 parallel Weaviate calls)
        ├─ DenseRetriever (near_vector)
        ├─ BM25Retriever (keyword)
        └─ HybridRetriever (native Weaviate hybrid, alpha=0.5)
              │
              ▼
  RRF (k=60) — fuse all rank lists
              │
              ▼
  MMR (λ=0.7) — diversity over top-100
              │
              ▼
  Context Expansion
        ├─ ParentDocument (fetch parent_chunk)
        └─ SentenceWindow (±2 chunks)
              │
              ▼
  Cohere Rerank (top-50 → top-12)
              │
              ▼
  CRAG quality gate → re-query loop (max 2 hops)
              │
              ▼
  Synthesizer (citation-tagged prompt)
              │
              ▼
  Self-RAG critique → regenerate if unsupported claims
              │
              ▼
  SSE streaming response + citation array
```

## Stack

- **Vector store**: Weaviate v4 (native hybrid + multi-tenancy)
- **LLM**: LiteLLM (OpenAI / Anthropic / Ollama / Vertex)
- **Orchestration**: LangChain LCEL + LlamaIndex (parallel implementations)
- **Reranker**: Cohere Rerank API (BGE cross-encoder for Phase 2)
- **Chunking**: Fixed / Recursive / Semantic / Hierarchical (RAPTOR) / Late
- **Evaluation**: RAGAS (faithfulness, answer relevancy, context precision/recall)
- **Serving**: FastAPI + SSE streaming + JWT auth + Redis rate limiting
- **Queue**: Arq + Redis for async ingestion
- **Observability**: OpenTelemetry + LangSmith + Prometheus + Grafana
- **Infra**: Docker Compose (dev + full) + Helm chart + GitHub Actions CI/CD

## Quick Start

```bash
# Install dependencies
pip install -e ".[dev]"

# Copy env config
cp .env.example .env  # fill in your API keys

# Start infrastructure (Weaviate + Redis)
docker compose --profile dev up -d

# Create Weaviate schema
python -m production_rag.cli schema create

# Ingest sample papers
python -m production_rag.cli ingest --arxiv-ids "2005.11401,1706.03762"

# Start API
uvicorn production_rag.api.main:app --reload

# Start UI
streamlit run ui/streamlit_app.py
```

## Project Structure

```
src/production_rag/
├── core/              # Config, LLM client, base types
├── vectorstore/       # Weaviate schema, client, tenant manager
├── ingestion/         # Loaders, chunkers, embedder, RAPTOR, pipeline
├── retrieval/         # Retrievers, fusion (RRF/MMR), context expansion, rerankers
├── query/             # Query transforms, router
├── generation/        # Synthesizer, citation validator, streaming
├── adaptive/          # CRAG, Self-RAG, quality estimator
├── chains/            # LCEL RAG chain, LlamaIndex pipeline
├── evaluation/        # RAGAS suite, chunking ablation, retrieval benchmark
├── observability/     # OTel tracing, LangSmith, Prometheus metrics
└── api/               # FastAPI app, routers, middleware
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

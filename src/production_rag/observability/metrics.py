"""Prometheus metrics for the RAG pipeline."""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram


PIPELINE_LATENCY = Histogram(
    "rag_pipeline_latency_seconds",
    "End-to-end latency per pipeline stage",
    ["stage", "tenant_id"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

RETRIEVAL_CHUNKS = Histogram(
    "rag_retrieval_chunks_returned",
    "Number of chunks returned per retriever call",
    ["retriever_type", "tenant_id"],
    buckets=(1, 5, 10, 20, 50, 100),
)


CRAG_HOPS_TOTAL = Counter(
    "rag_crag_hops_total",
    "Total CRAG re-retrieval hops performed",
    ["tenant_id"],
)

SELF_RAG_RETRIES_TOTAL = Counter(
    "rag_self_rag_retries_total",
    "Total Self-RAG critique-triggered regenerations",
    ["tenant_id"],
)


INGESTION_DOCS_TOTAL = Counter(
    "rag_ingestion_documents_total",
    "Total documents ingested",
    ["tenant_id", "strategy"],
)

INGESTION_CHUNKS_TOTAL = Counter(
    "rag_ingestion_chunks_total",
    "Total chunks ingested (including RAPTOR summary nodes)",
    ["tenant_id", "strategy"],
)


WEAVIATE_OBJECTS_TOTAL = Gauge(
    "rag_weaviate_objects_total",
    "Current number of objects per tenant in Weaviate",
    ["tenant_id"],
)


PIPELINE_ERRORS_TOTAL = Counter(
    "rag_pipeline_errors_total",
    "Total pipeline errors by stage",
    ["stage", "error_type"],
)


class StageTimer:
    """Context manager that records pipeline latency into PIPELINE_LATENCY."""

    def __init__(self, stage: str, tenant_id: str) -> None:
        self._stage = stage
        self._tenant_id = tenant_id
        self._timer = PIPELINE_LATENCY.labels(stage=stage, tenant_id=tenant_id).time()

    def __enter__(self):
        self._timer.__enter__()
        return self

    def __exit__(self, *args):
        self._timer.__exit__(*args)

    async def __aenter__(self):
        self._timer.__enter__()
        return self

    async def __aexit__(self, *args):
        self._timer.__exit__(*args)

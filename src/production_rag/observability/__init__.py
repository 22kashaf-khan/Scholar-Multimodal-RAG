"""Observability: OpenTelemetry tracing, LangSmith, Prometheus metrics."""

from production_rag.observability.tracing import (
    async_trace_span,
    get_tracer,
    setup_tracing,
    shutdown_tracing,
    trace_span,
    traced,
)
from production_rag.observability.langsmith import get_langsmith_handler
from production_rag.observability.metrics import (
    CRAG_HOPS_TOTAL,
    INGESTION_DOCS_TOTAL,
    PIPELINE_ERRORS_TOTAL,
    PIPELINE_LATENCY,
    RETRIEVAL_CHUNKS,
    SELF_RAG_RETRIES_TOTAL,
    StageTimer,
    WEAVIATE_OBJECTS_TOTAL,
)

__all__ = [
    "setup_tracing",
    "shutdown_tracing",
    "get_tracer",
    "trace_span",
    "async_trace_span",
    "traced",
    "get_langsmith_handler",
    "PIPELINE_LATENCY",
    "RETRIEVAL_CHUNKS",
    "CRAG_HOPS_TOTAL",
    "SELF_RAG_RETRIES_TOTAL",
    "INGESTION_DOCS_TOTAL",
    "PIPELINE_ERRORS_TOTAL",
    "WEAVIATE_OBJECTS_TOTAL",
    "StageTimer",
]

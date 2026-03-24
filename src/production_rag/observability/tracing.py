"""OpenTelemetry tracing helpers for the RAG pipeline."""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
from typing import Any, AsyncGenerator, Generator

import structlog
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.trace import StatusCode

log = structlog.get_logger(__name__)

_tracer: trace.Tracer | None = None
_provider: TracerProvider | None = None


def setup_tracing(settings: Any) -> None:
    """Initialise the global OTel TracerProvider.  Call once at startup."""
    global _tracer, _provider

    resource = Resource.create(
        {
            "service.name": settings.otel_service_name,
            "service.version": "1.0.0",
            "deployment.environment": settings.environment,
        }
    )

    _provider = TracerProvider(resource=resource)

    if settings.otel_exporter_otlp_endpoint:
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
    else:
        exporter = ConsoleSpanExporter()

    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("production_rag")
    log.info("tracing.setup", endpoint=getattr(settings, "otel_exporter_otlp_endpoint", "console"))


def get_tracer() -> trace.Tracer:
    if _tracer is None:
        return trace.get_tracer("production_rag")
    return _tracer


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.encode()).hexdigest()[:16]


@contextmanager
def trace_span(
    stage: str,
    *,
    query: str = "",
    tenant_id: str = "",
    attributes: dict[str, Any] | None = None,
) -> Generator[trace.Span, None, None]:
    """Synchronous context manager that wraps a code block in an OTel span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"rag.{stage}") as span:
        span.set_attribute("rag.stage", stage)
        if query:
            span.set_attribute("rag.query_hash", _query_hash(query))
        if tenant_id:
            span.set_attribute("rag.tenant_id", tenant_id)
        for k, v in (attributes or {}).items():
            span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            raise


@asynccontextmanager
async def async_trace_span(
    stage: str,
    *,
    query: str = "",
    tenant_id: str = "",
    attributes: dict[str, Any] | None = None,
) -> AsyncGenerator[trace.Span, None]:
    """Async context manager that wraps an awaitable block in an OTel span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"rag.{stage}") as span:
        span.set_attribute("rag.stage", stage)
        if query:
            span.set_attribute("rag.query_hash", _query_hash(query))
        if tenant_id:
            span.set_attribute("rag.tenant_id", tenant_id)
        for k, v in (attributes or {}).items():
            span.set_attribute(k, v)
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            raise


def traced(stage: str):
    """Decorator that wraps an async function in an OTel span.

    Usage:
        @traced("retrieval_dense")
        async def retrieve(self, query: str, ...) -> list[RetrievedChunk]:
            ...
    """
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            async with async_trace_span(stage):
                return await fn(*args, **kwargs)
        return wrapper
    return decorator


def shutdown_tracing() -> None:
    if _provider is not None:
        _provider.shutdown()
        log.info("tracing.shutdown")

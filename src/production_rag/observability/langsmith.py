"""LangSmith callback integration for LCEL chain observability.

Instruments:
  - All LangChain Runnable calls (via LangSmithCallbackHandler)
  - Custom RAG metadata: tenant_id, retrieval stage scores, CRAG hops,
    Self-RAG retry count, citation validation pass/fail

Usage:
    from production_rag.observability.langsmith import get_langsmith_handler

    handler = get_langsmith_handler(tenant_id="org_abc")
    await chain.ainvoke(query, config={"callbacks": [handler]})
"""

from __future__ import annotations

from typing import Any

import structlog

log = structlog.get_logger(__name__)


def get_langsmith_handler(
    tenant_id: str = "",
    extra_metadata: dict[str, Any] | None = None,
) -> Any:
    """Return a LangSmith callback handler, or None if LangSmith is disabled.

    Returns None (no-op) if LANGSMITH_API_KEY is not set or langsmith package
    is unavailable — never causes the serving path to fail.
    """
    try:
        import os
        if not os.environ.get("LANGSMITH_API_KEY"):
            return None

        from langsmith import Client  # type: ignore[import-untyped]
        from langchain_core.tracers import LangChainTracer  # type: ignore[import-untyped]

        client = Client()
        project = os.environ.get("LANGSMITH_PROJECT", "production-rag")
        handler = LangChainTracer(project_name=project, client=client)

        # Attach extra metadata accessible in LangSmith UI
        meta = {"tenant_id": tenant_id}
        if extra_metadata:
            meta.update(extra_metadata)
        handler.metadata = meta  # type: ignore[attr-defined]

        return handler
    except ImportError:
        log.warning("langsmith.not_available", reason="package not installed")
        return None
    except Exception as exc:
        log.warning("langsmith.init_failed", error=str(exc))
        return None


def build_run_tags(
    tenant_id: str,
    query_complexity: str = "",
    crag_hops: int = 0,
    self_rag_retries: int = 0,
) -> list[str]:
    """Build LangSmith run tags for filtering in the UI."""
    tags = [f"tenant:{tenant_id}"]
    if query_complexity:
        tags.append(f"complexity:{query_complexity}")
    if crag_hops > 0:
        tags.append(f"crag_hops:{crag_hops}")
    if self_rag_retries > 0:
        tags.append(f"self_rag_retries:{self_rag_retries}")
    return tags

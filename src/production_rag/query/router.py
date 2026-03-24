"""Query router — classify query complexity and route to retrieval strategy.

Three routing targets:
- factoid: Single-shot retrieval, leaf chunks only, no RAPTOR
- multi_hop: Full ensemble (multi-query + HyDE + step-back), leaf + RAPTOR section
- summary: RAPTOR paper/section level first, full ensemble

Routing uses the LLM fast model via LLMClient.classify_query_complexity.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

from production_rag.core.llm_client import LLMClient
from production_rag.core.types import ChunkLevel, QueryComplexity

log = structlog.get_logger(__name__)


@dataclass
class RoutingDecision:
    complexity: QueryComplexity
    enable_multi_query: bool
    enable_hyde: bool
    enable_step_back: bool
    raptor_levels: list[ChunkLevel]
    use_parent_expansion: bool


_ROUTING_TABLE: dict[QueryComplexity, RoutingDecision] = {
    QueryComplexity.FACTOID: RoutingDecision(
        complexity=QueryComplexity.FACTOID,
        enable_multi_query=False,
        enable_hyde=False,
        enable_step_back=False,
        raptor_levels=[ChunkLevel.LEAF],
        use_parent_expansion=False,
    ),
    QueryComplexity.MULTI_HOP: RoutingDecision(
        complexity=QueryComplexity.MULTI_HOP,
        enable_multi_query=True,
        enable_hyde=True,
        enable_step_back=True,
        raptor_levels=[ChunkLevel.LEAF, ChunkLevel.SECTION],
        use_parent_expansion=True,
    ),
    QueryComplexity.SUMMARY: RoutingDecision(
        complexity=QueryComplexity.SUMMARY,
        enable_multi_query=True,
        enable_hyde=False,
        enable_step_back=True,
        raptor_levels=[ChunkLevel.SECTION, ChunkLevel.PAPER],
        use_parent_expansion=True,
    ),
}


class QueryRouter:
    """Classify query and return routing decision."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def route(self, query: str) -> RoutingDecision:
        try:
            complexity_str = await self._llm.classify_query_complexity(query)
            complexity = QueryComplexity(complexity_str)
        except (ValueError, Exception) as e:
            log.warning("query_router.classify_failed", error=str(e), fallback="multi_hop")
            complexity = QueryComplexity.MULTI_HOP

        decision = _ROUTING_TABLE[complexity]
        log.info(
            "query_router.decision",
            query=query[:80],
            complexity=complexity.value,
        )
        return decision

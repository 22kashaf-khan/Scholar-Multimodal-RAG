"""Self-RAG — post-generation critique and conditional regeneration.

After generation, uses the LLM to verify every claim is supported by
the retrieved context. Regenerates once if unsupported claims are found.
"""

from __future__ import annotations

import structlog

from production_rag.core.llm_client import LLMClient
from production_rag.core.types import RAGResponse, RetrievedChunk

log = structlog.get_logger(__name__)

_STRICT_SYNTHESIS_SUFFIX = (
    "\n\nSTRICT RULE: Every sentence in your answer MUST be directly supported "
    "by one of the SOURCE passages. Do not include any information not present "
    "in the sources. If you cannot answer fully from the sources, say so explicitly."
)


class SelfRAGCritic:
    """Self-RAG post-generation critique stage.

    Args:
        llm: LLM client used for critique (and optional regeneration).
        max_retries: Maximum regeneration attempts (should be 1).
    """

    def __init__(self, llm: LLMClient, max_retries: int = 1) -> None:
        self._llm = llm
        self._max_retries = max_retries

    async def critique_and_refine(
        self,
        query: str,
        response: RAGResponse,
        chunks: list[RetrievedChunk],
    ) -> RAGResponse:
        """Critique the answer and regenerate if unsupported claims found."""
        context_blocks = "\n\n".join(
            f"[Source {i + 1}] {c.display_text[:600]}"
            for i, c in enumerate(chunks)
        )

        critique = await self._llm.critique_answer(response.answer, context_blocks)
        is_supported: bool = critique.get("supported", True)
        unsupported_claims: list[str] = critique.get("unsupported_claims", [])

        log.info(
            "self_rag.critique",
            supported=is_supported,
            unsupported_count=len(unsupported_claims),
        )

        if is_supported or not unsupported_claims:
            return response


        for attempt in range(self._max_retries):
            log.info("self_rag.regenerating", attempt=attempt + 1)
            from production_rag.generation.synthesizer import Synthesizer, _build_context_block

            # Build a new context block; synthesis will stream separately —
            # here we produce the final synchronous answer for the non-streaming path
            context_str, source_map = _build_context_block(chunks)

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert scientific assistant."
                        + _STRICT_SYNTHESIS_SUFFIX
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{context_str}\n\n{'─'*60}\nQUESTION: {query}"
                    ),
                },
                {
                    "role": "assistant",
                    "content": response.answer,
                },
                {
                    "role": "user",
                    "content": (
                        "The following claims in your previous answer lack source support: "
                        + "; ".join(unsupported_claims)
                        + "\n\nPlease revise your answer to remove or properly cite these claims."
                    ),
                },
            ]

            new_answer, new_tokens = await self._llm.complete(messages, max_tokens=2048)

            new_critique = await self._llm.critique_answer(new_answer, context_blocks)
            if new_critique.get("supported", True):
                log.info("self_rag.regeneration_succeeded")
                from production_rag.generation.synthesizer import _extract_citations
                source_map_numbered = {i + 1: c for i, c in enumerate(chunks)}
                new_citations = _extract_citations(new_answer, source_map_numbered)
                response.answer = new_answer
                response.citations = new_citations
                response.tokens_used += new_tokens
                return response

        log.warning("self_rag.unresolved_unsupported", claims=unsupported_claims)
        return response

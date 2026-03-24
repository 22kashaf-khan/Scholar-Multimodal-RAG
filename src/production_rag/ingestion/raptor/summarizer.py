"""RAPTOR Summarizer.

Calls the LLM to produce a concise summary of a cluster of chunks.
Summaries become the text of parent nodes in the RAPTOR tree.
"""

from __future__ import annotations

import asyncio

import structlog
import tiktoken

from production_rag.core.llm_client import LLMClient

log = structlog.get_logger(__name__)

_MAX_INPUT_TOKENS = 3000   # per cluster summary call
_MAX_OUTPUT_TOKENS = 300
_ENCODING = "cl100k_base"  # works for both GPT-4 and Claude (rough estimate)


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    try:
        enc = tiktoken.get_encoding(_ENCODING)
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except Exception:
        # Fallback: rough char truncation
        return text[: max_tokens * 4]


class RAPTORSummarizer:
    """Async LLM summarizer for RAPTOR cluster nodes."""

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    async def summarize_cluster(
        self, chunk_texts: list[str], level_hint: str = "section"
    ) -> str:
        """Summarize a list of chunk texts into one parent node summary.

        Args:
            chunk_texts: Texts of the chunks in this cluster.
            level_hint: "section" or "paper" — adjusts summary length/style.
        """
        combined = "\n\n---\n\n".join(chunk_texts)
        truncated = _truncate_to_tokens(combined, _MAX_INPUT_TOKENS)

        if level_hint == "paper":
            instruction = (
                "You are a scientific summarizer. Write a comprehensive but concise "
                "abstract-style summary (150–200 words) capturing the key contributions, "
                "methods, and findings of the following scientific passages."
            )
        else:
            instruction = (
                "You are a scientific summarizer. Write a concise section-level summary "
                "(75–120 words) of the following related scientific passages, preserving "
                "key technical terms and findings."
            )

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": truncated},
        ]

        try:
            summary, _ = await self._llm.complete(
                messages,
                use_fast_model=True,
                max_tokens=_MAX_OUTPUT_TOKENS,
            )
            return summary.strip()
        except Exception as e:
            log.error("raptor.summarizer.failed", error=str(e))
            # Fallback: naive extractive summary (first sentence of each chunk)
            sentences = [t.split(".")[0].strip() for t in chunk_texts if t]
            return ". ".join(sentences[:5]) + "."

    async def summarize_all_clusters(
        self,
        cluster_chunks: dict[int, list[str]],
        level_hint: str = "section",
        max_concurrency: int = 8,
    ) -> dict[int, str]:
        """Summarize all clusters concurrently.

        Returns dict[cluster_id → summary_text].
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded(cid: int, texts: list[str]) -> tuple[int, str]:
            async with semaphore:
                summary = await self.summarize_cluster(texts, level_hint)
                log.debug("raptor.summarizer.cluster_done", cluster_id=cid)
                return cid, summary

        tasks = [
            _bounded(cid, texts)
            for cid, texts in cluster_chunks.items()
        ]
        results = await asyncio.gather(*tasks)
        return dict(results)

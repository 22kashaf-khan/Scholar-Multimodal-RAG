"""Provider-agnostic async LLM client built on LiteLLM."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import AsyncGenerator
from typing import Any

import litellm
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from production_rag.core.config import Settings, get_settings

log = structlog.get_logger(__name__)

litellm.set_verbose = False  # type: ignore[attr-defined]


def _build_model_string(provider: str, model: str) -> str:
    """Convert provider + model to LiteLLM model string."""
    if provider == "openai":
        return model
    if provider == "anthropic":
        return f"anthropic/{model}"
    if provider == "ollama":
        return f"ollama/{model}"
    if provider == "vertex":
        return f"vertex_ai/{model}"
    if provider == "google":
        return f"gemini/{model}"
    return model


def _build_kwargs(settings: Settings, model_str: str) -> dict[str, Any]:
    """Build LiteLLM call kwargs from settings."""
    kwargs: dict[str, Any] = {"model": model_str}
    provider = settings.llm_default_provider
    if provider == "google":
        key = settings.google_api_key.get_secret_value()
        if key:
            kwargs["api_key"] = key
    elif provider == "anthropic":
        key = settings.anthropic_api_key.get_secret_value()
        if key:
            kwargs["api_key"] = key
    else:
        key = settings.openai_api_key.get_secret_value()
        if key:
            kwargs["api_key"] = key
    return kwargs


class LLMClient:
    """Async LLM client with streaming, retry, and token tracking."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    @retry(
        retry=retry_if_exception_type((litellm.APIConnectionError, litellm.Timeout)),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    async def _call(
        self,
        messages: list[dict[str, str]],
        model_str: str,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> Any:
        return await litellm.acompletion(  # type: ignore[misc]
            model=model_str,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> tuple[str, int]:
        """Return (text, total_tokens) for a single completion."""
        s = self._settings
        provider = s.llm_default_provider
        model = s.llm_fast_model if use_fast_model else s.llm_default_model
        model_str = _build_model_string(provider, model)
        temp = temperature if temperature is not None else s.llm_temperature

        log.debug("llm.complete", model=model_str, messages_len=len(messages))
        response = await self._call(messages, model_str, temp, max_tokens, stream=False)
        text: str = response.choices[0].message.content or ""
        tokens: int = response.usage.total_tokens if response.usage else 0
        return text, tokens

    async def stream(
        self,
        messages: list[dict[str, str]],
        *,
        use_fast_model: bool = False,
        temperature: float | None = None,
        max_tokens: int = 2048,
    ) -> AsyncGenerator[str, None]:
        """Yield token strings from a streaming completion."""
        s = self._settings
        provider = s.llm_default_provider
        model = s.llm_fast_model if use_fast_model else s.llm_default_model
        model_str = _build_model_string(provider, model)
        temp = temperature if temperature is not None else s.llm_temperature

        log.debug("llm.stream", model=model_str)
        response = await self._call(messages, model_str, temp, max_tokens, stream=True)
        async for chunk in response:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    async def complete_json(
        self,
        messages: list[dict[str, str]],
        *,
        use_fast_model: bool = True,
        max_tokens: int = 512,
    ) -> dict[str, Any]:
        """Complete and parse JSON response.  Raises ValueError on bad JSON."""
        text, _ = await self.complete(
            messages, use_fast_model=use_fast_model, max_tokens=max_tokens
        )
        # Strip markdown fences if model wraps output
        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(text)  # type: ignore[no-any-return]

    async def generate_multi_queries(
        self, query: str, n: int = 3
    ) -> list[str]:
        """Return N diverse paraphrases of the input query (fast model)."""
        cache_key = hashlib.sha256(f"mq:{n}:{query}".encode()).hexdigest()[:16]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scientific information retrieval assistant. "
                    "Generate diverse rephrasings of a query that preserve its intent "
                    "but use different vocabulary and structure. "
                    f"Return a JSON object: {{\"queries\": [list of {n} strings]}}. "
                    "Output only valid JSON."
                ),
            },
            {"role": "user", "content": f"Original query: {query}"},
        ]
        log.debug("query_transform.multi_query", cache_key=cache_key)
        result = await self.complete_json(messages, use_fast_model=True)
        queries: list[str] = result.get("queries", [])
        return queries[:n]

    async def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical abstract that would answer the query."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert scientific writer. Given a research question, "
                    "write a concise hypothetical abstract (150–200 words) from an "
                    "academic paper that would perfectly answer it. "
                    "Write only the abstract — no title, no labels."
                ),
            },
            {"role": "user", "content": query},
        ]
        text, _ = await self.complete(messages, use_fast_model=True, max_tokens=300)
        return text.strip()

    async def generate_step_back_query(self, query: str) -> str:
        """Abstract the query to a higher-level scientific concept question."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scientific reasoning assistant. "
                    "Given a specific research question, produce a more general, "
                    "conceptual question that captures the broader principle or "
                    "domain being asked about. Return only the abstracted question."
                ),
            },
            {"role": "user", "content": query},
        ]
        text, _ = await self.complete(messages, use_fast_model=True, max_tokens=128)
        return text.strip()

    async def classify_query_complexity(
        self, query: str
    ) -> str:  # "factoid" | "multi_hop" | "summary"
        """Route query to retrieval strategy based on complexity."""
        messages = [
            {
                "role": "system",
                "content": (
                    "Classify this scientific research query into exactly one category:\n"
                    "- factoid: seeks a specific fact, number, or definition\n"
                    "- multi_hop: requires synthesising information from multiple sources\n"
                    "- summary: asks for an overview, survey, or comparison\n"
                    'Return JSON: {"complexity": "<category>"}. Output only valid JSON.'
                ),
            },
            {"role": "user", "content": query},
        ]
        result = await self.complete_json(messages, use_fast_model=True, max_tokens=32)
        return result.get("complexity", "multi_hop")

    async def reformulate_query(self, query: str, feedback: str) -> str:
        """CRAG: reformulate a query that yielded low-quality retrieval."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scientific search query optimizer. "
                    "The original query retrieved low-quality results. "
                    "Using the feedback about what was missing, produce a better query. "
                    "Return only the improved query string."
                ),
            },
            {"role": "user", "content": f"Query: {query}\nFeedback: {feedback}"},
        ]
        text, _ = await self.complete(messages, use_fast_model=True, max_tokens=128)
        return text.strip()

    async def critique_answer(self, answer: str, context_blocks: str) -> dict[str, Any]:
        """Self-RAG: critique the answer for unsupported claims.

        Returns {"supported": bool, "unsupported_claims": list[str]}.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a scientific fact-checker. Given an answer and the source "
                    "context that was provided to generate it, identify any claims in "
                    "the answer that are NOT directly supported by the context.\n"
                    "Return JSON: "
                    '{"supported": true/false, "unsupported_claims": ["claim1", ...]}. '
                    "Output only valid JSON."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT:\n{context_blocks}\n\n"
                    f"ANSWER:\n{answer}"
                ),
            },
        ]
        return await self.complete_json(messages, use_fast_model=False, max_tokens=512)


_client: LLMClient | None = None
_lock = asyncio.Lock()


async def get_llm_client() -> LLMClient:
    """Return the global async-safe LLMClient singleton."""
    global _client
    if _client is None:
        async with _lock:
            if _client is None:
                _client = LLMClient()
    return _client

"""LlamaIndex parallel RAG pipeline implementation.

Demonstrates the same retrieval → rerank → synthesis flow using
LlamaIndex's native query engine primitives:
  - SentenceWindowNodeParser for small-to-big chunk strategy
  - MetadataReplacementPostProcessor for synthesis context expansion
  - CohereRerank as a LlamaIndex node postprocessor
  - RetrieverQueryEngine as the top-level orchestrator

Used in the comparison notebook (notebooks/04_langchain_vs_llamaindex.ipynb)
to benchmark quality and latency vs the LCEL RAGChain.

Note: This implementation uses a LlamaIndex in-memory vector store backed
by numpy for the notebook; swap to WeaviateVectorStore in production.
"""

from __future__ import annotations

from typing import Any

import structlog

from production_rag.core.config import Settings, get_settings
from production_rag.core.llm_client import LLMClient

log = structlog.get_logger(__name__)


class LlamaIndexPipeline:
    """LlamaIndex-based RAG pipeline for comparison benchmarking."""

    def __init__(
        self,
        llm: LLMClient,
        settings: Settings | None = None,
    ) -> None:
        self._llm = llm
        self._settings = settings or get_settings()
        self._query_engine: Any = None

    def build(self, documents: list[Any]) -> None:
        """Build the LlamaIndex pipeline from a list of LlamaIndex Documents."""
        try:
            from llama_index.core import Settings as LISettings, VectorStoreIndex
            from llama_index.core.node_parser import (
                HierarchicalNodeParser,
                SentenceWindowNodeParser,
            )
            from llama_index.core.postprocessor import MetadataReplacementPostProcessor
            from llama_index.postprocessor.cohere_rerank import CohereRerank
            from llama_index.core.response_synthesizers import get_response_synthesizer
            from llama_index.core.retrievers import AutoMergingRetriever
        except ImportError as e:
            raise ImportError(
                "LlamaIndex dependencies required: pip install llama-index"
            ) from e


        node_parser = SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        )

        index = VectorStoreIndex.from_documents(
            documents,
            transformations=[node_parser],
            show_progress=True,
        )

        metadata_replacer = MetadataReplacementPostProcessor(
            target_metadata_key="window"
        )


        cohere_key = self._settings.cohere_api_key.get_secret_value()
        reranker = CohereRerank(
            api_key=cohere_key,
            model=self._settings.cohere_rerank_model,
            top_n=self._settings.retrieval_final_top_k,
        )


        self._query_engine = index.as_query_engine(
            similarity_top_k=self._settings.retrieval_rerank_top_n,
            node_postprocessors=[metadata_replacer, reranker],
            response_synthesizer=get_response_synthesizer(
                response_mode="compact",
            ),
        )

        log.info("llamaindex_pipeline.built", docs=len(documents))

    async def query(self, question: str) -> dict[str, Any]:
        """Run a query and return answer + source nodes."""
        if self._query_engine is None:
            raise RuntimeError("Pipeline not built. Call build() first.")

        import asyncio

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, self._query_engine.query, question
        )

        sources = [
            {
                "node_id": node.node.node_id,
                "score": node.score,
                "text": node.node.text[:300],
            }
            for node in response.source_nodes
        ]

        return {
            "answer": str(response),
            "sources": sources,
        }

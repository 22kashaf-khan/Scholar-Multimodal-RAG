"""Application configuration — driven entirely by environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    weaviate_url: str = "http://localhost:8080"
    weaviate_api_key: SecretStr = Field(default=SecretStr(""))

    openai_api_key: SecretStr = Field(default=SecretStr(""))
    anthropic_api_key: SecretStr = Field(default=SecretStr(""))
    google_api_key: SecretStr = Field(default=SecretStr(""))
    ollama_base_url: str = "http://localhost:11434"

    # LLM routing
    llm_default_provider: Literal["openai", "anthropic", "ollama", "vertex", "google"] = "openai"
    llm_default_model: str = "gpt-4o"
    llm_fast_model: str = "gpt-4o-mini"  # used for query transforms / critique
    llm_temperature: float = 0.0

    embedding_provider: Literal["openai", "sentence_transformers", "jina"] = (
        "sentence_transformers"
    )
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dimension: int = 1024

    cohere_api_key: SecretStr = Field(default=SecretStr(""))
    reranker_provider: Literal["cohere", "bge"] = "cohere"
    cohere_rerank_model: str = "rerank-english-v3.0"
    bge_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    redis_url: str = "redis://localhost:6379/0"

    jwt_algorithm: str = "RS256"
    jwt_audience: str = "rag-api"
    jwt_issuer: str = ""
    jwks_uri: str = ""

    rate_limit_chat: int = 30
    rate_limit_ingest: int = 10

    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    langchain_api_key: SecretStr = Field(default=SecretStr(""))
    langchain_tracing_v2: bool = False
    langchain_project: str = "production-rag-pipeline"


    retrieval_top_k_candidates: int = 200  # total candidates pulled per retriever
    retrieval_rrf_k: int = 60              # RRF smoothing constant
    retrieval_mmr_lambda: float = 0.7      # MMR relevance weight (0=pure diversity)
    retrieval_rerank_top_n: int = 50       # candidates sent to reranker
    retrieval_final_top_k: int = 12        # chunks injected into synthesis prompt
    retrieval_hybrid_alpha: float = 0.5    # Weaviate hybrid: 0=BM25, 1=dense

    raptor_enabled: bool = True
    raptor_max_cluster_size: int = 10
    raptor_umap_n_components: int = 10
    raptor_umap_n_neighbors: int = 15

    crag_quality_threshold: float = 0.35   # below → trigger corrective re-query
    crag_max_hops: int = 2
    self_rag_enabled: bool = True

    @field_validator("retrieval_mmr_lambda")
    @classmethod
    def _validate_lambda(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("retrieval_mmr_lambda must be in [0, 1]")
        return v

    @field_validator("retrieval_hybrid_alpha")
    @classmethod
    def _validate_alpha(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("retrieval_hybrid_alpha must be in [0, 1]")
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()

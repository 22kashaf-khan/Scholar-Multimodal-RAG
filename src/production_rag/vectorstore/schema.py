"""Weaviate ScientificChunk collection schema definition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import weaviate.classes as wvc
from weaviate.classes.config import (
    Configure,
    DataType,
    Property,
    Tokenization,
    VectorDistances,
)

if TYPE_CHECKING:
    import weaviate

COLLECTION_NAME = "ScientificChunk"

# Full metadata property list — keep in sync with core.types.Chunk
PROPERTIES: list[Property] = [

    Property(
        name="chunk_text",
        data_type=DataType.TEXT,
        description="Full text of this chunk.",
        tokenization=Tokenization.WORD,          # BM25 on prose
        index_filterable=False,
        index_searchable=True,
    ),
    Property(
        name="title",
        data_type=DataType.TEXT,
        description="Paper title (duplicated for BM25 on title field).",
        tokenization=Tokenization.WORD,
        index_filterable=True,
        index_searchable=True,
    ),

    Property(
        name="chunk_id",
        data_type=DataType.TEXT,
        description="Stable deterministic chunk identifier.",
        tokenization=Tokenization.FIELD,         # exact match
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="doc_id",
        data_type=DataType.TEXT,
        description="Parent document identifier.",
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="parent_chunk_id",
        data_type=DataType.TEXT,
        description="ID of parent chunk in hierarchy (null for paper-level).",
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="chunk_level",
        data_type=DataType.TEXT,
        description="RAPTOR level: leaf | section | paper.",
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="chunk_index",
        data_type=DataType.INT,
        description="Positional index within doc for sentence-window expansion.",
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="section_index",
        data_type=DataType.INT,
        description="Section index within document.",
        index_filterable=True,
        index_searchable=False,
    ),

    Property(
        name="start_char",
        data_type=DataType.INT,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="end_char",
        data_type=DataType.INT,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="page",
        data_type=DataType.INT,
        index_filterable=True,
        index_searchable=False,
    ),

    Property(
        name="arxiv_id",
        data_type=DataType.TEXT,
        description="ArXiv paper identifier (e.g. '2005.11401').",
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=True,
    ),
    Property(
        name="paper_doi",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="publication_year",
        data_type=DataType.INT,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="authors",
        data_type=DataType.TEXT_ARRAY,
        tokenization=Tokenization.WORD,
        index_filterable=False,
        index_searchable=True,
    ),
    # tenant_id stored for query routing (Weaviate also tracks tenancy natively)
    Property(
        name="tenant_id",
        data_type=DataType.TEXT,
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    ),
    Property(
        name="chunk_type",
        data_type=DataType.TEXT,
        description="Content type: text | table | figure.",
        tokenization=Tokenization.FIELD,
        index_filterable=True,
        index_searchable=False,
    ),
]


def build_collection_config(embedding_dimension: int = 1024) -> dict:  # type: ignore[type-arg]
    """Return keyword args for weaviate client.collections.create()."""
    return {
        "name": COLLECTION_NAME,
        "description": "Scientific paper chunks with multi-level RAPTOR hierarchy.",
        "properties": PROPERTIES,
        "vectorizer_config": [
            Configure.NamedVectors.none(
                name="semantic_vector",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=256,
                    max_connections=64,
                    dynamic_ef_min=100,
                    dynamic_ef_max=500,
                    ef=-1,  # dynamic EF at query time
                ),
            ),
            Configure.NamedVectors.none(
                name="title_vector",
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric=VectorDistances.COSINE,
                    ef_construction=128,
                    max_connections=32,
                ),
            ),
        ],
        "inverted_index_config": Configure.inverted_index(
            bm25_b=0.75,
            bm25_k1=1.2,
            index_null_state=False,
            index_property_length=False,
            index_timestamps=False,
        ),
        "multi_tenancy_config": Configure.multi_tenancy(
            enabled=True,
            auto_tenant_creation=False,  # explicit lifecycle management
            auto_tenant_activation=True,
        ),
        "replication_config": Configure.replication(factor=1),
    }

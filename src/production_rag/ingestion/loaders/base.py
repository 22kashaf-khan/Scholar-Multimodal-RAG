"""Abstract base for all document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from production_rag.core.types import Document


class BaseLoader(ABC):
    """Loaders convert a source (file path, URL, API call) into Document objects."""

    @abstractmethod
    async def load(self, source: str) -> list[Document]:
        """Load one or more documents from the given source identifier."""
        ...

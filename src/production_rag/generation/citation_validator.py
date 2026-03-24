"""Citation validator.

Post-generation check: verifies that every [SOURCE N] cited in the answer
actually corresponds to a chunk that was retrieved, and that there is
meaningful text overlap between the claim and the source.

Flags low-confidence or invalid citations without raising exceptions —
callers decide whether to regenerate or surface the warning.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

from production_rag.core.types import Citation, RetrievedChunk

log = structlog.get_logger(__name__)


@dataclass
class ValidationResult:
    is_valid: bool
    invalid_citation_ids: list[str]  # cite_N that point to non-existent chunks
    low_overlap_citation_ids: list[str]  # citations with <overlap_threshold text match
    unsupported_claims: list[str]  # [UNSUPPORTED: ...] blocks found in answer


def _text_overlap_ratio(answer_fragment: str, chunk_text: str) -> float:
    """Naive token overlap ratio between a sentence and a chunk."""
    a_tokens = set(re.findall(r"\w+", answer_fragment.lower()))
    c_tokens = set(re.findall(r"\w+", chunk_text.lower()))
    if not a_tokens:
        return 0.0
    return len(a_tokens & c_tokens) / len(a_tokens)


class CitationValidator:
    def __init__(self, overlap_threshold: float = 0.15) -> None:
        self._threshold = overlap_threshold

    def validate(
        self,
        answer: str,
        citations: list[Citation],
        retrieved_chunks: list[RetrievedChunk],
    ) -> ValidationResult:
        chunk_id_set = {c.chunk.chunk_id for c in retrieved_chunks}
        chunk_text_map = {c.chunk.chunk_id: c.display_text for c in retrieved_chunks}

        invalid: list[str] = []
        low_overlap: list[str] = []

        for citation in citations:
            # Check chunk_id is in retrieved set
            if citation.chunk_id not in chunk_id_set:
                invalid.append(citation.citation_id)
                log.warning(
                    "citation_validator.invalid",
                    citation_id=citation.citation_id,
                    chunk_id=citation.chunk_id,
                )
                continue

            # Check text overlap
            source_text = chunk_text_map.get(citation.chunk_id, "")
            overlap = _text_overlap_ratio(citation.snippet, source_text)
            if overlap < self._threshold:
                low_overlap.append(citation.citation_id)
                log.debug(
                    "citation_validator.low_overlap",
                    citation_id=citation.citation_id,
                    overlap=round(overlap, 3),
                )

        # Extract [UNSUPPORTED: ...] markers from answer
        unsupported = re.findall(r"\[UNSUPPORTED:\s*([^\]]+)\]", answer)

        is_valid = not invalid and not unsupported

        if not is_valid:
            log.warning(
                "citation_validator.failed",
                invalid_count=len(invalid),
                unsupported_count=len(unsupported),
            )

        return ValidationResult(
            is_valid=is_valid,
            invalid_citation_ids=invalid,
            low_overlap_citation_ids=low_overlap,
            unsupported_claims=unsupported,
        )

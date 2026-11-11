"""Relevance-ranked, balanced retrieval with source-level deduplication."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from modules.embedding_engine import EmbeddingEngine
from modules.text_chunker import EvidenceChunk


@dataclass(frozen=True)
class RetrievedEvidence:
    chunk: EvidenceChunk
    score: float


REFERENCE_HEADING = re.compile(
    r"^\s*(?:references|bibliography|works cited|literature cited)\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
DOI_OR_URL = re.compile(r"\b(?:doi\s*:|https?://|www\.)", re.IGNORECASE)
YEAR_CITATION = re.compile(r"\b(?:18|19|20)\d{2}[a-z]?\b")
AUTHOR_CITATION = re.compile(r"\b(?:et al\.?|vol\.|no\.|pp?\.|doi)\b", re.IGNORECASE)
REFERENCE_SECTION_CUE = re.compile(
    r"\b(?:references|bibliography|works cited|literature cited)\b",
    re.IGNORECASE,
)
RESULT_CUES = re.compile(
    r"\b(?:abstract|results?|findings?|discussion|conclusion|concluding|achieved|"
    r"outperformed|highest|best|improved|increased|reduced|significant|"
    r"demonstrated|we found|our results)\b",
    re.IGNORECASE,
)
DIRECT_ANSWER_CUES = re.compile(
    r"\b(?:we found|results? (?:show|showed|indicate|indicated)|"
    r"evidence suggests?|was associated with|was higher than|"
    r"was lower than|no significant difference|significantly improved)\b",
    re.IGNORECASE,
)
LITERATURE_REVIEW_CUES = re.compile(
    r"\b(?:previous studies|prior research|related work|literature review|"
    r"earlier studies|previously proposed|existing studies|review of)\b",
    re.IGNORECASE,
)
WORD_PATTERN = re.compile(r"[a-z0-9]+")


def is_reference_passage(text: str) -> bool:
    """Detect reference headings and citation-dense bibliography entries."""
    stripped = text.strip()
    if not stripped:
        return False
    if REFERENCE_HEADING.search(stripped):
        return True
    if REFERENCE_SECTION_CUE.search(stripped) and len(stripped.split()) < 50:
        return True
    words = stripped.split()
    if len(words) < 12:
        return False
    years = len(YEAR_CITATION.findall(stripped))
    author_markers = len(AUTHOR_CITATION.findall(stripped))
    doi_or_urls = len(DOI_OR_URL.findall(stripped))
    citation_signals = years + author_markers + doi_or_urls
    return (
        (years >= 1 and author_markers >= 1 and doi_or_urls >= 1)
        or
        (years >= 2 and (author_markers >= 1 or doi_or_urls >= 1))
        or (doi_or_urls >= 2)
        or (citation_signals >= 4 and len(words) < 100)
    )


def _adjusted_rank_score(similarity: float, text: str) -> float:
    """Apply explainable section/value cues to ordering, not displayed similarity."""
    lowered = text.casefold()
    score = similarity
    if RESULT_CUES.search(lowered):
        score *= 1.18
    if DIRECT_ANSWER_CUES.search(lowered):
        score *= 1.10
    if LITERATURE_REVIEW_CUES.search(lowered):
        score *= 0.78
    words = lowered.split()
    citation_signals = (
        len(YEAR_CITATION.findall(lowered))
        + len(AUTHOR_CITATION.findall(lowered))
        + len(DOI_OR_URL.findall(lowered))
    )
    if len(words) >= 12 and citation_signals >= 3:
        score *= 0.55
    return score


def _normalized_text(text: str) -> str:
    return " ".join(WORD_PATTERN.findall(text.casefold()))


def _near_duplicate(first: str, second: str) -> bool:
    first_words = set(WORD_PATTERN.findall(first.casefold()))
    second_words = set(WORD_PATTERN.findall(second.casefold()))
    if not first_words or not second_words:
        return False
    numeric_tokens = re.compile(r"\d+(?:\.\d+)?%?")
    if set(numeric_tokens.findall(first)) != set(numeric_tokens.findall(second)):
        return False
    intersection = len(first_words & second_words)
    union = len(first_words | second_words)
    return intersection / union >= 0.90


def _deduplicate(
    ranked: list[tuple[EvidenceChunk, float, float]],
) -> list[tuple[EvidenceChunk, float, float]]:
    unique: list[tuple[EvidenceChunk, float, float]] = []
    seen_exact: set[str] = set()
    for candidate in ranked:
        chunk = candidate[0]
        normalized = _normalized_text(chunk.text)
        if normalized in seen_exact:
            continue
        if any(_near_duplicate(chunk.text, existing[0].text) for existing in unique):
            continue
        seen_exact.add(normalized)
        unique.append(candidate)
    return unique


def retrieve_evidence(
    query: str,
    chunks: Sequence[EvidenceChunk],
    top_k: int = 10,
    engine: EmbeddingEngine | None = None,
) -> list[RetrievedEvidence]:
    """Rank evidence across papers, balancing source papers and retaining provenance."""
    if not query.strip() or not chunks or top_k <= 0:
        return []

    searchable = [chunk for chunk in chunks if not is_reference_passage(chunk.text)]
    if not searchable:
        return []
    engine = engine or EmbeddingEngine()
    scores = engine.similarities(query.strip(), [chunk.text for chunk in searchable])
    ranked = [
        (chunk, float(score), _adjusted_rank_score(float(score), chunk.text))
        for chunk, score in zip(searchable, scores)
        if score > 0.0
    ]
    ranked.sort(
        key=lambda item: (
            -item[2],
            item[0].paper_name.casefold(),
            item[0].page_number,
            item[0].chunk_id,
        )
    )
    ranked = _deduplicate(ranked)

    # Give each paper its strongest nonduplicate hit before filling remaining slots.
    by_paper: dict[str, list[tuple[EvidenceChunk, float, float]]] = {}
    for candidate in ranked:
        by_paper.setdefault(candidate[0].paper_id, []).append(candidate)
    paper_leaders = sorted(
        (items[0] for items in by_paper.values()),
        key=lambda item: (-item[2], item[0].paper_name.casefold()),
    )
    selected = paper_leaders[:top_k]
    selected_ids = {item[0].chunk_id for item in selected}
    if len(selected) < top_k:
        selected.extend(
            item for item in ranked
            if item[0].chunk_id not in selected_ids
        )
    return [
        RetrievedEvidence(chunk=chunk, score=score)
        for chunk, score, _ in selected[:top_k]
    ]

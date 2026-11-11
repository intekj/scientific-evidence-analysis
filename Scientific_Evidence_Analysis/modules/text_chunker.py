"""Sentence-based chunking that retains paper and page provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass

from modules.pdf_processor import PaperPage, ProcessedPaper

SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


@dataclass(frozen=True)
class EvidenceChunk:
    chunk_id: str
    paper_id: str
    paper_name: str
    page_number: int
    text: str


def chunk_pages(
    pages: tuple[PaperPage, ...] | list[PaperPage],
    max_words: int = 180,
    overlap_sentences: int = 1,
) -> list[EvidenceChunk]:
    """Split each page independently so page references remain exact."""
    if max_words < 1:
        raise ValueError("max_words must be positive.")
    if overlap_sentences < 0:
        raise ValueError("overlap_sentences cannot be negative.")

    chunks: list[EvidenceChunk] = []
    for page in pages:
        sentences = [
            sentence.strip()
            for sentence in SENTENCE_BOUNDARY.split(page.text)
            if sentence.strip()
        ]
        if not sentences:
            continue
        bounded_sentences: list[str] = []
        for sentence in sentences:
            words = sentence.split()
            bounded_sentences.extend(
                " ".join(words[start:start + max_words])
                for start in range(0, len(words), max_words)
            )
        groups: list[list[str]] = []
        current: list[str] = []
        current_size = 0
        for sentence in bounded_sentences:
            sentence_words = len(sentence.split())
            if current and current_size + sentence_words > max_words:
                groups.append(current)
                current = current[-overlap_sentences:] if overlap_sentences else []
                current_size = sum(len(item.split()) for item in current)
            current.append(sentence)
            current_size += sentence_words
        if current:
            groups.append(current)

        for index, group in enumerate(groups, start=1):
            chunks.append(
                EvidenceChunk(
                    chunk_id=f"{page.paper_id}:{page.page_number}:{index}",
                    paper_id=page.paper_id,
                    paper_name=page.paper_name,
                    page_number=page.page_number,
                    text=" ".join(group),
                )
            )
    return chunks


def chunk_paper(paper: ProcessedPaper, max_words: int = 180) -> list[EvidenceChunk]:
    return chunk_pages(paper.pages, max_words=max_words)

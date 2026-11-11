"""Extractive, source-linked helpers for presenting retrieved evidence."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Sequence

from modules.claim_verification import (
    ClassifiedEvidence,
    EvidenceLabel,
    classify_evidence,
)
from modules.context_extraction import extract_context
from modules.embedding_engine import meaningful_terms
from modules.evidence_retrieval import RetrievedEvidence, is_reference_passage

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
ANSWER_HEADING = re.compile(r"^## 8\. Overall Answer\s*$", re.MULTILINE)
NEXT_HEADING = re.compile(r"^## \d+\.", re.MULTILINE)
QUALIFYING_CUES = re.compile(
    r"\b(?:although|but|however|may|might|could|suggests?|possibly|potentially|"
    r"limited|limitation|uncertain|uncertainty|not statistically significant|"
    r"only|except|whereas|caveat|association rather than causation)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KeyFinding:
    sentence: str
    evidence: RetrievedEvidence


def relevant_sentences(query: str, passage: str) -> list[str]:
    """Return exact passage sentences with meaningful query-term overlap."""
    query_terms = meaningful_terms(query)
    if not query_terms:
        return []
    return [
        sentence.strip()
        for sentence in SENTENCE_SPLIT.split(passage)
        if sentence.strip()
        and not is_reference_passage(sentence)
        and query_terms.intersection(meaningful_terms(sentence))
    ]


def extract_key_findings(
    query: str,
    evidence: Sequence[RetrievedEvidence],
    limit: int = 5,
) -> list[KeyFinding]:
    """Select up to five source-verbatim sentences that overlap the query."""
    if limit <= 0:
        return []
    query_terms = meaningful_terms(query)
    if not query_terms:
        return []

    candidates: list[tuple[int, float, int, str, RetrievedEvidence]] = []
    seen: set[str] = set()
    for item in evidence:
        for sentence in SENTENCE_SPLIT.split(item.chunk.text):
            sentence = sentence.strip()
            normalized = " ".join(sentence.casefold().split())
            if not sentence or normalized in seen or is_reference_passage(sentence):
                continue
            overlap = len(query_terms & meaningful_terms(sentence))
            if not overlap:
                continue
            seen.add(normalized)
            candidates.append(
                (overlap, overlap / len(query_terms), item.score, sentence, item)
            )
    candidates.sort(key=lambda candidate: (-candidate[0], -candidate[1], -candidate[2]))
    return [
        KeyFinding(sentence=sentence, evidence=item)
        for _, _, _, sentence, item in candidates[:limit]
    ]


def classify_for_display(
    query: str,
    item: RetrievedEvidence,
) -> tuple[str, ClassifiedEvidence]:
    """Use existing claim rules and distinguish caveats/background conservatively."""
    classified = classify_evidence(query, item)
    if classified.label != EvidenceLabel.NEUTRAL:
        return classified.label.value, classified
    if QUALIFYING_CUES.search(item.chunk.text):
        return "Qualifying Evidence", classified
    if not extract_context(item.chunk.text).get("Reported findings"):
        return "Background / Context", classified
    reason = classified.reason.casefold()
    if "research question" in reason:
        return "Relevant (not polarity-classified)", classified
    return "Insufficient / Unclassified", classified


def synthesis_answer(
    report: str,
    findings: Sequence[KeyFinding] = (),
) -> str:
    """Use synthesis to check sufficiency, then answer with cited source sentences."""
    heading = ANSWER_HEADING.search(report)
    if heading is None:
        return "The retrieved evidence is insufficient to provide an answer."
    end = NEXT_HEADING.search(report, heading.end())
    section = report[heading.end():end.start() if end else len(report)].strip()
    if not section or "no passages were retrieved" in section.casefold():
        return "The retrieved evidence is insufficient to provide an answer."
    answer = section.split("\n\n", maxsplit=1)[0].strip()
    if "did not contain enough attributable" in answer.casefold():
        return "The retrieved evidence is insufficient to provide an answer."
    if findings:
        cited_findings = []
        for finding in findings[:2]:
            chunk = finding.evidence.chunk
            cited_findings.append(
                f"“{finding.sentence}” "
                f"([{chunk.paper_name}, paper ID {chunk.paper_id}, "
                f"p. {chunk.page_number}])"
            )
        return "Retrieved evidence reports: " + " ".join(cited_findings)
    return answer


def highlighted_passage(query: str, passage: str) -> str:
    """Render the passage with exact query-overlapping sentences highlighted."""
    relevant = set(relevant_sentences(query, passage))
    rendered = []
    for sentence in SENTENCE_SPLIT.split(passage):
        if not sentence:
            continue
        escaped = html.escape(sentence)
        rendered.append(
            f"<mark>{escaped}</mark>" if sentence.strip() in relevant else escaped
        )
    return " ".join(rendered)

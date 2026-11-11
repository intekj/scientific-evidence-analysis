"""Context-aware comparison of evidence from separate studies."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from modules.context_extraction import extract_context
from modules.embedding_engine import meaningful_terms
from modules.text_chunker import EvidenceChunk


class DisagreementCategory(str, Enum):
    GENUINE = "Potential Genuine Contradiction"
    CONTEXT_DEPENDENT = "Context-Dependent Difference"
    INSUFFICIENT = "Insufficient Information"


@dataclass(frozen=True)
class DisagreementResult:
    category: DisagreementCategory
    reason: str
    differing_fields: tuple[str, ...]
    shared_fields: tuple[str, ...]
    first_context: dict[str, str | None]
    second_context: dict[str, str | None]


POSITIVE_CUES = {
    "increase", "increases", "increased", "improve", "improves", "improved",
    "higher", "enhance", "enhanced", "benefit", "beneficial", "effective",
    "positive", "promote", "promoted", "greater", "significant",
}
NEGATIVE_CUES = {
    "decrease", "decreases", "decreased", "reduce", "reduces", "reduced",
    "lower", "worse", "impair", "impaired", "harmful", "negative", "inhibit",
    "inhibited", "declined", "ineffective", "no", "not", "without", "failed",
    "nonsignificant", "non-significant",
}
CONTEXT_CUES = (POSITIVE_CUES | NEGATIVE_CUES) - {"no", "not", "without", "failed"}
FINDINGS_FIELD = "Reported findings"
NON_CONTEXT_FIELDS = {FINDINGS_FIELD, "Metric values"}
RANKING_CUES = {
    "highest", "best", "top", "top-performing", "rank", "ranked", "ranking",
    "outperformed", "outperform", "winner", "first",
}


def _context_signature(value: str | None) -> str:
    if not value:
        return ""
    words = re.findall(r"[a-z0-9]+", value.casefold())
    return " ".join(word for word in words if word not in CONTEXT_CUES)


def _finding_direction(text: str) -> int:
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & RANKING_CUES:
        return 0
    positive = bool(words & POSITIVE_CUES)
    negative = bool(words & NEGATIVE_CUES)
    if positive == negative:
        return 0
    direction = 1 if positive else -1
    if words & {"not", "no", "without", "failed"} and positive:
        direction *= -1
    return direction


def compare_evidence(first: EvidenceChunk, second: EvidenceChunk) -> DisagreementResult:
    """Assess whether opposing result cues could be explained by context differences."""
    first_context = extract_context(first.text)
    second_context = extract_context(second.text)
    shared_terms = meaningful_terms(first.text) & meaningful_terms(second.text)
    shared = tuple(
        field for field in first_context
        if field not in NON_CONTEXT_FIELDS
        if first_context[field] and second_context[field]
    )
    differing = tuple(
        field
        for field in shared
        if field != "Metric values"
        if _context_signature(first_context[field])
        != _context_signature(second_context[field])
    )
    opposite_findings = (
        _finding_direction(first.text) * _finding_direction(second.text) == -1
    )
    findings_differ = bool(
        first_context[FINDINGS_FIELD]
        and second_context[FINDINGS_FIELD]
        and _context_signature(first_context[FINDINGS_FIELD])
        != _context_signature(second_context[FINDINGS_FIELD])
    )
    ranking_only = bool(
        set(re.findall(r"[a-z]+", first.text.casefold())) & RANKING_CUES
        or set(re.findall(r"[a-z]+", second.text.casefold())) & RANKING_CUES
    )

    if not shared_terms:
        return DisagreementResult(
            DisagreementCategory.INSUFFICIENT,
            "The passages have no clear shared content terms, so they cannot be "
            "treated as competing findings based on these excerpts.",
            differing,
            shared,
            first_context,
            second_context,
        )
    if differing and findings_differ:
        difference_names = ", ".join(differing)
        ranking_note = (
            " The passages describe different model rankings or metric values; "
            "ranking differences alone are not treated as contradictions."
            if ranking_only
            else ""
        )
        return DisagreementResult(
            DisagreementCategory.CONTEXT_DEPENDENT,
            "Both passages report findings, and their extracted study context differs "
            f"in {difference_names}. These context differences may explain why the "
            "reported results are not directly "
            f"comparable.{ranking_note}",
            differing,
            shared,
            first_context,
            second_context,
        )
    if not opposite_findings:
        return DisagreementResult(
            DisagreementCategory.INSUFFICIENT,
            "The passages do not contain clearly opposing result cues or distinct "
            "findings alongside a detected context difference. Different model "
            "rankings alone are not treated as contradictions.",
            differing,
            shared,
            first_context,
            second_context,
        )
    if differing:
        return DisagreementResult(
            DisagreementCategory.CONTEXT_DEPENDENT,
            "The passages contain opposing result cues, but extracted context differs "
            f"in {', '.join(differing)}. Those differences may explain the results; "
            "a human should inspect the cited passages.",
            differing,
            shared,
            first_context,
            second_context,
        )
    comparable_fields = [
        field for field in shared
        if field not in NON_CONTEXT_FIELDS
        and _context_signature(first_context[field])
        == _context_signature(second_context[field])
    ]
    if len(comparable_fields) >= 2 and not differing and not ranking_only:
        return DisagreementResult(
            DisagreementCategory.GENUINE,
            "Opposing result cues appear under at least two matching extracted context "
            "fields and no differing shared fields were found. This is only a potential "
            "contradiction, not a scientific determination.",
            (),
            shared,
            first_context,
            second_context,
        )
    return DisagreementResult(
        DisagreementCategory.INSUFFICIENT,
        "Opposing result cues were detected, but too few context fields are available "
        "to establish whether the studies are comparable.",
        (),
        shared,
        first_context,
        second_context,
    )

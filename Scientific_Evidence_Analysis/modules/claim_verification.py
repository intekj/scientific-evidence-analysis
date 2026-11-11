"""Transparent heuristic claim/evidence grouping; not a natural-language inference model."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from modules.context_extraction import MODEL_PATTERN, extract_context
from modules.embedding_engine import meaningful_terms
from modules.evidence_retrieval import RetrievedEvidence, retrieve_evidence
from modules.text_chunker import EvidenceChunk


class EvidenceLabel(str, Enum):
    SUPPORTING = "Supporting Evidence"
    CONTRADICTING = "Contradicting Evidence"
    NEUTRAL = "Neutral / Insufficient Evidence"


@dataclass(frozen=True)
class ClassifiedEvidence:
    evidence: RetrievedEvidence
    label: EvidenceLabel
    reason: str


POSITIVE_CUES = {
    "increase", "increases", "increased", "increasing", "improve", "improves",
    "improved", "higher", "enhance", "enhances", "enhanced", "benefit",
    "beneficial", "effective", "positive", "promote", "promotes", "promoted",
    "greater", "improvement", "achieved",
    "outperformed", "outperforms", "best", "highest", "superior",
}
NEGATIVE_CUES = {
    "decrease", "decreases", "decreased", "decreasing", "reduce", "reduces",
    "reduced", "reducing", "lower", "worse", "impair", "impairs", "impaired",
    "harmful", "negative", "inhibit", "inhibits", "inhibited", "decline",
    "declines", "declined", "ineffective", "no", "not", "without", "failed",
    "failure", "nonsignificant", "non-significant", "underperformed",
    "underperforms", "inferior", "worsened",
}
NEGATION_WORDS = {"not", "no", "without", "never", "neither", "failed"}
ASSERTION_CUES = {
    "is", "are", "was", "were", "has", "have", "had", "achieves", "achieved",
    "reports", "reported", "shows", "showed", "equals", "equal", "reaches",
    "reached",
}
QUESTION_START = re.compile(
    r"^\s*(?:what|which|who|whom|whose|when|where|why|how|"
    r"does|do|did|is|are|was|were|can|could|would|should|will|has|have|had)\b",
    re.IGNORECASE,
)
RANKING_CLAIM_CUES = {"best", "highest", "top", "ranked"}
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
METRIC_PATTERN = re.compile(
    r"\b(AUC|overall accuracy|accuracy|F1(?:-score)?|precision|recall)\b"
    r"[^0-9<>≤≥]{0,24}(?:[<>]=?|≥|≤)?\s*(\d+(?:\.\d+)?)\s*%?",
    re.IGNORECASE,
)


def _direction(text: str) -> int:
    """Return polarity only for explicit outcome words, handling nearby negation."""
    words = re.findall(r"[a-z]+", text.lower())
    positive = 0
    negative = 0
    for index, word in enumerate(words):
        if word in POSITIVE_CUES or word in NEGATIVE_CUES:
            is_positive = word in POSITIVE_CUES
            negated = any(
                previous in NEGATION_WORDS
                for previous in words[max(0, index - 3):index]
            )
            if negated:
                is_positive = not is_positive
            if is_positive:
                positive += 1
            else:
                negative += 1
    if positive == negative:
        return 0
    return 1 if positive > negative else -1


def _ranking_relation(claim: str, passage: str) -> int:
    """Compare explicit metric values for named models in an asserted best claim.

    Returns 1 when the named claimed model has the best reported value, -1 when
    the passage explicitly reports another model with a higher value, or 0 when
    the excerpt is too incomplete to compare.
    """
    claim_words = set(re.findall(r"[a-z]+", claim.casefold()))
    if not (RANKING_CLAIM_CUES & claim_words):
        return 0
    claimed_models = {
        model.casefold() for model in MODEL_PATTERN.findall(claim)
    }
    if not claimed_models:
        return 0

    model_values: list[tuple[str, str, float]] = []
    for sentence in SENTENCE_PATTERN.split(passage):
        for clause in re.split(r",\s*|\s+and\s+", sentence, flags=re.IGNORECASE):
            models = [model.casefold() for model in MODEL_PATTERN.findall(clause)]
            if not models:
                continue
            metric_value = METRIC_PATTERN.search(clause)
            if not metric_value:
                continue
            try:
                value = float(metric_value.group(2))
            except (IndexError, ValueError):
                continue
            metric = metric_value.group(1).casefold()
            model_values.extend((model, metric, value) for model in models)
    if not model_values:
        return 0

    claimed_metrics = {
        metric for model, metric, _ in model_values if model in claimed_models
    }
    if not claimed_metrics:
        return 0
    metric_relations: list[int] = []
    for metric in claimed_metrics:
        values_for_metric = [
            (model, value) for model, item_metric, value in model_values
            if item_metric == metric
        ]
        claimed_for_metric = [
            value for model, value in values_for_metric if model in claimed_models
        ]
        if not claimed_for_metric:
            continue
        best_for_metric = max(value for _, value in values_for_metric)
        best_models = {
            model for model, value in values_for_metric if value == best_for_metric
        }
        metric_relations.append(
            1 if best_models & claimed_models
            else -1 if best_for_metric > max(claimed_for_metric)
            else 0
        )
    if any(relation < 0 for relation in metric_relations):
        return -1
    if metric_relations and all(relation > 0 for relation in metric_relations):
        return 1
    return 0


def _explicit_metric_relation(claim: str, passage: str) -> int | None:
    """Compare an exact named-model metric claim without assuming study equivalence.

    Returns 1 for an exact match, 0 for a reported mismatch (which remains
    unclassified as a contradiction because the excerpts may describe different
    experimental settings), or None when this is not an explicit metric claim.
    """
    claim_models = {
        model.casefold() for model in MODEL_PATTERN.findall(claim)
    }
    claim_metric = METRIC_PATTERN.search(claim)
    if not claim_models or not claim_metric:
        return None
    claimed_value = float(claim_metric.group(2))
    claimed_name = claim_metric.group(1).casefold()
    reported: list[tuple[str, str, float]] = []
    for sentence in SENTENCE_PATTERN.split(passage):
        for clause in re.split(r",\s*|\s+and\s+", sentence, flags=re.IGNORECASE):
            metric_match = METRIC_PATTERN.search(clause)
            models = MODEL_PATTERN.findall(clause)
            if metric_match and models:
                reported.extend(
                    (model.casefold(), metric_match.group(1).casefold(),
                     float(metric_match.group(2)))
                    for model in models
                )
    matching = [
        value for model, metric, value in reported
        if model in claim_models and metric == claimed_name
    ]
    if not matching:
        return None
    if any(abs(claimed_value - value) < 1e-9 for value in matching):
        return 1
    return 0


def classify_evidence(claim: str, evidence: RetrievedEvidence) -> ClassifiedEvidence:
    """Classify evidence conservatively with topical overlap and explicit cues."""
    normalized_claim = claim.strip()
    if normalized_claim.endswith("?") or QUESTION_START.match(normalized_claim):
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.NEUTRAL,
            "This is a research question, not a factual assertion. The passage is "
            "retained as relevant retrieved evidence, but is not labeled as supporting "
            "or contradicting the question.",
        )

    claim_terms = meaningful_terms(claim)
    passage_terms = meaningful_terms(evidence.chunk.text)
    overlap = claim_terms & passage_terms
    if len(overlap) < 1:
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.NEUTRAL,
            "The passage does not share identifiable content terms with the claim.",
        )

    metric_relation = _explicit_metric_relation(claim, evidence.chunk.text)
    if metric_relation is not None:
        if metric_relation == 1:
            return ClassifiedEvidence(
                evidence,
                EvidenceLabel.SUPPORTING,
                "The passage reports the same named-model metric value as the "
                "factual claim. This is a preliminary source-text match.",
            )
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.NEUTRAL,
            "The passage reports a different value for the named model and metric. "
            "Without confirming comparable experimental settings, this difference "
            "is not labeled a contradiction.",
        )

    ranking_relation = _ranking_relation(claim, evidence.chunk.text)
    if ranking_relation:
        label = (
            EvidenceLabel.SUPPORTING
            if ranking_relation > 0
            else EvidenceLabel.CONTRADICTING
        )
        return ClassifiedEvidence(
            evidence,
            label,
            "Explicit model names and reported metric values "
            + (
                "match the claimed overall ranking."
                if ranking_relation > 0
                else "show another named model with a higher reported value; "
                "an individual-versus-ensemble comparison requires checking the "
                "source passage and metric."
            ),
        )

    context = extract_context(evidence.chunk.text)
    reported_findings = context["Reported findings"]
    if not reported_findings:
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.NEUTRAL,
            "No attributable result or measured finding was identified in this "
            "passage. Titles, author descriptions, and methodology alone cannot "
            "establish a factual conflict.",
        )

    finding_terms = meaningful_terms(reported_findings)
    finding_overlap = claim_terms & finding_terms
    if len(finding_overlap) < 2:
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.NEUTRAL,
            "The passage contains a reported result, but it does not match enough "
            "specific claim content to attribute the result to this claim.",
        )

    claim_direction = _direction(claim)
    finding_direction = _direction(reported_findings)
    if not claim_direction or not finding_direction:
        claim_asserted = bool(
            set(re.findall(r"[a-z]+", claim.lower())) & ASSERTION_CUES
        )
        if claim_asserted and len(finding_overlap) >= 2:
            return ClassifiedEvidence(
                evidence,
                EvidenceLabel.SUPPORTING,
                "The passage reports a result that shares specific content with "
                "this factual claim, with no explicit opposing cue. This is a "
                "preliminary match, not proof of entailment.",
            )
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.NEUTRAL,
            "Topic terms overlap, but explicit claim polarity or a sufficiently "
            "specific result match is absent.",
        )
    if claim_direction == finding_direction:
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.SUPPORTING,
            "Shared topic terms and matching rule-based directional cues; this is preliminary.",
        )
    if len(finding_overlap) >= 2:
        return ClassifiedEvidence(
            evidence,
            EvidenceLabel.CONTRADICTING,
            "The passage contains an attributable reported result on the same "
            "specific claim content, with an explicit opposing directional cue. "
            "This rule-based conflict is preliminary and requires source review.",
        )
    return ClassifiedEvidence(
        evidence,
        EvidenceLabel.NEUTRAL,
        "Opposing language is present, but a specific shared outcome was not "
        "identified; contradiction is not established.",
    )


def verify_claim(
    claim: str,
    chunks: list[EvidenceChunk],
    top_k: int = 15,
) -> list[ClassifiedEvidence]:
    """Retrieve across papers and classify the retrieved passages transparently."""
    retrieved = retrieve_evidence(claim, chunks, top_k=top_k)
    return [classify_evidence(claim, item) for item in retrieved]

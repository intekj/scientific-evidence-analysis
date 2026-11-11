"""Evidence-grounded, template-based synthesis with traceable citations."""

from __future__ import annotations

import re
from collections import defaultdict

from modules.claim_verification import ClassifiedEvidence, EvidenceLabel
from modules.context_extraction import extract_context
from modules.text_chunker import EvidenceChunk

ENSEMBLE_CUES = re.compile(
    r"\b(?:ensemble|voting|stacking|bagging|boosting|combined model)\b",
    re.IGNORECASE,
)
MODEL_NAME_CUES = re.compile(
    r"\b(?:CFM|XGBoost|LightGBM|CatBoost|Random Forest|Decision Tree|"
    r"GLM|DNN|SVM|CNN|LSTM|RF|DT|model|algorithm)\b",
    re.IGNORECASE,
)
NON_CONTEXT_FIELDS = {"Reported findings", "Metric values"}


def _citation(chunk: EvidenceChunk) -> str:
    return f"[{chunk.paper_name}, p. {chunk.page_number}]"


def _is_question(prompt: str) -> bool:
    normalized = prompt.strip()
    return normalized.endswith("?") or bool(
        re.match(
            r"^(?:which|what|who|whom|whose|when|where|why|how|"
            r"does|do|did|is|are|was|were|can|could|would|should|will|has|have)\b",
            normalized,
            re.IGNORECASE,
        )
    )


def _paper_order(
    items: list[ClassifiedEvidence],
    all_chunks: list[EvidenceChunk],
) -> list[str]:
    names: dict[str, str] = {}
    for chunk in [*all_chunks, *(item.evidence.chunk for item in items)]:
        names.setdefault(chunk.paper_id, chunk.paper_name)
    return list(names)


def _unique_chunks(items: list[ClassifiedEvidence]) -> list[EvidenceChunk]:
    unique: list[EvidenceChunk] = []
    seen: set[tuple[str, int, str]] = set()
    for item in items:
        chunk = item.evidence.chunk
        key = (chunk.paper_id, chunk.page_number, " ".join(chunk.text.casefold().split()))
        if key not in seen:
            seen.add(key)
            unique.append(chunk)
    return unique


def _finding_text(chunk: EvidenceChunk) -> str | None:
    findings = extract_context(chunk.text).get("Reported findings")
    if not findings:
        return None
    return findings


def _collect_paper_context(
    paper_id: str,
    items: list[ClassifiedEvidence],
    all_chunks: list[EvidenceChunk],
) -> dict[str, list[tuple[str, EvidenceChunk]]]:
    source_chunks = [
        chunk for chunk in all_chunks if chunk.paper_id == paper_id
    ]
    if not source_chunks:
        source_chunks = [
            item.evidence.chunk for item in items
            if item.evidence.chunk.paper_id == paper_id
        ]
    fields: dict[str, list[tuple[str, EvidenceChunk]]] = defaultdict(list)
    seen: set[tuple[str, str, int]] = set()
    for chunk in source_chunks:
        context = extract_context(chunk.text)
        for field, value in context.items():
            if not value or field in NON_CONTEXT_FIELDS:
                continue
            key = (field, value, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)
            fields[field].append((value, chunk))
    return fields


def _render_context_comparison(
    lines: list[str],
    paper_ids: list[str],
    items: list[ClassifiedEvidence],
    all_chunks: list[EvidenceChunk],
) -> None:
    if not paper_ids:
        lines.append("No papers or passages were available for context comparison.")
        return
    contexts = {
        paper_id: _collect_paper_context(paper_id, items, all_chunks)
        for paper_id in paper_ids
    }
    context_fields = list(
        dict.fromkeys(field for fields in contexts.values() for field in fields)
    )
    found = False
    for field in context_fields:
        entries = [
            (paper_id, value, chunk)
            for paper_id in paper_ids
            for value, chunk in contexts[paper_id].get(field, [])
        ]
        if not entries:
            continue
        found = True
        lines.append(f"- **{field}:**")
        for paper_id, value, chunk in entries:
            lines.append(
                f"  - {chunk.paper_name}: {value} {_citation(chunk)}"
            )
    if not found:
        lines.append(
            "No study context fields could be extracted from the uploaded pages."
        )


def _differing_context_summary(
    paper_ids: list[str],
    items: list[ClassifiedEvidence],
    all_chunks: list[EvidenceChunk],
) -> list[str]:
    contexts = {
        paper_id: _collect_paper_context(paper_id, items, all_chunks)
        for paper_id in paper_ids
    }
    if len(paper_ids) < 2:
        return []
    fields = list(dict.fromkeys(
        field for paper_context in contexts.values() for field in paper_context
    ))
    differences: list[str] = []
    for field in fields:
        per_paper = [
            (paper_id, contexts[paper_id].get(field, []))
            for paper_id in paper_ids
        ]
        populated = [(paper_id, entries) for paper_id, entries in per_paper if entries]
        if len(populated) < 2:
            continue
        signatures = {
            re.sub(r"\W+", " ", " ".join(value for value, _ in entries).casefold()).strip()
            for _, entries in populated
        }
        if len(signatures) > 1:
            detail = "; ".join(
                f"{entries[0][1].paper_name}: {entries[0][0]} {_citation(entries[0][1])}"
                for _, entries in populated
            )
            differences.append(f"- **{field} differs across studies:** {detail}")
    return differences


def _append_question_findings(
    lines: list[str],
    paper_id: str,
    items: list[ClassifiedEvidence],
    *,
    ensemble: bool | None,
) -> int:
    count = 0
    seen: set[tuple[int, str]] = set()
    for chunk in _unique_chunks(items):
        if chunk.paper_id != paper_id:
            continue
        finding = _finding_text(chunk)
        if not finding:
            continue
        is_ensemble = bool(ENSEMBLE_CUES.search(finding))
        if ensemble is not None and is_ensemble != ensemble:
            continue
        key = (chunk.page_number, finding)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {finding} {_citation(chunk)}")
        count += 1
    if not count:
        lines.append("No relevant reported finding was identified in retrieved passages.")
    return count


def _question_synthesis(
    question: str,
    items: list[ClassifiedEvidence],
    all_chunks: list[EvidenceChunk],
) -> str:
    paper_ids = _paper_order(items, all_chunks)
    lines = [
        "# Evidence Synthesis",
        "",
        "## 1. Research Question",
        question.strip() or "Not provided.",
        "",
        "> This is a question-oriented synthesis of retrieved source passages. "
        "It does not assign support/contradiction labels to a question.",
        "",
    ]
    for index in range(2):
        paper_id = paper_ids[index] if index < len(paper_ids) else None
        name = next(
            (
                chunk.paper_name for chunk in [*all_chunks]
                if paper_id is not None and chunk.paper_id == paper_id
            ),
            next(
                (
                    item.evidence.chunk.paper_name for item in items
                    if paper_id is not None and item.evidence.chunk.paper_id == paper_id
                ),
                f"Paper {index + 1}",
            ),
        )
        lines.extend([
            f"## {index + 2}. Relevant Findings from Paper {index + 1} ({name})",
        ])
        if paper_id is None:
            lines.append("No relevant passages from this paper were retrieved.")
        else:
            _append_question_findings(lines, paper_id, items, ensemble=None)
        lines.append("")

    lines.extend(["## 4. Individual Model Performance"])
    individual_count = 0
    for paper_id in paper_ids:
        for chunk in _unique_chunks(items):
            if chunk.paper_id != paper_id:
                continue
            finding = _finding_text(chunk)
            if not finding or ENSEMBLE_CUES.search(finding):
                continue
            if not MODEL_NAME_CUES.search(finding):
                continue
            lines.append(f"- {finding} {_citation(chunk)}")
            individual_count += 1
    if not individual_count:
        lines.append("No individual-model performance finding was identified.")

    lines.extend(["", "## 5. Ensemble Model Performance"])
    ensemble_count = 0
    for paper_id in paper_ids:
        ensemble_count += _append_question_findings(
            lines, paper_id, items, ensemble=True
        )
    if not paper_ids:
        lines.append("No ensemble performance finding was identified.")

    lines.extend(["", "## 6. Study Context Comparison"])
    _render_context_comparison(lines, paper_ids, items, all_chunks)

    lines.extend(["", "## 7. Cross-Study Interpretation"])
    if len(paper_ids) < 2:
        lines.append(
            "Evidence from fewer than two papers was retrieved, so cross-study "
            "comparability cannot be assessed."
        )
    else:
        differences = _differing_context_summary(paper_ids, items, all_chunks)
        if differences:
            lines.extend(differences)
            lines.append(
                "The cited differences indicate these study-specific model results "
                "are not directly comparable."
            )
        lines.append(
            "The passages report study-specific model performance. Rankings and "
            "metric values should not be pooled as a universal model ranking unless "
            "datasets, geographic settings, input variables, train/validation "
            "procedures, and metric definitions are comparable."
        )

    lines.extend(["", "## 8. Overall Answer"])
    if individual_count or ensemble_count:
        cited_refs = list(dict.fromkeys(
            _citation(item.evidence.chunk)
            for item in items
            if _finding_text(item.evidence.chunk)
            and MODEL_NAME_CUES.search(_finding_text(item.evidence.chunk) or "")
        ))
        source_note = f" Sources: {', '.join(cited_refs)}." if cited_refs else ""
        lines.append(
            "The retrieved findings identify study-specific top performers and "
            "reported ensemble results, but do not establish one universally best "
            f"algorithm across the studies.{source_note}"
        )
        lines.append(
            "Where reported, individual algorithms and ensemble models are listed "
            "separately above; their performance should be compared only within "
            "matching evaluation settings."
        )
        cited_findings = [
            f"{_finding_text(item.evidence.chunk)} {_citation(item.evidence.chunk)}"
            for item in items
            if _finding_text(item.evidence.chunk)
            and MODEL_NAME_CUES.search(_finding_text(item.evidence.chunk) or "")
        ]
        if cited_findings:
            lines.append(
                "Retrieved model findings: " + " ".join(dict.fromkeys(cited_findings))
            )
    else:
        lines.append(
            "The retrieved passages did not contain enough attributable model "
            "performance findings to answer the question."
        )
    lines.extend([
        "",
        "## 9. Limitations and Uncertainty",
        "- Findings are extracted from retrieved passages; unreturned passages may "
        "contain relevant results.",
        "- Context is aggregated from uploaded page chunks, using transparent "
        "pattern-based extraction and source-page citations.",
        "- Similarity indicates topical relevance, not scientific truth or entailment.",
        "- Model names and reported metric values may use different datasets, "
        "validation procedures, and evaluation settings.",
        "- Image-based figures and scanned pages are not interpreted.",
    ])
    return "\n".join(lines)


def generate_synthesis(
    claim: str,
    items: list[ClassifiedEvidence],
    all_chunks: list[EvidenceChunk],
) -> str:
    """Create a question-oriented findings synthesis or factual-claim summary."""
    if _is_question(claim):
        return _question_synthesis(claim, items, all_chunks)
    return _claim_synthesis(claim, items, all_chunks)


def _claim_synthesis(
    claim: str,
    items: list[ClassifiedEvidence],
    all_chunks: list[EvidenceChunk],
) -> str:
    """Preserve the factual claim verification report for assertion inputs."""
    groups: dict[EvidenceLabel, list[ClassifiedEvidence]] = defaultdict(list)
    for item in items:
        groups[item.label].append(item)

    paper_ids = _paper_order(items, all_chunks)
    lines = [
        "# Evidence Synthesis",
        "",
        "## 1. Research Question",
        claim.strip() or "Not provided.",
        "",
    ]
    for index in range(2):
        paper_id = paper_ids[index] if index < len(paper_ids) else None
        paper_name = next(
            (
                chunk.paper_name for chunk in all_chunks
                if paper_id is not None and chunk.paper_id == paper_id
            ),
            f"Paper {index + 1}",
        )
        lines.extend([
            "",
            f"## {index + 2}. Relevant Findings from Paper {index + 1} ({paper_name})",
        ])
        paper_items = [
            item for item in items
            if paper_id is not None and item.evidence.chunk.paper_id == paper_id
        ]
        if not paper_items:
            lines.append("No relevant retrieved passages from this paper.")
        else:
            for label in EvidenceLabel:
                lines.extend(["", f"### {label.value}"])
                _append_evidence(
                    lines, [item for item in paper_items if item.label == label]
                )
    lines.extend(["", "## 4. Individual Model Performance"])
    _append_performance_lines(lines, items, ensemble=False)
    lines.extend(["", "## 5. Ensemble Model Performance"])
    _append_performance_lines(lines, items, ensemble=True)
    lines.extend(["", "## 6. Study Context Comparison"])
    _render_context_comparison(
        lines, _paper_order(items, all_chunks), items, all_chunks
    )
    lines.extend([
        "",
        "## 7. Cross-Study Interpretation",
        "Interpret retrieved findings in the context of study design and measurement.",
        "",
        "## 8. Overall Answer",
        (
            "no passages were retrieved; the claim cannot be assessed."
            if not items
            else
            "Review the cited evidence groups; classifications are preliminary and "
            "do not establish truth. Supporting or contradicting cues do not prove "
            "or disprove the claim; retrieved evidence does not prove the claim."
        ),
        "",
        "## 9. Limitations and Uncertainty",
        "Rule-based classification and pattern-extracted context require human review.",
    ])
    return "\n".join(lines)


def _append_performance_lines(
    lines: list[str],
    items: list[ClassifiedEvidence],
    *,
    ensemble: bool,
) -> None:
    seen: set[tuple[str, int, str]] = set()
    count = 0
    for item in items:
        chunk = item.evidence.chunk
        finding = _finding_text(chunk)
        if not finding or bool(ENSEMBLE_CUES.search(finding)) != ensemble:
            continue
        if not MODEL_NAME_CUES.search(finding):
            continue
        key = (chunk.paper_id, chunk.page_number, finding)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {finding} {_citation(chunk)}")
        count += 1
    if not count:
        lines.append("No sourced model-performance finding was identified.")


def _append_evidence(lines: list[str], items: list[ClassifiedEvidence]) -> None:
    if not items:
        lines.append("No relevant evidence was found in the retrieved passages.")
        return
    for item in items:
        chunk = item.evidence.chunk
        lines.append(
            f"- {chunk.text} {_citation(chunk)} "
            f"(similarity: {item.evidence.score:.3f}; {item.reason})"
        )

"""Validate, summarize, and show compact samples from the local QASPER dataset."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import DatasetDict, load_from_disk
import pyarrow

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "datasets" / "benchmarks" / "qasper" / "data"
PREPARED_DIR = DATA_DIR / "prepared"
REQUIRED_FIELDS = ("id", "title", "abstract", "full_text", "qas")


def validate_record(record: dict[str, Any]) -> list[str]:
    """Return missing required top-level keys without rejecting optional values."""
    return [field for field in REQUIRED_FIELDS if field not in record]


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _flatten_evidence(value: Any) -> list[str]:
    evidence: list[str] = []
    if isinstance(value, str):
        if value.strip():
            evidence.append(value)
    elif isinstance(value, dict):
        for nested in value.values():
            evidence.extend(_flatten_evidence(nested))
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, str)):
        for nested in value:
            evidence.extend(_flatten_evidence(nested))
    return evidence


def _answer_annotations(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        nested = value.get("answer")
        if nested is not None:
            return _answer_annotations(nested)
        return [value]
    if isinstance(value, list):
        return [
            annotation
            for entry in value
            for annotation in _answer_annotations(entry)
        ]
    return []


def extract_questions(record: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract questions and aligned annotations, tolerating absent nested fields."""
    qas = record.get("qas")
    if not isinstance(qas, dict):
        return []
    questions = _as_list(qas.get("question"))
    question_ids = _as_list(qas.get("question_id"))
    answer_groups = _as_list(qas.get("answers"))
    extracted: list[dict[str, Any]] = []
    for index, question in enumerate(questions):
        answer_group = answer_groups[index] if index < len(answer_groups) else None
        annotations = _answer_annotations(answer_group)
        extracted.append(
            {
                "question_id": (
                    question_ids[index]
                    if index < len(question_ids)
                    else None
                ),
                "question": question,
                "answer_annotations": annotations,
                "evidence_passages": [
                    evidence
                    for annotation in annotations
                    for evidence in _flatten_evidence(annotation.get("evidence"))
                ],
                "answerability": _question_answerability(annotations),
            }
        )
    return extracted


def _question_answerability(annotations: list[dict[str, Any]]) -> str:
    statuses = {
        annotation.get("unanswerable")
        for annotation in annotations
        if isinstance(annotation.get("unanswerable"), bool)
    }
    if statuses == {False}:
        return "answerable"
    if statuses == {True}:
        return "unanswerable"
    if statuses == {False, True}:
        return "mixed"
    return "not specified"


def inspect_split(records: Any, sample_limit: int = 5) -> dict[str, Any]:
    """Count papers/questions/answerability and capture compact actual samples."""
    paper_ids: set[str] = set()
    question_count = 0
    answerable = 0
    unanswerable = 0
    mixed = 0
    unknown_answerability = 0
    missing_field_records = 0
    samples: list[dict[str, Any]] = []
    for record in records:
        if validate_record(record):
            missing_field_records += 1
        paper_id = record.get("id")
        if isinstance(paper_id, str) and paper_id:
            paper_ids.add(paper_id)
        if len(samples) < sample_limit:
            samples.append(record)
        for question in extract_questions(record):
            question_count += 1
            status = question["answerability"]
            if status == "answerable":
                answerable += 1
            elif status == "unanswerable":
                unanswerable += 1
            elif status == "mixed":
                mixed += 1
            else:
                unknown_answerability += 1

    return {
        "record_count": len(records),
        "unique_papers": len(paper_ids),
        "question_count": question_count,
        "answerable_questions": answerable,
        "unanswerable_questions": unanswerable,
        "mixed_answerability_questions": mixed,
        "unspecified_answerability_questions": unknown_answerability,
        "records_missing_required_fields": missing_field_records,
        "samples": samples,
    }


def load_local_dataset(path: str | Path = PREPARED_DIR) -> DatasetDict:
    """Load a locally prepared QASPER DatasetDict; never accesses the network."""
    dataset_path = Path(path)
    if not dataset_path.is_dir():
        raise FileNotFoundError(
            f"Prepared QASPER data not found at {dataset_path}. "
            "Run python download_qasper.py first."
        )
    dataset = load_from_disk(str(dataset_path))
    if not isinstance(dataset, DatasetDict):
        raise ValueError(f"Expected a DatasetDict at {dataset_path}.")
    return dataset


def _sample_observation(record: dict[str, Any]) -> dict[str, Any]:
    questions = extract_questions(record)
    full_text = record.get("full_text")
    if not isinstance(full_text, dict):
        full_text = {}
    section_names = full_text.get("section_name") or []
    paragraphs = full_text.get("paragraphs") or []
    first_question = questions[0] if questions else None
    first_annotation = (
        first_question["answer_annotations"][0]
        if first_question and first_question["answer_annotations"]
        else None
    )
    compact_annotation = None
    if first_annotation:
        compact_annotation = {
            "unanswerable": first_annotation.get("unanswerable"),
            "yes_no": first_annotation.get("yes_no"),
            "free_form_answer": str(
                first_annotation.get("free_form_answer") or ""
            )[:300],
            "extractive_spans": [
                str(span)[:200]
                for span in _as_list(first_annotation.get("extractive_spans"))[:3]
            ],
        }
    return {
        "paper_id": record.get("id"),
        "title": record.get("title"),
        "abstract_present": bool(record.get("abstract")),
        "section_names": section_names[:5],
        "section_count": len(section_names),
        "paragraph_count": sum(
            len(section) for section in paragraphs if isinstance(section, list)
        ),
        "question_id": first_question.get("question_id") if first_question else None,
        "research_question": first_question.get("question") if first_question else None,
        "answerability": first_question.get("answerability") if first_question else None,
        "answer_annotation": compact_annotation,
        "evidence_passages": (
            [
                passage[:300]
                for passage in first_question.get("evidence_passages", [])[:2]
            ]
            if first_question
            else []
        ),
    }


def _describe_fields(dataset: DatasetDict) -> dict[str, Any]:
    if not dataset:
        return {}
    features = dataset[next(iter(dataset))]
    description: dict[str, Any] = {}
    for name in features.column_names:
        feature = features.features[name]
        description[name] = str(feature)
    return description


def main() -> int:
    try:
        dataset = load_local_dataset()
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        RuntimeError,
        pyarrow.ArrowException,
    ) as exc:
        print(f"Could not inspect local QASPER data: {exc}")
        return 1

    print(f"Local QASPER data: {DATA_DIR}")
    print(f"Reloadable dataset: {PREPARED_DIR}")
    print("Available fields and Hugging Face feature schemas:")
    print(json.dumps(_describe_fields(dataset), ensure_ascii=False, indent=2))

    total_records = 0
    total_papers: set[str] = set()
    total_questions = 0
    total_answerable = 0
    total_unanswerable = 0
    total_mixed = 0
    total_unspecified = 0
    missing_records = 0
    for split, records in dataset.items():
        summary = inspect_split(records)
        total_records += summary["record_count"]
        total_questions += summary["question_count"]
        total_answerable += summary["answerable_questions"]
        total_unanswerable += summary["unanswerable_questions"]
        total_mixed += summary["mixed_answerability_questions"]
        total_unspecified += summary["unspecified_answerability_questions"]
        missing_records += summary["records_missing_required_fields"]
        total_papers.update(
            record["id"] for record in records if record.get("id")
        )
        print(
            f"\n{split}: {summary['record_count']:,} records; "
            f"{summary['unique_papers']:,} unique papers; "
            f"{summary['question_count']:,} questions; "
            f"answerable={summary['answerable_questions']:,}, "
            f"unanswerable={summary['unanswerable_questions']:,}, "
            f"mixed={summary['mixed_answerability_questions']:,}, "
            f"unspecified={summary['unspecified_answerability_questions']:,}"
        )
        if summary["records_missing_required_fields"]:
            print(
                "  WARNING: "
                f"{summary['records_missing_required_fields']} records lack "
                "one or more expected top-level fields."
            )
        print("  Five sample observations (or fewer if the split is smaller):")
        for index, record in enumerate(summary["samples"], start=1):
            print(
                f"  Sample {index}: "
                + json.dumps(
                    _sample_observation(record),
                    ensure_ascii=False,
                    indent=2,
                )
            )

    print("\nOverall totals")
    print(f"Records: {total_records:,}")
    print(f"Unique papers: {len(total_papers):,}")
    print(f"Questions: {total_questions:,}")
    print(
        "Question answerability: "
        f"answerable={total_answerable:,}, "
        f"unanswerable={total_unanswerable:,}, "
        f"mixed={total_mixed:,}, "
        f"unspecified={total_unspecified:,}"
    )
    print(f"Records missing expected top-level fields: {missing_records:,}")
    print(f"Saved dataset location: {DATA_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

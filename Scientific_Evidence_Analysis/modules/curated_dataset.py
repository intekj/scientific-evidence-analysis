"""Safe loading and separate retrieval for the curated research-claim dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Sequence

import pandas as pd

from modules.embedding_engine import EmbeddingEngine
from modules.pdf_processor import PDFExtractionError, ProcessedPaper, process_pdf

if TYPE_CHECKING:
    from modules.paper_storage import LocalPaperStore

DATASET_ROOT = Path(__file__).resolve().parents[2] / "AI Papers Dataset"
MASTER_CLAIMS_PATH = DATASET_ROOT / "Master Dataset" / "claims.csv"
CANONICAL_PAPER_IDS = tuple(f"P{number:03d}" for number in range(1, 22))
CANONICAL_PAPER_FILENAMES = {
    "P001": "2023.emnlp-main.397.pdf",
    "P002": "2109.07958v2.pdf",
    "P003": "2305.14251v2.pdf",
    "P004": "2310.00741v2.pdf",
    "P005": "2024.acl-long.586.pdf",
    "P006": "2025.acl-long.1587.pdf",
    "P007": "2025.findings-emnlp.1035.pdf",
    "P008": "2025.ijcnlp-long.127.pdf",
    "P009": "2026.eacl-long.8.pdf",
    "P010": "2026.emnlp-main.610.pdf",
    "P011": "2026.groundlm-1.81.pdf",
    "P012": "2403.18802v4.pdf",
    "P013": "2405.07437v2.pdf",
    "P014": "s41586-025-10072-4.pdf",
    "P015": "06510-AAAI24.ChenJ.pdf",
    "P016": "2024.naacl-long.20.pdf",
    "P017": "2026.findings-aacl.151.pdf",
    "P018": "2309.15217v2.pdf",
    "P019": "2310.11511v1.pdf",
    "P020": "2021.naacl-main.365.pdf",
    "P021": "ooae021.pdf",
}

REQUIRED_CLAIM_COLUMNS = {
    "claim_id",
    "paper_id",
    "paper_title",
    "claim_text",
    "source_filename",
    "evidence_status",
    "annotation_status",
    "validation_decision",
    "verified_evidence_quote_or_table",
    "evidence_pdf_page",
    "verified_evidence_section",
    "atomicity_review_status",
    "atomicity_review_decision",
    "atomicity_review_rationale",
}

OPTIONAL_ANALYSIS_SCHEMAS = {
    "Cross-paper comparison": (
        "Batch12_Cross_Paper_Comparison.csv",
        {
            "comparison_id", "claim_ids", "question", "relationship",
            "classification", "interpretation",
        },
    ),
    "Disagreement analysis": (
        "Batch12B_Disagreement_Analysis.csv",
        {
            "test_id", "research_question", "claim_ids", "classification",
            "direct_contradiction", "safe_synthesis", "review_status",
        },
    ),
    "Uncertainty register": (
        "Batch12C_Uncertainty_Register.csv",
        {
            "uncertainty_id", "topic", "papers_involved", "uncertainty_source",
            "uncertainty_level", "safe_interpretation",
        },
    ),
}


class CuratedDatasetError(ValueError):
    """Raised when the required curated claims file is missing or invalid."""


@dataclass(frozen=True)
class CanonicalPDFImportResult:
    """Outcome of adding mapped source PDFs to the ordinary full-text store."""

    discovered_count: int
    indexed_count: int
    papers: Mapping[str, ProcessedPaper]
    warnings: tuple[str, ...]


VALIDATION_STATUSES = (
    "RETAINED",
    "CORRECTED",
    "MERGED",
    "REJECTED",
    "CANDIDATE / NEEDS REVIEW",
    "CONTEXT ONLY",
    "UNKNOWN / NEEDS REVIEW",
)


def validate_canonical_pdf_mapping(paper_id: str, filename: str) -> None:
    """Reject filenames that do not match the established canonical mapping."""
    expected = CANONICAL_PAPER_FILENAMES.get(paper_id)
    if expected is None or filename != expected:
        raise CuratedDatasetError(
            f"Invalid canonical PDF mapping for {paper_id}: expected "
            f"{expected!r}, found {filename!r}."
        )


def discover_canonical_pdfs(
    dataset_root: str | Path = DATASET_ROOT,
) -> tuple[dict[str, Path], tuple[str, ...]]:
    """Find only mapped canonical PDFs below the existing papers directory."""
    papers_root = Path(dataset_root).expanduser().resolve() / "papers"
    warnings: list[str] = []
    discovered: dict[str, Path] = {}
    if not papers_root.is_dir():
        return {}, (f"Canonical PDF directory was not found: {papers_root}",)

    for paper_id, filename in CANONICAL_PAPER_FILENAMES.items():
        matches = sorted(
            path for path in papers_root.rglob(filename) if path.is_file()
        )
        if not matches:
            warnings.append(
                f"Canonical PDF {paper_id} ({filename}) is missing under {papers_root}."
            )
        elif len(matches) > 1:
            warnings.append(
                f"Canonical PDF {paper_id} ({filename}) is ambiguous; found "
                f"{len(matches)} matching files."
            )
        else:
            discovered[paper_id] = matches[0]
    return discovered, tuple(warnings)


def import_canonical_pdfs(
    store: LocalPaperStore,
    *,
    dataset_root: str | Path = DATASET_ROOT,
    restored_papers: Mapping[str, ProcessedPaper] | None = None,
) -> CanonicalPDFImportResult:
    """Import mapped PDFs through LocalPaperStore and ordinary PDF extraction."""
    from modules.paper_storage import LocalPaperStore

    if not isinstance(store, LocalPaperStore):
        raise TypeError("store must be a LocalPaperStore.")
    restored = restored_papers or {}
    discovered, discovery_warnings = discover_canonical_pdfs(dataset_root)
    warnings = list(discovery_warnings)
    processed: dict[str, ProcessedPaper] = {}
    records = store.list_records()
    indexed_ids = {
        record.canonical_paper_id
        for record in records
        if record.canonical_paper_id is not None
        and record.processing_status == "ready"
        and record.paper_id in restored
    }

    for paper_id, source_path in discovered.items():
        filename = CANONICAL_PAPER_FILENAMES[paper_id]
        try:
            validate_canonical_pdf_mapping(paper_id, source_path.name)
            existing = next(
                (
                    record
                    for record in records
                    if record.canonical_paper_id == paper_id
                ),
                None,
            )
            if (
                existing is not None
                and existing.processing_status == "ready"
                and existing.paper_id in restored
            ):
                indexed_ids.add(paper_id)
                continue

            file_data = source_path.read_bytes()
            paper = process_pdf(
                file_data,
                filename,
                canonical_paper_id=paper_id,
            )
            record, _ = store.save_pdf(
                file_data,
                filename,
                canonical_paper_id=paper_id,
                canonical_filename=filename,
            )
            store.update_status(
                record.paper_id,
                "ready",
                page_count=paper.page_count,
            )
            processed[record.paper_id] = paper
            records = store.list_records()
            indexed_ids.add(paper_id)
        except (OSError, PDFExtractionError, CuratedDatasetError, ValueError) as exc:
            warnings.append(
                f"Could not index canonical PDF {paper_id} ({filename}): {exc}"
            )

    return CanonicalPDFImportResult(
        discovered_count=len(discovered),
        indexed_count=len(indexed_ids),
        papers=processed,
        warnings=tuple(warnings),
    )


@dataclass(frozen=True)
class CuratedDataset:
    """Validated original records with additive normalized status columns."""

    records: pd.DataFrame
    paper_files: Mapping[str, str]
    path: Path


@dataclass(frozen=True)
class CuratedClaimResult:
    """A curated claim hit; its score is not a PDF-passage score."""

    claim_id: str
    paper_id: str
    paper_title: str
    source_filename: str
    claim_text: str
    evidence_quote: str
    evidence_text: str
    evidence_page: str
    evidence_section: str
    evidence_status: str
    annotation_status: str
    validation_decision: str
    review_rationale: str
    normalized_validation_status: str
    validation_status_reason: str
    similarity: float
    original_record: Mapping[str, str]


def _text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def normalize_validation_status(
    validation_decision: object,
    annotation_status: object,
    evidence_status: object = "",
    claim_type: object = "",
    atomicity_review_decision: object = "",
    atomicity_review_status: object = "",
) -> tuple[str, str]:
    """Normalize explicit review fields; evidence-quote presence is not approval.

    Precedence is rejection, merge/duplicate, context-only, correction, approved
    retention with a review signal, explicit candidate/pending, then unknown.
    """
    decision = _text(validation_decision).casefold()
    annotation = _text(annotation_status).casefold()
    evidence = _text(evidence_status).casefold()
    claim_kind = _text(claim_type).casefold()
    atomic_decision = _text(atomicity_review_decision).casefold()
    atomic_status = _text(atomicity_review_status).casefold()
    review_text = " ".join((decision, annotation, evidence, atomic_decision, atomic_status))

    if re.search(r"\breject(?:ed|ion)?\b", decision):
        return "REJECTED", "Validation decision explicitly rejects this record."
    if re.search(r"\b(?:merge|merged|duplicate|duplicated)\b", decision):
        return "MERGED", "Validation decision marks this record as merged or duplicate."
    if (
        re.search(r"\bcontext only\b", review_text)
        or (
            "context" in claim_kind
            and "not an independent claim" in review_text
        )
        or re.search(r"\bverified as study context\b", review_text)
    ):
        return "CONTEXT ONLY", "Review identifies this record as context, not an independent claim."
    if re.search(r"\b(?:correct|corrected|correction|replacement|replace)\b", decision):
        return "CORRECTED", "Validation decision indicates wording, evidence, or scope was corrected."

    pending_review_pattern = re.compile(
        r"\b(?:"
        r"needs?\s+review|pending(?:\s+review)?|unreviewed|"
        r"not\s+(?:yet\s+)?reviewed|review\s+not\s+completed"
        r")\b"
    )
    pending_review = bool(
        pending_review_pattern.search(decision)
        or pending_review_pattern.search(annotation)
    )
    if pending_review:
        return "CANDIDATE / NEEDS REVIEW", "The review fields explicitly indicate that review is pending."

    approved_decision = bool(
        re.search(r"\b(?:retain|retained|supported|verified|keep|accepted|validated)\b", decision)
    )
    reviewed_annotation = bool(
        re.search(
            r"\b(?:reviewed|review|validated|validation|adjudicated|source check|"
            r"full-text review|human review)\b",
            annotation,
        )
    )
    if approved_decision and reviewed_annotation:
        return "RETAINED", "An affirmative validation decision and review/verification status are present."

    if re.search(r"\b(?:candidate|needs? review|pending|unreviewed)\b", decision):
        return "CANDIDATE / NEEDS REVIEW", "The validation decision marks this record as pending review."
    if re.search(r"\bcandidate\b", annotation):
        return "CANDIDATE / NEEDS REVIEW", "The annotation status marks this record as pending review."
    if approved_decision:
        return "CANDIDATE / NEEDS REVIEW", (
            "An affirmative decision is present, but no review or verification status confirms it."
        )
    return "UNKNOWN / NEEDS REVIEW", "No explicit validation category could be safely inferred."


def _read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        raise CuratedDatasetError(f"{label} file was not found: {path}")
    try:
        return pd.read_csv(path, dtype="string", keep_default_na=False, encoding="utf-8-sig")
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
    ) as exc:
        raise CuratedDatasetError(f"Could not read {label} file {path}: {exc}") from exc


def load_curated_dataset(
    path: str | Path = MASTER_CLAIMS_PATH,
) -> CuratedDataset:
    """Load claims.csv without changing it and validate claim and paper identity."""
    claims_path = Path(path).expanduser().resolve()
    records = _read_csv(claims_path, "Master claims dataset")
    duplicate_columns = records.columns[records.columns.duplicated()].unique()
    if len(duplicate_columns):
        raise CuratedDatasetError(
            "Master claims dataset has duplicate columns: "
            + ", ".join(map(str, duplicate_columns))
        )
    missing_columns = sorted(REQUIRED_CLAIM_COLUMNS - set(records.columns))
    if missing_columns:
        raise CuratedDatasetError(
            "Master claims dataset is missing required columns: "
            + ", ".join(missing_columns)
        )

    for column in ("claim_id", "paper_id", "paper_title", "claim_text", "source_filename"):
        invalid_rows = records[column].isna() | records[column].str.strip().eq("")
        if invalid_rows.any():
            lines = (records.index[invalid_rows] + 2).tolist()[:10]
            raise CuratedDatasetError(
                f"Master claims dataset has missing {column} values at CSV row(s): {lines}"
            )

    duplicate_ids = records.loc[
        records["claim_id"].duplicated(keep=False), "claim_id"
    ].dropna().unique()
    if len(duplicate_ids):
        raise CuratedDatasetError(
            "Duplicate claim_id values found: " + ", ".join(map(str, duplicate_ids[:10]))
        )

    invalid_papers = sorted(
        paper_id
        for paper_id in records["paper_id"].dropna().astype(str).unique()
        if not re.fullmatch(r"P\d{3}", paper_id)
        or paper_id not in CANONICAL_PAPER_IDS
    )
    if invalid_papers:
        raise CuratedDatasetError(
            "Noncanonical paper_id values found: " + ", ".join(invalid_papers[:10])
        )
    found_papers = set(records["paper_id"].astype(str))
    missing_papers = sorted(set(CANONICAL_PAPER_IDS) - found_papers)
    if missing_papers:
        raise CuratedDatasetError(
            "Master claims dataset is missing canonical paper IDs: "
            + ", ".join(missing_papers)
        )

    paper_files: dict[str, str] = {}
    for paper_id, filename in records[["paper_id", "source_filename"]].itertuples(
        index=False, name=None
    ):
        existing = paper_files.setdefault(str(paper_id), str(filename).strip())
        if existing != str(filename).strip():
            raise CuratedDatasetError(
                f"Canonical paper {paper_id} maps to multiple source filenames."
            )
    incorrect_mappings = [
        f"{paper_id}: expected {expected!r}, found {paper_files[paper_id]!r}"
        for paper_id, expected in CANONICAL_PAPER_FILENAMES.items()
        if paper_files.get(paper_id) != expected
    ]
    if incorrect_mappings:
        raise CuratedDatasetError(
            "Canonical paper filename mapping does not match the established "
            "P001–P021 mapping: " + "; ".join(incorrect_mappings)
        )
    duplicate_files = pd.Series(list(paper_files.values())).duplicated(keep=False)
    if duplicate_files.any():
        raise CuratedDatasetError(
            "A source_filename is mapped to multiple canonical paper IDs."
        )

    statuses = records.apply(
        lambda row: normalize_validation_status(
            row["validation_decision"],
            row["annotation_status"],
            row["evidence_status"],
            row.get("claim_type", ""),
            row["atomicity_review_decision"],
            row["atomicity_review_status"],
        ),
        axis=1,
        result_type="expand",
    )
    statuses.columns = ["normalized_validation_status", "validation_status_reason"]
    records = records.copy()
    records[statuses.columns] = statuses
    return CuratedDataset(records=records, paper_files=paper_files, path=claims_path)


def retrieve_curated_claims(
    query: str,
    dataset: CuratedDataset,
    engine: EmbeddingEngine,
    top_k: int = 10,
) -> list[CuratedClaimResult]:
    """Search claim text and verified evidence using a dedicated embedding index."""
    if not query.strip() or top_k <= 0 or dataset.records.empty:
        return []

    documents: list[str] = []
    rows: list[dict[str, str]] = []
    for record in dataset.records.to_dict(orient="records"):
        claim_text = _text(record.get("claim_text"))
        verified_quote = _text(record.get("verified_evidence_quote_or_table"))
        fallback_evidence = _text(record.get("evidence_text"))
        searchable_evidence = verified_quote or fallback_evidence
        documents.append("\n".join(part for part in (claim_text, searchable_evidence) if part))
        rows.append({key: _text(value) for key, value in record.items()})

    scores = engine.similarities(query.strip(), documents)
    ranked = sorted(
        (
            (float(score), row)
            for score, row in zip(scores, rows)
            if score > 0.0
        ),
        key=lambda item: (-item[0], item[1]["claim_id"]),
    )
    results: list[CuratedClaimResult] = []
    for score, row in ranked[:top_k]:
        quote = row.get("verified_evidence_quote_or_table", "")
        page = row.get("evidence_pdf_page", "") or row.get("evidence_page", "")
        section = row.get("verified_evidence_section", "") or row.get("evidence_section", "")
        if page.endswith(".0") and page[:-2].isdigit():
            page = page[:-2]
        results.append(
            CuratedClaimResult(
                claim_id=row["claim_id"],
                paper_id=row["paper_id"],
                paper_title=row["paper_title"],
                source_filename=dataset.paper_files[row["paper_id"]],
                claim_text=row["claim_text"],
                evidence_quote=quote,
                evidence_text=row.get("evidence_text", ""),
                evidence_page=page,
                evidence_section=section,
                evidence_status=row.get("evidence_status", ""),
                annotation_status=row.get("annotation_status", ""),
                validation_decision=row.get("validation_decision", ""),
                review_rationale=row.get("review_rationale", ""),
                normalized_validation_status=row["normalized_validation_status"],
                validation_status_reason=row["validation_status_reason"],
                similarity=score,
                original_record=row,
            )
        )
    return results


def load_optional_analysis_files(
    dataset_root: str | Path = DATASET_ROOT,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """Load optional analysis CSVs independently, reporting absent/invalid files."""
    root = Path(dataset_root).expanduser().resolve()
    analysis_dir = root / "Cross Paper Analysis"
    loaded: dict[str, pd.DataFrame] = {}
    warnings: dict[str, str] = {}
    for label, (filename, required_columns) in OPTIONAL_ANALYSIS_SCHEMAS.items():
        path = analysis_dir / filename
        if not path.is_file():
            warnings[label] = f"Optional {label.lower()} file is missing: {path}"
            continue
        try:
            frame = _read_csv(path, label)
        except CuratedDatasetError as exc:
            warnings[label] = str(exc)
            continue
        missing_columns = sorted(required_columns - set(frame.columns))
        if missing_columns:
            warnings[label] = (
                f"Optional {label.lower()} file has invalid schema; missing columns: "
                + ", ".join(missing_columns)
            )
            continue
        loaded[label] = frame
    return loaded, warnings


def analysis_related_to_claims(
    analysis: Mapping[str, pd.DataFrame],
    claim_ids: Sequence[str],
    paper_ids: Sequence[str],
) -> dict[str, pd.DataFrame]:
    """Filter original analysis rows by explicit claim-ID/paper-ID references."""
    selected_claims = set(claim_ids)
    selected_papers = set(paper_ids)
    related: dict[str, pd.DataFrame] = {}
    for label, frame in analysis.items():
        if label == "Cross-paper comparison" and "claim_ids" in frame:
            mask = frame["claim_ids"].fillna("").map(
                lambda value: bool(selected_claims & set(
                    re.findall(r"P\d{3}-[A-Za-z0-9-]+", str(value))
                ))
            )
        elif label == "Disagreement analysis" and "claim_ids" in frame:
            mask = frame["claim_ids"].fillna("").map(
                lambda value: bool(selected_claims & set(
                    re.findall(r"P\d{3}-[A-Za-z0-9-]+", str(value))
                ))
            )
        elif label == "Uncertainty register" and "papers_involved" in frame:
            mask = frame["papers_involved"].fillna("").map(
                lambda value: bool(selected_papers & set(
                    re.findall(r"P\d{3}", str(value))
                ))
            )
        else:
            continue
        related[label] = frame.loc[mask].copy()
    return related

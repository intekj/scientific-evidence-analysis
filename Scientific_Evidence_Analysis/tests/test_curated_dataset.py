import csv

import pandas as pd
import pytest

from modules.curated_dataset import (
    CANONICAL_PAPER_IDS,
    CANONICAL_PAPER_FILENAMES,
    OPTIONAL_ANALYSIS_SCHEMAS,
    CuratedDatasetError,
    REQUIRED_CLAIM_COLUMNS,
    VALIDATION_STATUSES,
    analysis_related_to_claims,
    load_curated_dataset,
    load_optional_analysis_files,
    normalize_validation_status,
    retrieve_curated_claims,
)
from modules.embedding_engine import EmbeddingEngine


def _records() -> pd.DataFrame:
    rows = []
    for paper_id in CANONICAL_PAPER_IDS:
        rows.append(
            {
                column: ""
                for column in REQUIRED_CLAIM_COLUMNS
            }
            | {
                "claim_id": f"{paper_id}-C01",
                "paper_id": paper_id,
                "paper_title": f"Paper title for {paper_id}",
                "claim_text": f"The model improves factual accuracy in {paper_id}.",
                "source_filename": CANONICAL_PAPER_FILENAMES[paper_id],
                "evidence_status": "WITHIN-PAPER VERIFIED",
                "annotation_status": "Manually reviewed against original PDF",
                "validation_decision": "Retain — verified result",
                "verified_evidence_quote_or_table": (
                    f"The model improved factual accuracy in {paper_id}."
                ),
                "evidence_pdf_page": "7.0",
                "verified_evidence_section": "Results",
                "atomicity_review_status": "",
                "atomicity_review_decision": "",
                "atomicity_review_rationale": "",
                "review_rationale": "Reviewed source result.",
                "claim_type": "Empirical finding",
                "evidence_text": "",
                "evidence_page": "",
                "evidence_section": "",
            }
        )
    return pd.DataFrame(rows)


def _write_claims(path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def test_loads_claims_and_preserves_original_columns_and_paper_mapping(tmp_path):
    source = tmp_path / "Master Dataset" / "claims.csv"
    original = _records()
    _write_claims(source, original)

    dataset = load_curated_dataset(source)

    assert len(dataset.records) == len(CANONICAL_PAPER_IDS)
    assert set(original.columns).issubset(dataset.records.columns)
    assert dataset.paper_files == CANONICAL_PAPER_FILENAMES
    assert dataset.records.loc[0, "claim_text"] == original.loc[0, "claim_text"]
    assert dataset.records.loc[0, "normalized_validation_status"] == "RETAINED"


def test_actual_master_dataset_has_291_unique_claims_when_available():
    from modules.curated_dataset import MASTER_CLAIMS_PATH

    if not MASTER_CLAIMS_PATH.is_file():
        pytest.skip("The sibling AI Papers Dataset is not present in this checkout.")
    dataset = load_curated_dataset()

    assert len(dataset.records) == 291
    assert dataset.records["claim_id"].nunique() == 291
    assert set(dataset.paper_files) == set(CANONICAL_PAPER_IDS)


def test_duplicate_claim_ids_are_rejected(tmp_path):
    source = tmp_path / "claims.csv"
    frame = _records()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    _write_claims(source, frame)

    with pytest.raises(CuratedDatasetError, match="Duplicate claim_id"):
        load_curated_dataset(source)


def test_missing_required_column_is_rejected(tmp_path):
    source = tmp_path / "claims.csv"
    frame = _records().drop(columns=["validation_decision"])
    _write_claims(source, frame)

    with pytest.raises(CuratedDatasetError, match="missing required columns"):
        load_curated_dataset(source)


def test_noncanonical_paper_id_is_rejected(tmp_path):
    source = tmp_path / "claims.csv"
    frame = _records()
    frame.loc[20, "paper_id"] = "P999"
    _write_claims(source, frame)

    with pytest.raises(CuratedDatasetError, match="Noncanonical paper_id"):
        load_curated_dataset(source)


def test_incorrect_p001_filename_mapping_is_rejected(tmp_path):
    source = tmp_path / "claims.csv"
    frame = _records()
    frame.loc[frame["paper_id"] == "P001", "source_filename"] = "wrong.pdf"
    _write_claims(source, frame)

    with pytest.raises(
        CuratedDatasetError,
        match=r"P001: expected '2023\.emnlp-main\.397\.pdf', found 'wrong\.pdf'",
    ):
        load_curated_dataset(source)


def test_correct_p001_to_p021_filename_mapping_is_accepted(tmp_path):
    source = tmp_path / "claims.csv"
    _write_claims(source, _records())

    dataset = load_curated_dataset(source)

    assert dataset.paper_files == CANONICAL_PAPER_FILENAMES
    assert tuple(dataset.paper_files) == CANONICAL_PAPER_IDS


def test_missing_required_values_are_rejected_but_missing_evidence_is_allowed(tmp_path):
    source = tmp_path / "claims.csv"
    frame = _records()
    frame.loc[0, "verified_evidence_quote_or_table"] = ""
    frame.loc[0, "evidence_text"] = ""
    frame.loc[0, "evidence_pdf_page"] = ""
    frame.loc[0, "evidence_page"] = ""
    _write_claims(source, frame)

    dataset = load_curated_dataset(source)
    engine = EmbeddingEngine()
    engine._model_checked = True
    results = retrieve_curated_claims(
        "improves factual accuracy",
        dataset,
        engine,
        top_k=1,
    )
    assert results[0].claim_id == "P001-C01"
    assert results[0].evidence_quote == ""
    assert results[0].evidence_page == ""

    frame.loc[0, "claim_text"] = ""
    _write_claims(source, frame)
    with pytest.raises(CuratedDatasetError, match="missing claim_text"):
        load_curated_dataset(source)


@pytest.mark.parametrize(
    ("decision", "annotation", "evidence", "claim_type", "expected"),
    [
        ("Reject — unsupported", "Manually reviewed", "", "", "REJECTED"),
        ("Merge with P001-C02; duplicate", "Manually reviewed", "", "", "MERGED"),
        ("Retain corrected wording", "Manually reviewed", "", "", "CORRECTED"),
        ("Retain — scoped result", "Manually reviewed against PDF", "", "", "RETAINED"),
        ("Supported with scope", "Candidate validated with scope note", "", "", "RETAINED"),
        ("Pending review", "Candidate", "", "", "CANDIDATE / NEEDS REVIEW"),
        ("", "Manually reviewed", "Verified as study context, not an independent claim",
         "Study context", "CONTEXT ONLY"),
        ("Context only — not an independent claim", "Manually reviewed", "", "",
         "CONTEXT ONLY"),
        ("", "", "", "", "UNKNOWN / NEEDS REVIEW"),
        ("Retain", "Candidate, not yet reviewed", "", "", "CANDIDATE / NEEDS REVIEW"),
        ("Retain", "Not reviewed", "", "", "CANDIDATE / NEEDS REVIEW"),
        ("Retain", "Review not completed", "", "", "CANDIDATE / NEEDS REVIEW"),
        ("Retain", "Pending review", "", "", "CANDIDATE / NEEDS REVIEW"),
    ],
)
def test_validation_status_normalization_uses_review_decisions(
    decision, annotation, evidence, claim_type, expected
):
    status, reason = normalize_validation_status(
        decision,
        annotation,
        evidence,
        claim_type,
    )

    assert status == expected
    assert reason


def test_normalization_covers_all_seven_documented_statuses():
    cases = [
        ("Retain result", "Manually reviewed", "", "", "RETAINED"),
        ("Retain corrected wording", "Manually reviewed", "", "", "CORRECTED"),
        ("Merge duplicate", "Manually reviewed", "", "", "MERGED"),
        ("Reject claim", "Manually reviewed", "", "", "REJECTED"),
        ("Pending review", "", "", "", "CANDIDATE / NEEDS REVIEW"),
        ("Context only", "", "", "", "CONTEXT ONLY"),
        ("", "", "", "", "UNKNOWN / NEEDS REVIEW"),
    ]
    normalized = {
        normalize_validation_status(decision, annotation, evidence, claim_type)[0]
        for decision, annotation, evidence, claim_type, _ in cases
    }

    assert normalized == set(VALIDATION_STATUSES)


def test_curated_retrieval_preserves_claim_quote_page_section_and_status(tmp_path):
    source = tmp_path / "claims.csv"
    _write_claims(source, _records())
    dataset = load_curated_dataset(source)
    engine = EmbeddingEngine()
    engine._model_checked = True

    results = retrieve_curated_claims(
        "How does the model affect factual accuracy?",
        dataset,
        engine,
        top_k=1,
    )

    result = results[0]
    assert result.claim_id == "P001-C01"
    assert result.paper_id == "P001"
    assert result.paper_title == "Paper title for P001"
    assert result.source_filename == CANONICAL_PAPER_FILENAMES["P001"]
    assert result.evidence_quote == "The model improved factual accuracy in P001."
    assert result.evidence_page == "7"
    assert result.evidence_section == "Results"
    assert result.normalized_validation_status == "RETAINED"
    assert result.original_record["validation_decision"] == "Retain — verified result"


def test_missing_master_dataset_is_reported(tmp_path):
    with pytest.raises(CuratedDatasetError, match="was not found"):
        load_curated_dataset(tmp_path / "missing.csv")


def test_missing_optional_analysis_files_are_reported_without_raising(tmp_path):
    loaded, warnings = load_optional_analysis_files(tmp_path)

    assert loaded == {}
    assert "Cross-paper comparison" in warnings
    assert "Disagreement analysis" in warnings
    assert "Uncertainty register" in warnings


def test_optional_analysis_schemas_and_explicit_links_are_preserved(tmp_path):
    analysis_dir = tmp_path / "Cross Paper Analysis"
    analysis_dir.mkdir()
    samples = {
        "Cross-paper comparison": {
            "comparison_id": "B12-001",
            "claim_ids": "['P001-C01']",
            "question": "How do these studies compare?",
            "relationship": "Methodological difference",
            "classification": "context-dependent",
            "interpretation": "Do not pool the scores.",
        },
        "Disagreement analysis": {
            "test_id": "B12B-001",
            "research_question": "Do findings conflict?",
            "claim_ids": "P001-C01; P002-C01",
            "classification": "context-dependent disagreement",
            "direct_contradiction": "No",
            "safe_synthesis": "Results differ by context.",
            "review_status": "Reviewed",
        },
        "Uncertainty register": {
            "uncertainty_id": "U-001",
            "topic": "Metric comparability",
            "papers_involved": "P001, P002",
            "uncertainty_source": "Different metrics.",
            "uncertainty_level": "High",
            "safe_interpretation": "Do not pool scores.",
        },
    }
    for label, (filename, required_columns) in OPTIONAL_ANALYSIS_SCHEMAS.items():
        pd.DataFrame([samples[label]]).to_csv(
            analysis_dir / filename,
            index=False,
            columns=sorted(required_columns),
        )

    loaded, warnings = load_optional_analysis_files(tmp_path)
    related = analysis_related_to_claims(
        loaded,
        ["P001-C01"],
        ["P001"],
    )

    assert warnings == {}
    assert {label: len(frame) for label, frame in related.items()} == {
        "Cross-paper comparison": 1,
        "Disagreement analysis": 1,
        "Uncertainty register": 1,
    }
    assert (
        related["Cross-paper comparison"].iloc[0]["relationship"]
        == "Methodological difference"
    )
    assert related["Uncertainty register"].iloc[0]["safe_interpretation"] == (
        "Do not pool scores."
    )

from modules.claim_verification import EvidenceLabel, classify_evidence
from modules.evidence_retrieval import retrieve_evidence
from modules.text_chunker import EvidenceChunk


class FixedEngine:
    backend = "test"

    def similarities(self, query, passages):
        return [0.8 if "paper one" in passage else 0.6 for passage in passages]


def chunk(paper_id: str, name: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(paper_id, paper_id, name, 1, text)


def test_retrieval_searches_all_papers_and_keeps_source():
    chunks = [
        chunk("1", "one.pdf", "paper one intervention improved outcome"),
        chunk("2", "two.pdf", "paper two intervention reduced outcome"),
    ]
    results = retrieve_evidence("intervention outcome", chunks, engine=FixedEngine())
    assert {item.chunk.paper_name for item in results} == {"one.pdf", "two.pdf"}
    assert results[0].chunk.page_number == 1


def test_claim_groups_are_preliminary_and_directional():
    positive = chunk("1", "one.pdf", "The intervention improved the outcome.")
    negative = chunk("2", "two.pdf", "The intervention reduced the outcome.")
    from modules.evidence_retrieval import RetrievedEvidence

    claim = "The intervention improves the outcome."
    assert classify_evidence(claim, RetrievedEvidence(positive, 0.7)).label == (
        EvidenceLabel.SUPPORTING
    )
    assert classify_evidence(claim, RetrievedEvidence(negative, 0.6)).label == (
        EvidenceLabel.CONTRADICTING
    )


def test_ambiguous_evidence_stays_neutral():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk("1", "one.pdf", "The intervention was evaluated for outcome Y.")
    result = classify_evidence(
        "The intervention affects outcome Y.",
        RetrievedEvidence(evidence, 0.6),
    )
    assert result.label == EvidenceLabel.NEUTRAL


def test_claim_polarity_handles_negation():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk("1", "one.pdf", "The intervention improved the outcome.")
    result = classify_evidence(
        "The intervention does not improve the outcome.",
        RetrievedEvidence(evidence, 0.6),
    )
    assert result.label == EvidenceLabel.CONTRADICTING


def test_question_is_not_automatically_classified_as_supporting_or_contradicting():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk(
        "1",
        "one.pdf",
        "The intervention improved the outcome.",
    )
    result = classify_evidence(
        "Does the intervention improve the outcome?",
        RetrievedEvidence(evidence, 0.8),
    )
    assert result.label == EvidenceLabel.NEUTRAL
    assert "research question" in result.reason


def test_title_and_author_paragraph_is_not_a_contradicting_finding():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk(
        "1",
        "one.pdf",
        "Title: The intervention does not improve the outcome. "
        "Authors: Smith and Lee. Correspondence: researcher@example.org.",
    )
    result = classify_evidence(
        "The intervention improves the outcome.",
        RetrievedEvidence(evidence, 0.7),
    )
    assert result.label == EvidenceLabel.NEUTRAL


def test_methodology_description_is_not_a_contradicting_finding():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk(
        "1",
        "one.pdf",
        "The methods compare two algorithms and tune parameters to reduce prediction error.",
    )
    result = classify_evidence(
        "The algorithm reduces prediction error.",
        RetrievedEvidence(evidence, 0.7),
    )
    assert result.label == EvidenceLabel.NEUTRAL


def test_metric_value_difference_without_verified_comparable_settings_stays_neutral():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk(
        "1",
        "one.pdf",
        "XGBoost achieved AUC = 0.994 in the test dataset.",
    )
    result = classify_evidence(
        "XGBoost achieved AUC = 0.985.",
        RetrievedEvidence(evidence, 0.8),
    )
    assert result.label == EvidenceLabel.NEUTRAL
    assert "not labeled a contradiction" in result.reason


def test_retrieval_excludes_reference_entries_and_exact_duplicates():
    chunks = [
        chunk("1", "one.pdf", "XGBoost achieved the highest AUC in the results."),
        chunk("1", "one.pdf", "XGBoost achieved the highest AUC in the results."),
        chunk(
            "1",
            "one.pdf",
            "Smith et al. (2020). Journal of Flood Research. "
            "https://doi.org/10.0000/example",
        ),
    ]
    results = retrieve_evidence("XGBoost AUC", chunks, engine=FixedEngine())
    assert len(results) == 1
    assert "XGBoost achieved" in results[0].chunk.text


def test_retrieval_keeps_similar_results_with_different_reported_values():
    chunks = [
        chunk("1", "one.pdf", "XGBoost achieved an AUC score of 0.985."),
        chunk("2", "two.pdf", "XGBoost achieved an AUC score of 0.994."),
    ]
    results = retrieve_evidence("XGBoost AUC", chunks, engine=FixedEngine())
    assert len(results) == 2
    assert {item.chunk.paper_name for item in results} == {"one.pdf", "two.pdf"}


def test_results_section_ranks_ahead_of_literature_review():
    chunks = [
        chunk("1", "one.pdf", "Previous studies used XGBoost for AUC prediction."),
        chunk("2", "two.pdf", "Results: XGBoost achieved the highest AUC."),
    ]
    results = retrieve_evidence("XGBoost AUC", chunks, engine=FixedEngine())
    assert results[0].chunk.paper_name == "two.pdf"


def test_retrieval_balances_papers_and_preserves_page_references():
    chunks = [
        EvidenceChunk("1a", "1", "one.pdf", 1, "paper one result passage AUC"),
        EvidenceChunk("1b", "1", "one.pdf", 2, "paper one second result AUC"),
        EvidenceChunk("2a", "2", "two.pdf", 8, "paper two result AUC"),
    ]
    results = retrieve_evidence("paper result AUC", chunks, engine=FixedEngine())
    assert {item.chunk.paper_name for item in results[:2]} == {"one.pdf", "two.pdf"}
    assert results[-1].chunk.page_number in {1, 2, 8}


def test_overall_best_claim_does_not_ignore_higher_ensemble_auc():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk(
        "1",
        "study.pdf",
        "From 2000–2018 XGBoost achieved AUC 0.985, Random Forest achieved AUC 0.984, "
        "LightGBM achieved AUC 0.982, and the voting ensemble reported AUC 0.994.",
    )
    result = classify_evidence(
        "XGBoost is the best overall model.",
        RetrievedEvidence(evidence, 0.9),
    )
    assert result.label == EvidenceLabel.CONTRADICTING
    assert "another named model with a higher reported value" in result.reason


def test_ensemble_best_claim_matches_highest_reported_metric():
    from modules.evidence_retrieval import RetrievedEvidence

    evidence = chunk(
        "1",
        "study.pdf",
        "XGBoost achieved AUC 0.985, and the voting ensemble reported AUC 0.994.",
    )
    result = classify_evidence(
        "The voting ensemble is the best overall model.",
        RetrievedEvidence(evidence, 0.9),
    )
    assert result.label == EvidenceLabel.SUPPORTING

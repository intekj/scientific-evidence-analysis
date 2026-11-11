from modules.claim_verification import ClassifiedEvidence, EvidenceLabel
from modules.evidence_retrieval import RetrievedEvidence
from modules.evidence_synthesis import generate_synthesis
from modules.text_chunker import EvidenceChunk


def test_synthesis_cites_uploaded_passages_and_states_missing_groups():
    chunk = EvidenceChunk(
        "paper-id",
        "paper-id",
        "source.pdf",
        3,
        "The intervention improved the reported outcome.",
    )
    item = ClassifiedEvidence(
        RetrievedEvidence(chunk, 0.73),
        EvidenceLabel.SUPPORTING,
        "Matching preliminary directional cues.",
    )
    report = generate_synthesis("The intervention improves outcome.", [item], [chunk])
    assert "[source.pdf, p. 3]" in report
    assert "No relevant evidence was found" in report
    assert "does not prove the claim" in report


def test_empty_synthesis_does_not_invent_evidence():
    report = generate_synthesis("An unsearched claim.", [], [])
    assert "no passages were retrieved" in report
    assert "source.pdf" not in report


def test_generate_synthesis_accepts_claim_items_and_full_paper_chunks():
    retrieved = EvidenceChunk(
        "paper-id",
        "paper-id",
        "source.pdf",
        3,
        "The voting ensemble achieved AUC = 0.994.",
    )
    context = EvidenceChunk(
        "paper-context",
        "paper-id",
        "source.pdf",
        21,
        "The study area was Karun Basin, Iran.",
    )
    item = ClassifiedEvidence(
        RetrievedEvidence(retrieved, 0.82),
        EvidenceLabel.NEUTRAL,
        "Retrieved result for an open research question.",
    )

    report = generate_synthesis(
        "Which model performs best?",
        [item],
        [retrieved, context],
    )

    assert "The voting ensemble achieved AUC = 0.994" in report
    assert "Karun Basin" in report
    assert "[source.pdf, p. 3]" in report
    assert "[source.pdf, p. 21]" in report


def test_synthesis_separates_ensemble_and_individual_source_findings():
    chunk = EvidenceChunk(
        "paper-id",
        "paper-id",
        "scientific-report.pdf",
        12,
        "XGBoost achieved AUC 0.985, Random Forest achieved AUC 0.984, "
        "LightGBM achieved AUC 0.982. The voting ensemble reported AUC 0.994 "
        "in the conclusion.",
    )
    item = ClassifiedEvidence(
        RetrievedEvidence(chunk, 0.9),
        EvidenceLabel.SUPPORTING,
        "Retrieved result passage.",
    )
    report = generate_synthesis("Which model has the highest AUC?", [item], [chunk])
    assert "Individual versus ensemble performance" in report
    assert "voting ensemble reported AUC 0.994" in report
    assert "scientific-report.pdf, p. 12" in report
    assert "no ranking is inferred" not in report


def test_question_synthesis_answers_from_neutral_findings_without_polarity_labels():
    first = EvidenceChunk(
        "p1",
        "p1",
        "study-one.pdf",
        4,
        "Cascade Forest Model (CFM) achieved the highest accuracy across two basins.",
    )
    second = EvidenceChunk(
        "p2",
        "p2",
        "study-two.pdf",
        3,
        "XGBoost was the best individual algorithm among those evaluated.",
    )
    ensemble = EvidenceChunk(
        "p2",
        "p2",
        "study-two.pdf",
        21,
        "The voting ensemble achieved AUC = 0.994.",
    )
    items = [
        ClassifiedEvidence(RetrievedEvidence(chunk, 0.8), EvidenceLabel.NEUTRAL,
                           "Question evidence is not polarity-classified.")
        for chunk in (first, second, ensemble)
    ]

    report = generate_synthesis(
        "Which machine learning algorithm performs best across these studies?",
        items,
        [first, second, ensemble],
    )

    assert "## 1. Research Question" in report
    assert "## 2. Relevant Findings from Paper 1" in report
    assert "## 3. Relevant Findings from Paper 2" in report
    assert "## 4. Individual Model Performance" in report
    assert "## 5. Ensemble Model Performance" in report
    assert "## 6. Study Context Comparison" in report
    assert "## 7. Cross-Study Interpretation" in report
    assert "## 8. Overall Answer" in report
    assert "## 9. Limitations and Uncertainty" in report
    assert "CFM) achieved the highest accuracy" in report
    assert "XGBoost was the best individual algorithm" in report
    assert "voting ensemble achieved AUC = 0.994" in report
    assert "study-one.pdf, p. 4" in report
    assert "study-two.pdf, p. 21" in report
    assert "do not establish one universally best algorithm" in report
    assert "Neutral / Insufficient Evidence" not in report


def test_full_paper_context_aggregates_later_page_and_filters_related_work():
    retrieved = EvidenceChunk(
        "p2",
        "p2",
        "study-two.pdf",
        1,
        "XGBoost was the best individual algorithm among those evaluated.",
    )
    later_page = EvidenceChunk(
        "p2-page21",
        "p2",
        "study-two.pdf",
        21,
        "The observational flood data used satellite-based GFD data from "
        "Iran's Karun Basin, Khuzestan Province. The voting ensemble achieved "
        "AUC = 0.994.",
    )
    related_work = EvidenceChunk(
        "p2-page5",
        "p2",
        "study-two.pdf",
        5,
        "Previous studies in Majuli Island and southern Iran reported other locations.",
    )
    items = [
        ClassifiedEvidence(
            RetrievedEvidence(retrieved, 0.8),
            EvidenceLabel.NEUTRAL,
            "Research question.",
        )
    ]

    report = generate_synthesis(
        "Which model performs best?",
        items,
        [retrieved, related_work, later_page],
    )

    assert "Karun Basin" in report
    assert "Khuzestan Province" in report
    assert "study-two.pdf, p. 21" in report
    assert "Majuli Island" not in report
    assert "southern Iran" not in report


def test_context_aggregation_uses_study_area_not_related_work_section():
    current_study = EvidenceChunk(
        "p1",
        "p1",
        "study-one.pdf",
        6,
        "Study Area\nThe study area is Karun Basin, Iran.",
    )
    related = EvidenceChunk(
        "p1-related",
        "p1",
        "study-one.pdf",
        2,
        "Related Work\nFlood susceptibility was evaluated on Majuli Island.",
    )
    evidence = ClassifiedEvidence(
        RetrievedEvidence(current_study, 0.75),
        EvidenceLabel.NEUTRAL,
        "Retrieved location context.",
    )
    report = generate_synthesis(
        "Which locations did the studies evaluate?",
        [evidence],
        [related, current_study],
    )
    assert "Karun Basin" in report
    assert "study-one.pdf, p. 6" in report
    assert "Majuli Island" not in report

from modules.claim_verification import EvidenceLabel
from modules.evidence_presentation import (
    classify_for_display,
    extract_key_findings,
    highlighted_passage,
    synthesis_answer,
)
from modules.evidence_retrieval import RetrievedEvidence
from modules.evidence_synthesis import generate_synthesis
from modules.text_chunker import EvidenceChunk


def test_key_findings_are_exact_sentences_with_original_provenance():
    chunk = EvidenceChunk(
        "sha256-paper-id",
        "sha256-paper-id",
        "research.pdf",
        7,
        "The intervention was evaluated in adults. "
        "The intervention improved the measured outcome by 12%.",
    )
    findings = extract_key_findings(
        "Did the intervention improve the measured outcome?",
        [RetrievedEvidence(chunk, 0.81)],
    )

    assert findings
    assert findings[0].sentence == (
        "The intervention improved the measured outcome by 12%."
    )
    assert findings[0].evidence.chunk.paper_name == "research.pdf"
    assert findings[0].evidence.chunk.paper_id == "sha256-paper-id"
    assert findings[0].evidence.chunk.page_number == 7


def test_highlighting_marks_verbatim_relevant_sentence_and_escapes_html():
    passage = "A relevant <finding> improved the outcome. Another sentence is unrelated."

    rendered = highlighted_passage("relevant finding outcome", passage)

    assert (
        "<mark>A relevant &lt;finding&gt; improved the outcome.</mark>"
        in rendered
    )
    assert "Another sentence is unrelated." in rendered
    assert "<finding>" not in rendered


def test_question_retrieval_is_not_labeled_supporting_by_similarity():
    chunk = EvidenceChunk(
        "paper-id",
        "paper-id",
        "study.pdf",
        2,
        "The intervention improved the outcome.",
    )
    category, classified = classify_for_display(
        "Does the intervention improve the outcome?",
        RetrievedEvidence(chunk, 0.99),
    )

    assert category == "Relevant (not polarity-classified)"
    assert classified.label == EvidenceLabel.NEUTRAL


def test_qualifying_and_background_passages_have_distinct_categories():
    qualifying = EvidenceChunk(
        "paper-id",
        "paper-id",
        "study.pdf",
        3,
        "The intervention may improve the outcome, although evidence is limited.",
    )
    background = EvidenceChunk(
        "paper-id",
        "paper-id",
        "study.pdf",
        1,
        "The study recruited 120 participants from two clinics.",
    )

    assert classify_for_display(
        "Does the intervention improve the outcome?",
        RetrievedEvidence(qualifying, 0.7),
    )[0] == "Qualifying Evidence"
    assert classify_for_display(
        "Does the intervention improve the outcome?",
        RetrievedEvidence(background, 0.6),
    )[0] == "Background / Context"


def test_synthesis_answer_reports_insufficient_evidence_without_invention():
    report = generate_synthesis("Which treatment works best?", [], [])

    assert synthesis_answer(report) == (
        "The retrieved evidence is insufficient to provide an answer."
    )


def test_synthesis_answer_uses_the_existing_cited_answer_section():
    chunk = EvidenceChunk(
        "paper-id",
        "paper-id",
        "study.pdf",
        8,
        "The voting ensemble achieved AUC 0.994.",
    )
    item = RetrievedEvidence(chunk, 0.9)
    from modules.claim_verification import ClassifiedEvidence

    report = generate_synthesis(
        "Which model achieved the highest AUC?",
        [
            ClassifiedEvidence(
                item,
                EvidenceLabel.NEUTRAL,
                "Questions are not assigned polarity labels.",
            )
        ],
        [chunk],
    )
    findings = extract_key_findings(
        "Which model achieved the highest AUC?",
        [item],
    )
    answer = synthesis_answer(report, findings)

    assert "voting ensemble" in answer
    assert "study.pdf, paper ID paper-id, p. 8" in answer

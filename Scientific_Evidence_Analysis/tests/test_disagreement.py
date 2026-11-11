from modules.disagreement_analysis import (
    DisagreementCategory,
    compare_evidence,
)
from modules.text_chunker import EvidenceChunk
from tests.test_context_extraction import PAPER_ONE_PASSAGE, PAPER_TWO_PASSAGE


def make_chunk(paper_id: str, name: str, text: str) -> EvidenceChunk:
    return EvidenceChunk(paper_id, paper_id, name, 1, text)


def test_context_dependent_when_dataset_differs():
    first = make_chunk(
        "1",
        "one.pdf",
        "Dataset A randomized trial included adults and reported improved accuracy.",
    )
    second = make_chunk(
        "2",
        "two.pdf",
        "Dataset B randomized trial included adults and reported reduced accuracy.",
    )
    result = compare_evidence(first, second)
    assert result.category == DisagreementCategory.CONTEXT_DEPENDENT
    assert "Dataset" in result.differing_fields


def test_potential_genuine_only_when_context_clues_match():
    first = make_chunk(
        "1",
        "one.pdf",
        "Dataset A randomized trial included adults and reported improved accuracy.",
    )
    second = make_chunk(
        "2",
        "two.pdf",
        "Dataset A randomized trial included adults and reported reduced accuracy.",
    )
    result = compare_evidence(first, second)
    assert result.category == DisagreementCategory.GENUINE
    assert len(result.shared_fields) >= 2


def test_insufficient_without_opposing_findings():
    first = make_chunk("1", "one.pdf", "The randomized trial included adult patients.")
    second = make_chunk("2", "two.pdf", "The randomized trial included adult patients.")
    assert compare_evidence(first, second).category == DisagreementCategory.INSUFFICIENT


def test_unrelated_passages_are_not_reported_as_competing_findings():
    first = make_chunk("1", "one.pdf", "The intervention improved accuracy.")
    second = make_chunk("2", "two.pdf", "Rainfall decreased across the region.")
    result = compare_evidence(first, second)
    assert result.category == DisagreementCategory.INSUFFICIENT
    assert "no clear shared content terms" in result.reason


def test_supplied_flood_paper_examples_are_context_dependent():
    first = make_chunk("1", "paper-one.pdf", PAPER_ONE_PASSAGE)
    second = make_chunk("2", "paper-two.pdf", PAPER_TWO_PASSAGE)

    result = compare_evidence(first, second)

    assert result.category == DisagreementCategory.CONTEXT_DEPENDENT
    assert "Dataset" in result.differing_fields
    assert "different model rankings" in result.reason
    assert result.first_context["Geographic location"]
    assert result.second_context["Geographic location"] is None
    assert result.second_context["Dataset"]


def test_different_model_rankings_are_not_a_genuine_contradiction():
    first = make_chunk(
        "1",
        "one.pdf",
        "In basin A, model CFM achieved the highest accuracy.",
    )
    second = make_chunk(
        "2",
        "two.pdf",
        "In basin A, model XGBoost achieved the highest accuracy.",
    )

    result = compare_evidence(first, second)

    assert result.category != DisagreementCategory.GENUINE

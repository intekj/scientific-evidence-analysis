import pytest

from datasets import Dataset, DatasetDict

from download_qasper import (
    QasperDownloadError,
    load_prepared_dataset,
    load_raw_dataset,
    save_prepared_dataset,
)
from inspect_qasper import (
    extract_questions,
    inspect_split,
    load_local_dataset,
    validate_record,
)


def sample_record(
    *,
    paper_id: str = "paper-1",
    questions: list[str] | None = None,
    include_title: bool = True,
    include_answerability: bool = True,
) -> dict:
    question_list = questions if questions is not None else ["What is the result?"]
    answers = []
    for index, _ in enumerate(question_list):
        annotation = {
            "unanswerable": index == 1 if include_answerability else None,
            "extractive_spans": ["an evidence span"],
            "yes_no": None,
            "free_form_answer": "A concise answer",
            "evidence": ["A source paragraph supporting the answer."],
            "highlighted_evidence": ["A highlighted sentence."],
        }
        answers.append(
            {
                "annotation_id": [f"annotation-{index}"],
                "answer": [annotation],
            }
        )
    record = {
        "id": paper_id,
        "abstract": "A short paper abstract.",
        "full_text": {
            "section_name": ["Introduction", "Results"],
            "paragraphs": [["Intro paragraph"], ["Result paragraph"]],
        },
        "qas": {
            "question": question_list,
            "question_id": [
                f"question-{index}" for index in range(len(question_list))
            ],
            "answers": answers,
        },
    }
    if include_title:
        record["title"] = "Example scientific paper"
    return record


def test_loads_dataset_splits_from_local_parquet_files(tmp_path):
    train_path = tmp_path / "train.parquet"
    validation_path = tmp_path / "validation.parquet"
    Dataset.from_list([sample_record()]).to_parquet(str(train_path))
    Dataset.from_list([sample_record(paper_id="paper-2")]).to_parquet(
        str(validation_path)
    )

    loaded = load_raw_dataset(
        {"train": train_path, "validation": validation_path}
    )

    assert set(loaded) == {"train", "validation"}
    assert len(loaded["train"]) == 1
    assert loaded["validation"][0]["id"] == "paper-2"


def test_split_inspection_counts_papers_questions_and_answerability():
    records = [
        sample_record(
            paper_id="paper-1",
            questions=["Question one?", "Question two?"],
        ),
        sample_record(paper_id="paper-2"),
    ]

    summary = inspect_split(records)

    assert summary["record_count"] == 2
    assert summary["unique_papers"] == 2
    assert summary["question_count"] == 3
    assert summary["answerable_questions"] == 2
    assert summary["unanswerable_questions"] == 1
    assert len(summary["samples"]) == 2


def test_required_field_validation_and_missing_nested_fields():
    record = sample_record(include_title=False)

    assert validate_record(record) == ["title"]

    record["title"] = "Paper"
    record.pop("qas")
    assert validate_record(record) == ["qas"]


def test_missing_questions_answers_and_fields_are_handled():
    record = {
        "id": "paper-with-gaps",
        "qas": {
            "question": ["Question with missing answer fields?"],
        },
    }

    extracted = extract_questions(record)

    assert len(extracted) == 1
    assert extracted[0]["question_id"] is None
    assert extracted[0]["answer_annotations"] == []
    assert extracted[0]["evidence_passages"] == []
    assert extracted[0]["answerability"] == "not specified"


def test_sample_record_extraction_includes_evidence_and_annotation_fields():
    question = extract_questions(sample_record())[0]

    assert question["question_id"] == "question-0"
    assert question["question"] == "What is the result?"
    assert question["answerability"] == "answerable"
    assert question["answer_annotations"][0]["free_form_answer"] == "A concise answer"
    assert question["evidence_passages"] == [
        "A source paragraph supporting the answer."
    ]


def test_local_dataset_save_and_reload(tmp_path):
    dataset = DatasetDict(
        {
            "train": Dataset.from_list([sample_record()]),
            "validation": Dataset.from_list(
                [sample_record(paper_id="paper-2")]
            ),
        }
    )
    destination = tmp_path / "prepared"

    save_prepared_dataset(dataset, destination)
    reloaded = load_prepared_dataset(destination)

    assert set(reloaded) == {"train", "validation"}
    assert reloaded["train"][0]["title"] == "Example scientific paper"
    assert reloaded["validation"][0]["id"] == "paper-2"


def test_inspector_loads_locally_saved_dataset_without_network(tmp_path):
    destination = tmp_path / "prepared"
    DatasetDict(
        {"train": Dataset.from_list([sample_record()])}
    ).save_to_disk(str(destination))

    loaded = load_local_dataset(destination)

    assert len(loaded["train"]) == 1
    assert loaded["train"][0]["id"] == "paper-1"


def test_non_datasetdict_is_not_accepted(tmp_path):
    destination = tmp_path / "dataset"
    Dataset.from_list([sample_record()]).save_to_disk(str(destination))

    with pytest.raises(QasperDownloadError, match="DatasetDict"):
        load_prepared_dataset(destination)

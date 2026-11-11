import json
from hashlib import sha256

import pymupdf

from modules.claim_verification import ClassifiedEvidence, EvidenceLabel
from modules.evidence_retrieval import retrieve_evidence
from modules.evidence_synthesis import generate_synthesis
from modules.paper_storage import LocalPaperStore
from modules.text_chunker import chunk_paper


def make_pdf(text: str, title: str = "Persistent Test Paper") -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.set_metadata({"title": title, "author": "Local Researcher"})
    data = document.tobytes()
    document.close()
    return data


class FixedEngine:
    backend = "test"

    def similarities(self, query, passages):
        return [0.9 if "AUC" in passage else 0.1 for passage in passages]


def test_upload_saves_original_pdf_in_local_upload_directory(tmp_path):
    content = make_pdf("The voting ensemble achieved AUC 0.994.")
    store = LocalPaperStore(tmp_path)

    record, duplicate = store.save_pdf(content, "Original Flood Study.pdf")

    saved_file = tmp_path / "data" / "uploads" / record.stored_filename
    assert saved_file.read_bytes() == content
    assert record.original_filename == "Original Flood Study.pdf"
    assert record.sha256 == sha256(content).hexdigest()
    assert record.paper_id == record.sha256
    assert record.processing_status == "processing"
    assert not duplicate


def test_uploading_same_pdf_twice_does_not_create_duplicate_copy(tmp_path):
    content = make_pdf("The voting ensemble achieved AUC 0.994.")
    store = LocalPaperStore(tmp_path)

    first, first_duplicate = store.save_pdf(content, "first-name.pdf")
    second, second_duplicate = store.save_pdf(content, "another-name.pdf")

    assert not first_duplicate
    assert second_duplicate
    assert first.paper_id == second.paper_id
    assert len(list(store.upload_dir.glob("*.pdf"))) == 1
    assert len(store.list_records()) == 1


def test_metadata_survives_a_simulated_application_restart(tmp_path):
    content = make_pdf("The voting ensemble achieved AUC 0.994.")
    first_store = LocalPaperStore(tmp_path)
    record, _ = first_store.save_pdf(content, "durable.pdf")
    first_store.update_status(record.paper_id, "ready", page_count=1)

    restarted_store = LocalPaperStore(tmp_path)
    restored = restarted_store.restore()
    persisted = json.loads(restarted_store.metadata_path.read_text(encoding="utf-8"))

    assert record.paper_id in restored.papers
    assert restored.records[0].original_filename == "durable.pdf"
    assert restored.records[0].page_count == 1
    assert persisted["papers"][0]["processing_status"] == "ready"


def test_previously_uploaded_paper_is_automatically_restored(tmp_path):
    content = make_pdf("The voting ensemble achieved AUC 0.994.")
    store = LocalPaperStore(tmp_path)
    record, _ = store.save_pdf(content, "restored.pdf")
    store.update_status(record.paper_id, "ready", page_count=1)

    result = LocalPaperStore(tmp_path).restore()

    paper = result.papers[record.paper_id]
    assert paper.paper_name == "restored.pdf"
    assert paper.page_count == 1
    assert "AUC 0.994" in paper.extracted_text
    assert result.records[0].processing_status == "ready"


def test_interrupted_processing_is_retried_on_startup(tmp_path):
    content = make_pdf("The voting ensemble achieved AUC 0.994.")
    store = LocalPaperStore(tmp_path)
    record, _ = store.save_pdf(content, "interrupted.pdf")

    result = LocalPaperStore(tmp_path).restore()

    assert record.paper_id in result.papers
    assert result.records[0].processing_status == "ready"


def test_removing_one_paper_preserves_other_files_and_metadata(tmp_path):
    store = LocalPaperStore(tmp_path)
    first, _ = store.save_pdf(make_pdf("First finding: AUC 0.91."), "first.pdf")
    second, _ = store.save_pdf(make_pdf("Second finding: AUC 0.99."), "second.pdf")
    first_path = store.upload_dir / first.stored_filename
    second_path = store.upload_dir / second.stored_filename

    removed, errors = store.remove_papers([first.paper_id])

    assert removed == [first.paper_id]
    assert errors == []
    assert not first_path.exists()
    assert second_path.exists()
    assert [record.paper_id for record in store.list_records()] == [second.paper_id]


def test_missing_pdf_is_reported_without_crashing_restore(tmp_path):
    store = LocalPaperStore(tmp_path)
    record, _ = store.save_pdf(make_pdf("A result."), "missing.pdf")
    (store.upload_dir / record.stored_filename).unlink()

    result = store.restore()

    assert result.papers == {}
    assert result.records[0].processing_status == "missing"
    assert any("is missing" in warning for warning in result.warnings)


def test_corrupt_metadata_recovers_papers_from_local_pdf_files(tmp_path):
    content = make_pdf("The voting ensemble achieved AUC 0.994.")
    store = LocalPaperStore(tmp_path)
    record, _ = store.save_pdf(content, "recoverable.pdf")
    store.metadata_path.write_text("{ broken JSON", encoding="utf-8")

    result = LocalPaperStore(tmp_path).restore()

    assert record.paper_id in result.papers
    assert result.records[0].original_filename == "recoverable.pdf"
    assert any("metadata could not be read" in item for item in result.warnings)
    assert json.loads(store.metadata_path.read_text(encoding="utf-8"))["papers"]


def test_corrupted_pdf_is_retained_with_failed_status(tmp_path):
    store = LocalPaperStore(tmp_path)
    record, _ = store.save_pdf(b"not a PDF", "corrupted.pdf")

    result = LocalPaperStore(tmp_path).restore()

    assert result.papers == {}
    assert result.records[0].processing_status == "failed"
    assert "readable PDF" in result.records[0].error


def test_restored_papers_are_available_to_retrieval_and_synthesis(tmp_path):
    content = make_pdf("The voting ensemble achieved AUC 0.994.")
    store = LocalPaperStore(tmp_path)
    record, _ = store.save_pdf(content, "evidence.pdf")
    store.update_status(record.paper_id, "ready", page_count=1)
    restored = LocalPaperStore(tmp_path).restore()
    chunks = chunk_paper(restored.papers[record.paper_id])

    retrieved = retrieve_evidence(
        "voting ensemble AUC",
        chunks,
        engine=FixedEngine(),
    )
    classified = [
        ClassifiedEvidence(
            evidence=item,
            label=EvidenceLabel.NEUTRAL,
            reason="Restored local evidence.",
        )
        for item in retrieved
    ]
    report = generate_synthesis("Which model performs best?", classified, chunks)

    assert retrieved[0].chunk.paper_name == "evidence.pdf"
    assert retrieved[0].chunk.page_number == 1
    assert "voting ensemble achieved AUC 0.994" in report
    assert "[evidence.pdf, p. 1]" in report

from hashlib import sha256

import pymupdf
import pytest

from modules.curated_dataset import (
    CANONICAL_PAPER_FILENAMES,
    DATASET_ROOT,
    CuratedDatasetError,
    discover_canonical_pdfs,
    import_canonical_pdfs,
    validate_canonical_pdf_mapping,
)
from modules.evidence_retrieval import retrieve_evidence
from modules.paper_storage import LocalPaperStore
from modules.pdf_processor import process_pdf
from modules.text_chunker import chunk_paper


def make_pdf(text: str) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    content = document.tobytes()
    document.close()
    return content


class FixedEngine:
    def similarities(self, query, passages):
        return [0.9 if query.casefold() in passage.casefold() else 0.1 for passage in passages]


def test_actual_canonical_pdfs_are_discovered_and_readable():
    if not (DATASET_ROOT / "papers").is_dir():
        pytest.skip("The sibling AI Papers Dataset is not present.")

    discovered, warnings = discover_canonical_pdfs()

    assert not warnings
    assert set(discovered) == set(CANONICAL_PAPER_FILENAMES)
    assert len(discovered) == 21
    assert all(
        path.name == CANONICAL_PAPER_FILENAMES[paper_id]
        for paper_id, path in discovered.items()
    )
    assert "2608.07370v2.pdf" not in {path.name for path in discovered.values()}
    for paper_id, path in discovered.items():
        paper = process_pdf(
            path.read_bytes(),
            path.name,
            canonical_paper_id=paper_id,
        )
        assert paper.pages
        assert all(page.canonical_paper_id == paper_id for page in paper.pages)


def test_missing_canonical_pdfs_are_reported(tmp_path):
    papers_dir = tmp_path / "papers" / "subfolder"
    papers_dir.mkdir(parents=True)
    (papers_dir / CANONICAL_PAPER_FILENAMES["P001"]).write_bytes(
        make_pdf("P001 has readable evidence.")
    )

    discovered, warnings = discover_canonical_pdfs(tmp_path)

    assert set(discovered) == {"P001"}
    assert any("P002" in warning and "missing" in warning for warning in warnings)


def test_incorrect_canonical_filename_mapping_is_rejected():
    with pytest.raises(CuratedDatasetError, match="Invalid canonical PDF mapping"):
        validate_canonical_pdf_mapping("P001", "incorrect.pdf")


def test_unreadable_canonical_pdf_is_reported_without_registering_it(tmp_path):
    papers_dir = tmp_path / "papers"
    papers_dir.mkdir()
    (papers_dir / CANONICAL_PAPER_FILENAMES["P001"]).write_bytes(b"not a PDF")
    store = LocalPaperStore(tmp_path / "project")

    result = import_canonical_pdfs(store, dataset_root=tmp_path)

    assert result.discovered_count == 1
    assert result.indexed_count == 0
    assert any("Could not index canonical PDF P001" in item for item in result.warnings)
    assert store.list_records() == []


def test_canonical_import_is_idempotent_after_restart(tmp_path):
    source_root = tmp_path / "dataset"
    papers_dir = source_root / "papers"
    papers_dir.mkdir(parents=True)
    (papers_dir / CANONICAL_PAPER_FILENAMES["P001"]).write_bytes(
        make_pdf("P001 canonical evidence.")
    )
    project_dir = tmp_path / "project"

    first_store = LocalPaperStore(project_dir)
    first = import_canonical_pdfs(first_store, dataset_root=source_root)

    restarted_store = LocalPaperStore(project_dir)
    restored = restarted_store.restore()
    second = import_canonical_pdfs(
        restarted_store,
        dataset_root=source_root,
        restored_papers=restored.papers,
    )

    assert first.indexed_count == second.indexed_count == 1
    assert len(first_store.list_records()) == len(restarted_store.list_records()) == 1
    record = restarted_store.list_records()[0]
    assert record.canonical_paper_id == "P001"
    assert record.sha256 == record.paper_id == sha256(
        (papers_dir / CANONICAL_PAPER_FILENAMES["P001"]).read_bytes()
    ).hexdigest()
    assert second.papers == {}


def test_canonical_retrieval_preserves_paper_id_filename_and_page(tmp_path):
    source_root = tmp_path / "dataset"
    papers_dir = source_root / "papers"
    papers_dir.mkdir(parents=True)
    filename = CANONICAL_PAPER_FILENAMES["P001"]
    (papers_dir / filename).write_bytes(
        make_pdf("The retrieval evaluation improved scientific evidence accuracy.")
    )
    store = LocalPaperStore(tmp_path / "project")

    imported = import_canonical_pdfs(store, dataset_root=source_root)
    paper = next(iter(imported.papers.values()))
    chunks = chunk_paper(paper)
    results = retrieve_evidence(
        "retrieval evaluation",
        chunks,
        top_k=1,
        engine=FixedEngine(),
    )

    assert results
    assert results[0].chunk.paper_id == "P001"
    assert results[0].chunk.paper_name == filename
    assert results[0].chunk.page_number == 1
    assert results[0].chunk.text.startswith("The retrieval evaluation")


def test_existing_user_uploads_keep_hash_ids_and_restore_behavior(tmp_path):
    content = make_pdf("User-uploaded paper reports retrieval evaluation.")
    store = LocalPaperStore(tmp_path)

    record, duplicate = store.save_pdf(content, "user-upload.pdf")
    processed = process_pdf(content, record.original_filename)
    store.update_status(record.paper_id, "ready", page_count=processed.page_count)
    restored = LocalPaperStore(tmp_path).restore()
    chunks = chunk_paper(restored.papers[record.paper_id])

    assert not duplicate
    assert record.paper_id == sha256(content).hexdigest()
    assert chunks[0].paper_id == record.paper_id
    assert chunks[0].paper_name == "user-upload.pdf"
    assert chunks[0].page_number == 1

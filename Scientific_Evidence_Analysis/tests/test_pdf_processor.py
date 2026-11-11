import pymupdf
import pytest

from modules.pdf_processor import PDFExtractionError, process_pdf
from modules.text_chunker import chunk_paper


def make_pdf(text: str | None) -> bytes:
    document = pymupdf.open()
    page = document.new_page()
    if text:
        page.insert_text((72, 72), text)
    document.set_metadata({"title": "Test Paper", "author": "Researcher"})
    data = document.tobytes()
    document.close()
    return data


def test_extracts_metadata_text_and_page_number():
    paper = process_pdf(
        make_pdf("Intervention improved outcome in the study."),
        "study.pdf",
    )
    assert paper.title == "Test Paper"
    assert paper.author == "Researcher"
    assert paper.page_count == 1
    assert paper.pages[0].page_number == 1
    assert "improved outcome" in paper.extracted_text
    chunks = chunk_paper(paper)
    assert chunks[0].paper_name == "study.pdf"
    assert chunks[0].page_number == 1


def test_rejects_empty_or_scanned_pdf():
    with pytest.raises(PDFExtractionError, match="empty"):
        process_pdf(b"", "empty.pdf")
    with pytest.raises(PDFExtractionError, match="scanned"):
        process_pdf(make_pdf(None), "scan.pdf")


def test_rejects_invalid_pdf():
    with pytest.raises(PDFExtractionError, match="readable PDF"):
        process_pdf(b"not a pdf", "invalid.pdf")

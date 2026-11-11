"""PDF text extraction with page-level provenance."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import pymupdf


class PDFExtractionError(ValueError):
    """Raised when a PDF cannot provide usable text evidence."""


@dataclass(frozen=True)
class PaperPage:
    paper_id: str
    paper_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class ProcessedPaper:
    paper_id: str
    paper_name: str
    title: str
    author: str
    page_count: int
    pages: tuple[PaperPage, ...]
    extracted_text: str


def process_pdf(file_data: bytes, paper_name: str) -> ProcessedPaper:
    """Extract text and basic metadata from a PDF without writing it to disk."""
    if not file_data:
        raise PDFExtractionError("The uploaded file is empty.")

    paper_id = sha256(file_data).hexdigest()
    try:
        document = pymupdf.open(stream=file_data, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
        raise PDFExtractionError("This file is not a readable PDF.") from exc

    try:
        if document.is_encrypted:
            raise PDFExtractionError("This PDF is password-protected.")
        try:
            page_texts = [
                (page_number, page.get_text("text").strip())
                for page_number, page in enumerate(document, start=1)
            ]
        except (pymupdf.FileDataError, RuntimeError, ValueError) as exc:
            raise PDFExtractionError(
                "This PDF contains damaged or unreadable page data."
            ) from exc
        useful_pages = [
            PaperPage(paper_id, paper_name, number, text)
            for number, text in page_texts
            if text
        ]
        if not useful_pages:
            raise PDFExtractionError(
                "No selectable text was found. This may be a scanned PDF; "
                "OCR is not implemented, so it cannot be analyzed."
            )
        metadata = document.metadata or {}
        title = (metadata.get("title") or "").strip() or paper_name
        author = (metadata.get("author") or "").strip() or "Not available"
        return ProcessedPaper(
            paper_id=paper_id,
            paper_name=paper_name,
            title=title,
            author=author,
            page_count=document.page_count,
            pages=tuple(useful_pages),
            extracted_text="\n\n".join(page.text for page in useful_pages),
        )
    finally:
        document.close()

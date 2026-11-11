"""Streamlit dashboard for transparent, context-aware scientific evidence analysis."""

from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from modules.claim_verification import (
    ClassifiedEvidence,
    EvidenceLabel,
    classify_evidence,
    verify_claim,
)
from modules.context_extraction import extract_context
from modules.disagreement_analysis import compare_evidence
from modules.embedding_engine import EmbeddingEngine
from modules.evidence_retrieval import retrieve_evidence
from modules.evidence_synthesis import generate_synthesis
from modules.pdf_processor import PDFExtractionError, ProcessedPaper, process_pdf
from modules.paper_storage import LocalPaperStore, StoredPaper
from modules.report_export import create_report
from modules.text_chunker import EvidenceChunk, chunk_paper

PROJECT_TITLE = (
    "A Multimodal Context-Aware Framework for Scientific Evidence Alignment, "
    "Claim Verification, and Disagreement-Aware Synthesis"
)

st.set_page_config(
    page_title="Scientific Evidence Analysis",
    page_icon="📚",
    layout="wide",
)


def initialize_state() -> None:
    if "paper_store" not in st.session_state:
        st.session_state["paper_store"] = LocalPaperStore()
    defaults = {
        "papers": {},
        "paper_chunks": {},
        "last_retrieval": [],
        "last_verified": [],
        "last_query": "",
        "embedding_engine": EmbeddingEngine(),
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if not st.session_state.get("paper_storage_initialized"):
        restored = st.session_state.paper_store.restore()
        st.session_state.papers = restored.papers
        st.session_state.paper_records = restored.records
        st.session_state.storage_warnings = restored.warnings
        st.session_state.paper_storage_initialized = True


def refresh_paper_records() -> list[StoredPaper]:
    records = st.session_state.paper_store.list_records()
    st.session_state.paper_records = records
    return records


def all_chunks() -> list[EvidenceChunk]:
    papers: dict[str, ProcessedPaper] = st.session_state.papers
    cached: dict[str, list[EvidenceChunk]] = st.session_state.paper_chunks
    for paper_id in list(cached):
        if paper_id not in papers:
            del cached[paper_id]
    for paper_id, paper in papers.items():
        if paper_id not in cached:
            cached[paper_id] = chunk_paper(paper)
    return [
        chunk
        for paper_id in papers
        for chunk in cached[paper_id]
    ]


def page_reference(chunk: EvidenceChunk) -> str:
    return f"{chunk.paper_name} — page {chunk.page_number}"


def show_evidence(item, label: str | None = None) -> None:
    chunk = item.evidence.chunk if isinstance(item, ClassifiedEvidence) else item.chunk
    score = item.evidence.score if isinstance(item, ClassifiedEvidence) else item.score
    heading = label or page_reference(chunk)
    with st.expander(f"{heading} · similarity {score:.3f}", expanded=False):
        st.caption(f"Source: {page_reference(chunk)}")
        st.write(chunk.text)
        if isinstance(item, ClassifiedEvidence):
            st.caption(f"Preliminary classification: {item.label.value}. {item.reason}")


def show_classifications(items: list[ClassifiedEvidence]) -> None:
    counts = Counter(item.label for item in items)
    columns = st.columns(3)
    for column, label in zip(columns, EvidenceLabel):
        column.metric(label.value, counts[label])
    st.warning(
        "Classifications are preliminary keyword-based signals, not proof of truth "
        "or contradiction. Similarity indicates topical relevance only; verify each "
        "passage in its original paper."
    )
    if any("research question" in item.reason.casefold() for item in items):
        st.info(
            "This is phrased as a research question. Retrieved passages are shown "
            "as relevant findings without forcing supporting or contradicting labels."
        )
    for label in EvidenceLabel:
        st.subheader(label.value)
        group = [item for item in items if item.label == label]
        if not group:
            st.info("No relevant evidence was found in this category.")
        for item in group:
            show_evidence(item)


def dashboard(chunks: list[EvidenceChunk]) -> None:
    st.title("Scientific Evidence Analysis")
    st.markdown(f"### {PROJECT_TITLE}")
    st.write(
        "A local research dashboard for retrieving traceable evidence, reviewing "
        "preliminary claim signals, and checking whether study context may explain "
        "different findings."
    )
    a, b, c = st.columns(3)
    a.metric("Uploaded papers", len(st.session_state.papers))
    b.metric("Extracted passages", len(chunks))
    c.metric("Similarity backend", st.session_state.embedding_engine.backend)
    st.subheader("Quick navigation")
    st.markdown(
        "- **Upload Research Papers:** add and inspect text-based PDF papers.\n"
        "- **Evidence Retrieval:** search passages across all uploaded papers.\n"
        "- **Claim Verification:** group retrieved passages by preliminary cues.\n"
        "- **Disagreement Analysis:** compare two papers and their extracted contexts.\n"
        "- **Evidence Synthesis:** produce and download a cited Markdown report."
    )
    st.caption(
        "Original PDFs and metadata are stored locally in the project’s data folder. "
        "Files are not sent to an external service."
    )


def upload_page() -> None:
    st.title("Upload Research Papers")
    st.write(
        "Select text-based PDFs. Original files and metadata are saved locally under "
        "`data/uploads/` and `data/papers.json`. "
        "Password-protected, unreadable, or scanned/image-only files are not treated "
        "as successfully extracted."
    )
    uploads = st.file_uploader(
        "Choose one or more scientific PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_upload",
    )
    if st.button("Process selected PDFs", type="primary", disabled=not uploads):
        for uploaded in uploads or []:
            try:
                record, duplicate = st.session_state.paper_store.save_pdf(
                    uploaded.getvalue(),
                    uploaded.name,
                )
            except (OSError, ValueError) as exc:
                st.error(f"{uploaded.name}: could not save the PDF locally: {exc}")
                continue
            if duplicate and record.paper_id in st.session_state.papers:
                paper = st.session_state.papers[record.paper_id]
                if record.processing_status != "ready":
                    st.session_state.paper_store.update_status(
                        record.paper_id,
                        "ready",
                        page_count=paper.page_count,
                    )
                    refresh_paper_records()
                st.info(f"{uploaded.name}: this PDF is already in the collection.")
                continue
            try:
                processed = process_pdf(
                    uploaded.getvalue(),
                    record.original_filename,
                )
            except PDFExtractionError as exc:
                st.session_state.paper_store.update_status(
                    record.paper_id,
                    "failed",
                    error=str(exc),
                )
                refresh_paper_records()
                st.error(f"{uploaded.name}: {exc}")
                continue
            st.session_state.paper_store.update_status(
                record.paper_id,
                "ready",
                page_count=processed.page_count,
            )
            st.session_state.papers[processed.paper_id] = processed
            st.session_state.paper_chunks.pop(processed.paper_id, None)
            refresh_paper_records()
            message = (
                f"Restored {uploaded.name} from local storage."
                if duplicate
                else f"Extracted {len(processed.pages)} text-bearing page(s) "
                     f"from {uploaded.name}."
            )
            st.success(message)

    records: list[StoredPaper] = st.session_state.paper_records
    if not records:
        st.info("No papers are currently loaded.")
        return

    st.subheader("Paper collection")
    st.dataframe(
        [
            {
                "Original filename": record.original_filename,
                "Pages": record.page_count or "Not available",
                "Processing status": record.processing_status.title(),
                "Uploaded": record.upload_date,
                "Issue": record.error or "",
            }
            for record in records
        ],
        hide_index=True,
        width="stretch",
    )
    for record in records:
        paper = st.session_state.papers.get(record.paper_id)
        if paper is None:
            continue
        with st.expander(f"{record.original_filename} · {record.page_count} pages"):
            first, second = st.columns(2)
            first.write(f"**Title:** {paper.title}")
            second.write(f"**Author:** {paper.author}")
            st.write(f"**Text-bearing pages:** {len(paper.pages)}")
            st.text_area(
                "Extracted text preview",
                paper.extracted_text,
                height=220,
                key=f"preview_{record.paper_id}",
                disabled=True,
            )
            st.caption(
                "PDF text extraction preserves page references. Extractable text "
                "tables/captions may appear in the preview; figures are not visually analyzed."
            )
    names = {
        f"{record.original_filename} ({record.paper_id[:8]})": record.paper_id
        for record in records
    }
    selected = st.multiselect("Select papers to remove", list(names))
    if selected and st.button("Remove selected papers"):
        removed, errors = st.session_state.paper_store.remove_papers(
            [names[name] for name in selected]
        )
        for paper_id in removed:
            st.session_state.papers.pop(paper_id, None)
            st.session_state.paper_chunks.pop(paper_id, None)
        for error in errors:
            st.error(error)
        refresh_paper_records()
        if removed:
            st.session_state.last_retrieval = []
            st.session_state.last_verified = []
            st.rerun()

    confirm_clear = st.checkbox(
        "I understand this permanently removes every locally stored paper."
    )
    if st.button(
        "Clear complete paper collection",
        disabled=not confirm_clear,
    ):
        removed, errors = st.session_state.paper_store.remove_papers(
            [record.paper_id for record in records]
        )
        for paper_id in removed:
            st.session_state.papers.pop(paper_id, None)
            st.session_state.paper_chunks.pop(paper_id, None)
        for error in errors:
            st.error(error)
        refresh_paper_records()
        if removed:
            st.session_state.last_retrieval = []
            st.session_state.last_verified = []
            st.rerun()


def retrieval_page(chunks: list[EvidenceChunk]) -> None:
    st.title("Evidence Retrieval")
    query = st.text_area(
        "Scientific claim or research question",
        value=st.session_state.last_query,
        placeholder="For example: Does intervention X improve outcome Y?",
    )
    top_k = st.slider("Maximum passages", min_value=1, max_value=30, value=10)
    if st.button("Search all papers", type="primary"):
        if not chunks:
            st.warning("Upload at least one paper before searching.")
        elif not query.strip():
            st.warning("Enter a claim or question to search.")
        else:
            st.session_state.last_query = query.strip()
            st.session_state.last_retrieval = retrieve_evidence(
                query,
                chunks,
                top_k=top_k,
                engine=st.session_state.embedding_engine,
            )
    if st.session_state.last_retrieval:
        st.caption(f"Similarity backend: {st.session_state.embedding_engine.backend}")
        st.info(
            "Similarity ranks topical relevance; it is not evidence that a claim is true."
        )
        for item in st.session_state.last_retrieval:
            show_evidence(item)
    elif chunks:
        st.info("Enter a question and search to view relevant passages.")
    else:
        st.info("No papers are loaded. Add papers from Upload Research Papers.")


def verification_page(chunks: list[EvidenceChunk]) -> None:
    st.title("Claim Verification")
    claim = st.text_area(
        "Scientific claim",
        value=st.session_state.last_query,
        placeholder="State a claim whose evidence you want to inspect.",
    )
    top_k = st.slider("Passages to inspect", 1, 30, 15, key="verify_top_k")
    if st.button("Retrieve and classify evidence", type="primary"):
        if not chunks:
            st.warning("Upload at least one paper before verifying a claim.")
        elif not claim.strip():
            st.warning("Enter a scientific claim.")
        else:
            retrieved = retrieve_evidence(
                claim,
                chunks,
                top_k=top_k,
                engine=st.session_state.embedding_engine,
            )
            st.session_state.last_query = claim.strip()
            st.session_state.last_verified = [
                classify_evidence(claim, item) for item in retrieved
            ]
    if st.session_state.last_verified:
        show_classifications(st.session_state.last_verified)
    elif chunks:
        st.info("Enter a claim to retrieve balanced evidence groups.")
    else:
        st.info("No papers are loaded. Add papers from Upload Research Papers.")


def disagreement_page(chunks: list[EvidenceChunk]) -> None:
    st.title("Context-Aware Disagreement Analysis")
    st.write(
        "Compare selected passages from different papers. Context fields are "
        "pattern-extracted clues, not a substitute for full-methods review."
    )
    if len({chunk.paper_id for chunk in chunks}) < 2:
        st.info("Upload at least two papers to compare evidence across studies.")
        return
    claim = st.text_input(
        "Optional retrieval topic",
        value=st.session_state.last_query,
        placeholder="Search term or claim to narrow candidate passages",
    )
    candidates = (
        retrieve_evidence(
            claim,
            chunks,
            top_k=30,
            engine=st.session_state.embedding_engine,
        )
        if claim.strip()
        else []
    )
    candidate_chunks = [item.chunk for item in candidates] if candidates else chunks
    options = {
        f"{chunk.paper_name} · p. {chunk.page_number} · {chunk.text[:100]}": chunk
        for chunk in candidate_chunks
    }
    labels = list(options)
    first_label = st.selectbox("First evidence passage", labels, key="compare_first")
    eligible_second = [
        label for label in labels
        if options[label].paper_id != options[first_label].paper_id
    ]
    if not eligible_second:
        st.warning(
            "The current candidates contain passages from only one paper. "
            "Try a broader/blank topic or upload another paper."
        )
        return
    second_label = st.selectbox(
        "Second evidence passage", eligible_second, key="compare_second"
    )
    if st.button("Compare study context", type="primary"):
        result = compare_evidence(options[first_label], options[second_label])
        st.subheader(result.category.value)
        st.write(result.reason)
        if result.shared_fields:
            st.write("**Fields with clues in both passages:** " + ", ".join(result.shared_fields))
        else:
            st.write("**Fields with clues in both passages:** none")
        if result.differing_fields:
            st.write("**Differing context fields:** " + ", ".join(result.differing_fields))
        else:
            st.write("**Differing context fields:** none identified")
        left, right = st.columns(2)
        for column, title, context, chunk in (
            (left, "First study passage", result.first_context, options[first_label]),
            (right, "Second study passage", result.second_context, options[second_label]),
        ):
            with column:
                st.markdown(f"#### {title}")
                st.caption(page_reference(chunk))
                st.write(chunk.text)
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Context field": key,
                                "Extracted clue": value or "Not identified",
                                "Source paper/page": (
                                    page_reference(chunk) if value else "Not available"
                                ),
                            }
                         for key, value in context.items()]
                    ),
                    hide_index=True,
                    width="stretch",
                )
        st.warning(
            "A potential genuine contradiction is only a comparison signal. "
            "Study design, uncertainty, and full-paper context require human review."
        )


def synthesis_page() -> None:
    st.title("Evidence Synthesis")
    st.write(
        "Generate a structured Markdown report using passages from the most recent "
        "claim-verification run. Search/re-run verification after changing the claim."
    )
    items: list[ClassifiedEvidence] = st.session_state.last_verified
    claim = st.text_input(
        "Claim for synthesis",
        value=st.session_state.last_query,
        key="synthesis_claim",
    )
    if not items:
        st.info("Run Claim Verification to retrieve cited passages for a synthesis.")
        return
    st.caption(f"{len(items)} retrieved passages are available for this report.")
    chunks = all_chunks()
    report = generate_synthesis(claim, items, chunks)
    st.markdown(report)
    st.download_button(
        "Download evidence report (.md)",
        data=create_report(claim, items, chunks),
        file_name="scientific_evidence_synthesis.md",
        mime="text/markdown",
    )


def about_page() -> None:
    st.title("About the Project")
    st.markdown(
        f"""
### {PROJECT_TITLE}

**Objective:** Help users inspect evidence passages across scientific papers,
consider study context, and produce traceable preliminary summaries.

**Architecture:** Streamlit UI → PyMuPDF page extraction → sentence chunking
with page provenance → cached CPU Sentence Transformers embeddings when available
(otherwise scikit-learn TF-IDF) → explainable rule-based evidence grouping →
context comparison → Markdown report export.

**Methods:** Retrieval uses cosine similarity. Claim grouping uses overlapping
content terms and simple directional keywords; disagreement analysis requires
opposing textual cues and examines extracted context fields. These heuristics
do not establish entailment or scientific truth.

**Current limitations:** OCR for scanned PDFs, image/chart interpretation,
validated natural-language inference, and deep metadata extraction are not
implemented. Original PDFs and metadata are persisted locally; extracted page
text and chunks are regenerated from those files at startup. Extractable text
from tables and captions is retained as ordinary page text. Only a cached model
is loaded automatically; otherwise TF-IDF works offline.

**Future improvements:** Add optional user-controlled OCR, evaluation on reviewed
scientific examples, improved table structure parsing, richer context review, and
human-annotated validation of disagreement categories.
"""
    )


def main() -> None:
    initialize_state()
    chunks = all_chunks()
    for warning in st.session_state.storage_warnings:
        st.warning(warning)
    with st.sidebar:
        st.title("Research Dashboard")
        st.caption("Local evidence review prototype")
        selection = st.radio(
            "Navigation",
            [
                "Dashboard",
                "Upload Research Papers",
                "Evidence Retrieval",
                "Claim Verification",
                "Disagreement Analysis",
                "Evidence Synthesis",
                "About the Project",
            ],
        )
        st.divider()
        st.metric("Papers", len(st.session_state.papers))
        st.metric("Passages", len(chunks))
    pages = {
        "Dashboard": lambda: dashboard(chunks),
        "Upload Research Papers": upload_page,
        "Evidence Retrieval": lambda: retrieval_page(chunks),
        "Claim Verification": lambda: verification_page(chunks),
        "Disagreement Analysis": lambda: disagreement_page(chunks),
        "Evidence Synthesis": synthesis_page,
        "About the Project": about_page,
    }
    pages[selection]()


if __name__ == "__main__":
    main()

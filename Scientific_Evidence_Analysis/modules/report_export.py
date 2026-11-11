"""Export an evidence synthesis as a downloadable Markdown report."""

from modules.evidence_synthesis import generate_synthesis
from modules.text_chunker import EvidenceChunk


def create_report(
    claim: str,
    classified_items: list,
    all_chunks: list[EvidenceChunk],
) -> bytes:
    """Return UTF-8 Markdown; all evidence citations come from source chunks."""
    return generate_synthesis(claim, classified_items, all_chunks).encode("utf-8")

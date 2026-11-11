from modules.pdf_processor import PaperPage
from modules.text_chunker import chunk_pages


def test_long_sentences_are_split_into_bounded_chunks_with_page_provenance():
    page = PaperPage(
        "paper-id",
        "source.pdf",
        7,
        " ".join(f"term{index}" for index in range(25)),
    )
    chunks = chunk_pages([page], max_words=8, overlap_sentences=0)
    assert len(chunks) == 4
    assert all(len(chunk.text.split()) <= 8 for chunk in chunks)
    assert all(chunk.page_number == 7 for chunk in chunks)

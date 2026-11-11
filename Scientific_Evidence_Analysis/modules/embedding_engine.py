"""CPU-only text similarity with an optional cached Sentence Transformers model."""

from __future__ import annotations

import logging
import re
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingEngine:
    """Use a locally cached MiniLM model when available, otherwise TF-IDF."""

    def __init__(self) -> None:
        self._model = None
        self._model_checked = False
        self._indexed_passages: tuple[str, ...] | None = None
        self._vectorizer: TfidfVectorizer | None = None
        self._tfidf_matrix = None
        self._passage_embeddings = None

    @property
    def index_ready(self) -> bool:
        return self._indexed_passages is not None

    def invalidate_index(self) -> None:
        """Discard the corpus index after papers are added or removed."""
        self._indexed_passages = None
        self._vectorizer = None
        self._tfidf_matrix = None
        self._passage_embeddings = None

    @property
    def backend(self) -> str:
        self._load_cached_model()
        return "Sentence Transformers (CPU)" if self._model is not None else "TF-IDF"

    def _load_cached_model(self) -> None:
        if self._model_checked:
            return
        self._model_checked = True
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                MODEL_NAME,
                device="cpu",
                local_files_only=True,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as exc:
            logger.info("Cached embedding model unavailable; using TF-IDF: %s", exc)
            self._model = None

    def similarities(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        self._load_cached_model()
        corpus = tuple(passages)
        if corpus != self._indexed_passages:
            self._index_passages(corpus)
        if self._model is not None:
            query_embedding = self._model.encode(
                [query],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return cosine_similarity(
                query_embedding,
                self._passage_embeddings,
            )[0].tolist()
        if self._vectorizer is None or self._tfidf_matrix is None:
            return [0.0] * len(passages)
        query_vector = self._vectorizer.transform([query])
        return cosine_similarity(query_vector, self._tfidf_matrix)[0].tolist()

    def _index_passages(self, passages: tuple[str, ...]) -> None:
        """Fit word unigram and bigram features across the searchable passages."""
        vectorizer = None
        tfidf_matrix = None
        passage_embeddings = None
        if self._model is not None:
            passage_embeddings = self._model.encode(
                list(passages),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            )
            try:
                tfidf_matrix = vectorizer.fit_transform(passages)
            except ValueError:
                vectorizer = None
        self._vectorizer = vectorizer
        self._tfidf_matrix = tfidf_matrix
        self._passage_embeddings = passage_embeddings
        self._indexed_passages = passages


def meaningful_terms(text: str) -> set[str]:
    """Return normalized, nontrivial tokens for explainable evidence heuristics."""
    stop_words = {
        "about", "after", "again", "against", "among", "also", "and", "are",
        "because", "been", "before", "being", "between", "both", "but", "can",
        "could", "did", "does", "during", "each", "for", "from", "further",
        "had", "has", "have", "having", "into", "its", "may", "might", "more",
        "most", "not", "our", "over", "same", "should", "some", "such", "than",
        "that", "the", "their", "then", "there", "these", "they", "this", "those",
        "through", "under", "until", "very", "was", "were", "what", "when",
        "where", "which", "while", "who", "will", "with", "would", "your",
        "study", "research", "paper", "results", "result", "using", "used",
    }
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop_words
    }

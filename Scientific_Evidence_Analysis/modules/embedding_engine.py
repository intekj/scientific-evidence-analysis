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
        if self._model is not None:
            embeddings = self._model.encode(
                [query, *passages],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return cosine_similarity(embeddings[:1], embeddings[1:])[0].tolist()
        return self._tfidf_similarities(query, passages)

    @staticmethod
    def _tfidf_similarities(query: str, passages: Sequence[str]) -> list[float]:
        """Use word and character features so short scientific terms still match."""
        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                token_pattern=r"(?u)\b\w+\b",
            )
            matrix = vectorizer.fit_transform([query, *passages])
        except ValueError:
            return [0.0] * len(passages)
        return cosine_similarity(matrix[:1], matrix[1:])[0].tolist()


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

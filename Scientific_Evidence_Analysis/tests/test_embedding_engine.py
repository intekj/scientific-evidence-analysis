from modules import embedding_engine
from modules.embedding_engine import EmbeddingEngine


def test_tfidf_index_is_reused_and_refit_for_a_changed_corpus(monkeypatch):
    fit_calls = 0
    transform_calls = 0
    original_fit_transform = embedding_engine.TfidfVectorizer.fit_transform
    original_transform = embedding_engine.TfidfVectorizer.transform

    def counted_fit_transform(vectorizer, documents):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit_transform(vectorizer, documents)

    def counted_transform(vectorizer, documents):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(vectorizer, documents)

    monkeypatch.setattr(
        embedding_engine.TfidfVectorizer,
        "fit_transform",
        counted_fit_transform,
    )
    monkeypatch.setattr(
        embedding_engine.TfidfVectorizer,
        "transform",
        counted_transform,
    )
    engine = EmbeddingEngine()
    engine._model_checked = True
    passages = ["A study reports improved outcomes.", "A study reports no change."]

    first = engine.similarities("improved outcomes", passages)
    second = engine.similarities("no change", passages)

    assert first[0] > first[1]
    assert second[1] > second[0]
    assert fit_calls == 1
    assert transform_calls == 2

    engine.similarities("new result", passages + ["A new result was reported."])
    assert fit_calls == 2
    engine.invalidate_index()
    engine.similarities("same papers after invalidation", passages)
    assert fit_calls == 3


def test_sentence_embeddings_are_cached_for_passages_not_queries():
    class FakeModel:
        def __init__(self):
            self.encoded = []

        def encode(self, texts, **kwargs):
            self.encoded.append(texts)
            return [
                [1.0, 0.0] if text != "different" else [0.0, 1.0]
                for text in texts
            ]

    model = FakeModel()
    engine = EmbeddingEngine()
    engine._model = model
    engine._model_checked = True
    passages = ["paper finding", "different"]

    engine.similarities("first query", passages)
    engine.similarities("second query", passages)

    assert model.encoded == [
        passages,
        ["first query"],
        ["second query"],
    ]

    engine.invalidate_index()
    engine.similarities("after paper change", passages)
    assert model.encoded[-2:] == [passages, ["after paper change"]]

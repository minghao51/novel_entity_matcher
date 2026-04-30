import numpy as np
import pytest


class _MockModel2Vec:
    dim = 256

    def __init__(self, model_name=None):
        pass

    @classmethod
    def from_pretrained(cls, model_name):
        return cls()

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), self.dim)


@pytest.mark.smoke
def test_static_embedding_backend_model2vec_path(monkeypatch):
    monkeypatch.setattr("model2vec.StaticModel", _MockModel2Vec)

    from novelentitymatcher.backends.static_embedding import StaticEmbeddingBackend

    backend = StaticEmbeddingBackend("minishlab/potion-base-8M")
    embeddings = backend.encode(["hello world"])

    assert len(embeddings) == 1
    assert len(embeddings[0]) == 256


@pytest.mark.smoke
def test_static_embedding_backend_sentence_transformer_path(monkeypatch):
    monkeypatch.setattr(
        "model2vec.StaticModel.from_pretrained",
        lambda name: (_ for _ in ()).throw(RuntimeError("fall through to ST")),
    )

    class FakeSentenceTransformer:
        def get_sentence_embedding_dimension(self):
            return 384

        def encode(self, texts, batch_size=32, show_progress_bar=False):
            if isinstance(texts, str):
                texts = [texts]
            return np.random.rand(len(texts), 384)

        def modules(self):
            return []

    import novelentitymatcher.backends.static_embedding as se

    original = se.get_cached_sentence_transformer
    se.get_cached_sentence_transformer = lambda name, trust_remote_code=False: (
        FakeSentenceTransformer()
    )

    try:
        from novelentitymatcher.backends.static_embedding import StaticEmbeddingBackend

        backend = StaticEmbeddingBackend("RikkaBotan/test-model")
        embeddings = backend.encode(["test query"])

        assert len(embeddings) == 1
        assert len(embeddings[0]) == 384
    finally:
        se.get_cached_sentence_transformer = original

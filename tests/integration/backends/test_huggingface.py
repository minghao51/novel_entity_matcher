import numpy as np
import pytest
import torch

from novelentitymatcher.backends.sentencetransformer import HFEmbedding, HFReranker

pytestmark = [pytest.mark.hf]


class TestHFEmbedding:
    """Tests for HuggingFace SentenceTransformer embedding backend."""

    def test_hf_embedding_init(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        assert backend.model is not None

    def test_hf_embedding_encode(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = backend.encode(["hello world"])
        assert isinstance(embeddings, list)
        assert len(embeddings) == 1
        assert len(embeddings[0]) > 0

    def test_hf_embedding_encode_multiple(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = backend.encode(["hello", "world", "foo"])
        assert len(embeddings) == 3

    def test_hf_embedding_similarity(self):
        backend = HFEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        embeddings = backend.encode(["cat", "dog"])
        results = backend.similarity(embeddings[:1], embeddings, top_k=2)
        assert len(results) == 1
        assert len(results[0]) == 2

    def test_hf_embedding_similarity_normalizes_input_types(self, monkeypatch):
        backend = object.__new__(HFEmbedding)
        calls = {}

        def fake_semantic_search(query_embeddings, corpus_embeddings, top_k):
            calls["query_type"] = type(query_embeddings)
            calls["corpus_type"] = type(corpus_embeddings)
            calls["query_shape"] = query_embeddings.shape
            calls["corpus_shape"] = corpus_embeddings.shape
            calls["top_k"] = top_k
            return [[{"corpus_id": 0, "score": 1.0}]]

        monkeypatch.setattr(
            "novelentitymatcher.backends.sentencetransformer.semantic_search",
            fake_semantic_search,
        )

        results = backend.similarity(
            torch.tensor([[1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            top_k=1,
        )

        assert results == [[{"corpus_id": 0, "score": 1.0}]]
        assert calls["query_type"] is np.ndarray
        assert calls["corpus_type"] is np.ndarray
        assert calls["query_shape"] == (1, 2)
        assert calls["corpus_shape"] == (2, 2)
        assert calls["top_k"] == 1


class TestHFReranker:
    """Tests for HuggingFace CrossEncoder reranker backend."""

    def test_hf_reranker_init(self):
        backend = HFReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert backend.model is not None

    def test_hf_reranker_score(self):
        backend = HFReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = backend.score(
            "What is the capital of France?", ["Paris", "London", "Berlin"]
        )
        assert isinstance(scores, list)
        assert len(scores) == 3
        assert all(isinstance(s, (int, float)) for s in scores)

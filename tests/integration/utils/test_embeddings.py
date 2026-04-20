import pytest
import numpy as np
from novelentitymatcher.utils.embeddings import (
    compute_embeddings,
    cosine_sim,
    batch_encode,
)


class TestEmbeddings:
    """Tests for embedding utilities."""

    def test_compute_embeddings_basic(self):
        texts = ["hello world", "foo bar"]
        embeddings = compute_embeddings(texts)
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0

    def test_compute_embeddings_single(self):
        text = "hello"
        embedding = compute_embeddings([text])
        assert embedding.shape[0] == 1

    def test_cosine_sim_same(self):
        a = np.array([1.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_sim(a, b) == pytest.approx(1.0)

    def test_cosine_sim_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_sim(a, b) == pytest.approx(-1.0)

    def test_cosine_sim_perpendicular(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_sim(a, b) == pytest.approx(0.0)

    def test_batch_encode(self):
        texts = ["a", "b", "c", "d", "e"] * 10
        batches = list(batch_encode(texts, batch_size=10))
        assert len(batches) == 5

    def test_batch_encode_small(self):
        texts = ["a", "b"]
        batches = list(batch_encode(texts, batch_size=5))
        assert len(batches) == 1

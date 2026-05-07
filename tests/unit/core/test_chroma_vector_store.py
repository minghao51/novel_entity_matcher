"""Tests for ChromaVectorStore."""

import numpy as np
import pytest

from novelentitymatcher.core.vector_store import ChromaVectorStore


class TestChromaVectorStore:
    @pytest.fixture
    def sample_vectors(self):
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.5, 0.5, 0.0],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def store(self, sample_vectors):
        s = ChromaVectorStore()
        s.upsert(
            ids=["a", "b", "c", "d"],
            vectors=sample_vectors,
            metadata=[
                {"type": "x", "priority": 1},
                {"type": "y", "priority": 2},
                {"type": "x", "priority": 3},
                {"type": "z", "priority": 1},
            ],
        )
        return s

    def test_upsert_and_query_round_trip(self, store):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"
        assert results[0]["score"] == pytest.approx(1.0, abs=0.01)

    def test_query_returns_multiple_results(self, store):
        query = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        results = store.query(query, top_k=2)
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "d" in ids

    def test_query_with_metadata_filter(self, store):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4, filter={"type": "x"})
        assert len(results) >= 1
        assert all(r["metadata"]["type"] == "x" for r in results)

    def test_query_with_no_match_filter_returns_empty(self, store):
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4, filter={"type": "nonexistent"})
        assert results == []

    def test_count_returns_total_elements(self, store):
        assert store.count() == 4

    def test_delete_removes_from_query(self, store):
        store.delete(["a"])
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4)
        ids = [r["id"] for r in results]
        assert "a" not in ids

    def test_delete_reduces_count(self, store):
        assert store.count() == 4
        store.delete(["a", "b"])
        assert store.count() == 2

    def test_delete_idempotent(self, store):
        store.delete(["a"])
        store.delete(["a"])
        assert store.count() == 3
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4)
        ids = [r["id"] for r in results]
        assert "a" not in ids

    def test_query_includes_metadata_when_present(self, store):
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=1)
        assert "metadata" in results[0]
        assert results[0]["metadata"]["type"] == "y"

    def test_query_omits_metadata_when_absent(self):
        s = ChromaVectorStore(collection_name="test_no_metadata")
        s.upsert(
            ids=["only"],
            vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        results = s.query(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
        assert "metadata" not in results[0]

    def test_upsert_revives_deleted_id(self, store):
        store.delete(["a"])
        store.upsert(
            ids=["a"],
            vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            metadata=[{"type": "x", "priority": 9}],
        )
        results = store.query(np.array([1.0, 0.0, 0.0], dtype=np.float32), top_k=1)
        assert results[0]["id"] == "a"
        assert store.count() == 4

    def test_upsert_same_id_latest_wins(self):
        store = ChromaVectorStore(collection_name="test_latest_wins")
        store.upsert(ids=["a"], vectors=np.array([[1.0, 0.0]], dtype=np.float32))
        store.upsert(ids=["a"], vectors=np.array([[0.0, 1.0]], dtype=np.float32))

        results = store.query(np.array([0.0, 1.0], dtype=np.float32), top_k=5)
        ids = [row["id"] for row in results]
        assert ids == ["a"]
        assert store.count() == 1

    def test_dim_property_none_before_upsert(self):
        store = ChromaVectorStore()
        assert store.dim is None

    def test_dim_property_after_upsert(self):
        store = ChromaVectorStore()
        store.upsert(
            ids=["x"],
            vectors=np.array([[0.1, 0.2, 0.3]], dtype=np.float32),
        )
        assert store.dim == 3

    def test_persist_directory(self, tmp_path):
        persist = str(tmp_path / "chroma_test")
        store1 = ChromaVectorStore(persist_directory=persist)
        store1.upsert(
            ids=["persist_me"],
            vectors=np.array([[0.5, 0.5]], dtype=np.float32),
            metadata=[{"key": "val"}],
        )
        assert store1.count() == 1

        store2 = ChromaVectorStore(persist_directory=persist)
        assert store2.count() == 1
        results = store2.query(np.array([0.5, 0.5], dtype=np.float32), top_k=1)
        assert results[0]["id"] == "persist_me"

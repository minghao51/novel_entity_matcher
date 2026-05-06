"""Tests for VectorStore protocol and InMemoryVectorStore."""

import numpy as np
import pytest

from novelentitymatcher.core.vector_store import InMemoryVectorStore


class TestInMemoryVectorStore:
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
        s = InMemoryVectorStore(dim=3, backend="exact")
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
        """Basic upsert followed by query should return the closest vectors."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "a"
        assert results[0]["score"] == pytest.approx(1.0, abs=0.01)

    def test_query_returns_multiple_results(self, store):
        """Query with top_k > 1 should return ordered results."""
        query = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        results = store.query(query, top_k=2)
        assert len(results) == 2
        ids = [r["id"] for r in results]
        assert "d" in ids

    def test_query_with_metadata_filter(self, store):
        """Filtering should exclude non-matching results."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4, filter={"type": "x"})
        assert len(results) == 2
        assert {r["id"] for r in results} == {"a", "c"}

    def test_query_with_no_match_filter_returns_empty(self, store):
        """A filter that matches nothing should return an empty list."""
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4, filter={"type": "nonexistent"})
        assert results == []

    def test_count_returns_total_elements(self, store):
        """count should reflect the number of indexed vectors."""
        assert store.count() == 4

    def test_delete_removes_from_query(self, store):
        """After delete, the deleted ID should not appear in query results."""
        store.delete(["a"])
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4)
        ids = [r["id"] for r in results]
        assert "a" not in ids

    def test_delete_reduces_count(self, store):
        """count should decrease after deletion."""
        assert store.count() == 4
        store.delete(["a", "b"])
        assert store.count() == 2

    def test_delete_idempotent(self, store):
        """Deleting the same ID twice should not raise or corrupt state."""
        store.delete(["a"])
        store.delete(["a"])
        assert store.count() == 3
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=4)
        ids = [r["id"] for r in results]
        assert "a" not in ids

    def test_query_returns_correct_top_k_after_deletions(self, store):
        """When deleted items would have been in top_k, query should backfill."""
        store.delete(["a"])
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=1)
        assert len(results) == 1
        assert results[0]["id"] != "a"

    def test_query_includes_metadata_when_present(self, store):
        """Results should include metadata for IDs that have it."""
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        results = store.query(query, top_k=1)
        assert "metadata" in results[0]
        assert results[0]["metadata"]["type"] == "y"

    def test_query_omits_metadata_when_absent(self, store):
        """Results should not include a metadata key if none was stored."""
        s = InMemoryVectorStore(dim=2, backend="exact")
        s.upsert(
            ids=["only"],
            vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        )
        results = s.query(np.array([1.0, 0.0], dtype=np.float32), top_k=1)
        assert "metadata" not in results[0]

    def test_dim_property_matches_constructor(self):
        """The dim property should reflect the constructor argument."""
        store = InMemoryVectorStore(dim=128, backend="exact")
        assert store.dim == 128

    def test_upsert_revives_deleted_id(self, store):
        """Upserting a deleted ID should make it queryable again."""
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
        """Repeated upserts should return only the latest vector for an ID."""
        store = InMemoryVectorStore(dim=2, backend="exact")
        store.upsert(ids=["a"], vectors=np.array([[1.0, 0.0]], dtype=np.float32))
        store.upsert(ids=["a"], vectors=np.array([[0.0, 1.0]], dtype=np.float32))

        results = store.query(np.array([0.0, 1.0], dtype=np.float32), top_k=5)
        ids = [row["id"] for row in results]
        assert ids == ["a"]
        assert store.count() == 1

"""Tests for EmbeddingMatcher.add_entities."""

import pytest

from novelentitymatcher.core.embedding_matcher import EmbeddingMatcher


def _make_entities(names, start=0):
    return [{"id": f"ent_{i}", "name": n} for i, n in enumerate(names, start=start)]


@pytest.fixture
def matcher():
    """A minimal EmbeddingMatcher with static embeddings (no model required)."""
    entities = _make_entities(["apple", "banana", "cherry"])
    m = EmbeddingMatcher(entities=entities, model_name="potion-32m")
    m.build_index()
    return m


class TestAddEntities:
    def test_add_entities_new_ids_in_index(self, matcher):
        new = _make_entities(["zzzzunique", "yyyyother"], start=3)
        matcher.add_entities(new)
        assert "ent_3" in matcher.entity_ids
        assert "ent_4" in matcher.entity_ids

    def test_add_entities_new_texts_in_index(self, matcher):
        new = _make_entities(["zzzzunique"], start=3)
        matcher.add_entities(new)
        assert "zzzzunique" in matcher.entity_texts

    def test_add_entities_increases_entity_count(self, matcher):
        assert len(matcher.entities) == 3
        matcher.add_entities(_make_entities(["date", "elderberry"], start=3))
        assert len(matcher.entities) == 5

    def test_add_entities_increases_embeddings_shape(self, matcher):
        old_shape = matcher.embeddings.shape
        matcher.add_entities(_make_entities(["date"], start=3))
        assert matcher.embeddings.shape[0] == old_shape[0] + 1
        assert matcher.embeddings.shape[1] == old_shape[1]

    def test_add_entities_empty_does_nothing(self, matcher):
        old_count = len(matcher.entities)
        old_emb_count = matcher.embeddings.shape[0]
        matcher.add_entities([])
        assert len(matcher.entities) == old_count
        assert matcher.embeddings.shape[0] == old_emb_count

    def test_add_entities_raises_before_build_index(self):
        entities = _make_entities(["test"])
        m = EmbeddingMatcher(entities=entities, model_name="potion-32m")
        with pytest.raises(RuntimeError, match="Index not built"):
            m.add_entities(_make_entities(["new"], start=1))

    def test_match_unchanged_for_existing(self, matcher):
        result_before = matcher.match("apple")
        matcher.add_entities(_make_entities(["new fruit"], start=3))
        result_after = matcher.match("apple")
        assert result_after["id"] == result_before["id"]
        assert result_after["score"] == pytest.approx(result_before["score"], abs=0.01)

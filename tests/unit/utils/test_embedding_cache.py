"""Tests for LRUEmbeddingCache."""

import numpy as np
import pytest

from novelentitymatcher.utils.embedding_cache import LRUEmbeddingCache


class TestLRUEmbeddingCache:
    def test_basic_hit_and_miss(self):
        """A stored embedding should be returned on subsequent get."""
        cache = LRUEmbeddingCache(max_entries=10)
        emb = np.array([1.0, 2.0, 3.0])
        assert cache.get("hello") is None
        cache.put("hello", emb)
        result = cache.get("hello")
        assert result is not None
        assert np.allclose(result, emb)

    def test_hit_and_miss_counts(self):
        """Stats should accurately reflect hits and misses."""
        cache = LRUEmbeddingCache(max_entries=10)
        cache.put("a", np.array([1.0]))
        cache.get("a")  # hit
        cache.get("b")  # miss
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        assert cache.stats["hit_rate"] == pytest.approx(0.5)

    def test_lru_eviction_at_capacity(self):
        """Least recently used item should be evicted when capacity exceeded."""
        cache = LRUEmbeddingCache(max_entries=3)
        cache.put("a", np.array([1.0]))
        cache.put("b", np.array([2.0]))
        cache.put("c", np.array([3.0]))
        # Access 'a' to make it most recently used
        cache.get("a")
        # Add 'd' — 'b' should be evicted (least recently used)
        cache.put("d", np.array([4.0]))
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None
        assert cache.get("d") is not None
        assert cache.size == 3

    def test_batch_api(self):
        """get_batch should return cached items and indices of uncached items."""
        cache = LRUEmbeddingCache(max_entries=10)
        cache.put("a", np.array([1.0]))
        cache.put("c", np.array([3.0]))
        results, uncached = cache.get_batch(["a", "b", "c"])
        assert results[0] is not None
        assert np.allclose(results[0], [1.0])
        assert results[1] is None
        assert results[2] is not None
        assert np.allclose(results[2], [3.0])
        assert uncached == [1]

    def test_put_batch(self):
        """put_batch should store all embeddings."""
        cache = LRUEmbeddingCache(max_entries=10)
        texts = ["a", "b", "c"]
        embeddings = np.array([[1.0], [2.0], [3.0]])
        cache.put_batch(texts, embeddings)
        assert cache.size == 3
        assert np.allclose(cache.get("b"), [2.0])

    def test_clear_resets_everything(self):
        """clear should empty the cache and zero stats."""
        cache = LRUEmbeddingCache(max_entries=10)
        cache.put("a", np.array([1.0]))
        cache.get("a")
        cache.clear()
        assert cache.size == 0
        assert cache.stats["hits"] == 0
        assert cache.stats["misses"] == 0
        assert cache.get("a") is None

    def test_key_prefix_isolation(self):
        """Caches with different prefixes should not share entries."""
        cache_a = LRUEmbeddingCache(max_entries=10, key_prefix="model_a")
        cache_b = LRUEmbeddingCache(max_entries=10, key_prefix="model_b")
        emb = np.array([1.0, 2.0])
        cache_a.put("hello", emb)
        assert cache_a.get("hello") is not None
        assert cache_b.get("hello") is None

    def test_key_prefix_uses_prefixed_key_internally(self):
        """The internal cache key should include the prefix."""
        cache = LRUEmbeddingCache(max_entries=10, key_prefix="shared")
        emb = np.array([1.0, 2.0])
        cache.put("hello", emb)
        assert "shared:hello" in cache._cache
        assert cache.get("hello") is not None

    def test_empty_prefix_no_colon_in_key(self):
        """When key_prefix is empty, the raw text should be used as key."""
        cache = LRUEmbeddingCache(max_entries=10, key_prefix="")
        emb = np.array([1.0])
        cache.put("hello", emb)
        assert cache.get("hello") is not None
        # Verify internal key has no leading colon
        assert "hello" in cache._cache
        assert ":hello" not in cache._cache

    def test_hit_moves_to_end(self):
        """Accessing an existing item should mark it as most recently used."""
        cache = LRUEmbeddingCache(max_entries=3)
        cache.put("a", np.array([1.0]))
        cache.put("b", np.array([2.0]))
        cache.put("c", np.array([3.0]))
        # Access 'a' to make it MRU
        cache.get("a")
        # Add 'd' — 'b' should be evicted
        cache.put("d", np.array([4.0]))
        assert cache.get("a") is not None
        assert cache.get("b") is None
        assert cache.get("c") is not None
        assert cache.get("d") is not None

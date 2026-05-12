"""Tests for ModelCache."""

from novelentitymatcher.utils.embeddings import ModelCache


class _FakeConfig:
    vocab_size = 1000
    hidden_size = 64
    intermediate_size = 128
    num_hidden_layers = 2


class _FakeModel:
    config = _FakeConfig()


def _factory() -> _FakeModel:
    return _FakeModel()


class TestModelCache:
    def test_clear_resets_memory_bookkeeping(self):
        cache = ModelCache(max_memory_gb=1.0)
        cache.get_or_load("model-a", _factory)

        assert cache._memory_bytes  # internal state sanity check before clear

        cache.clear()

        assert cache._memory_bytes == {}
        assert cache.stats()["size"] == 0
        assert cache.stats()["hits"] == 0
        assert cache.stats()["misses"] == 0

    def test_cache_remains_usable_after_clear(self):
        cache = ModelCache(max_memory_gb=1.0)
        cache.get_or_load("model-a", _factory)
        cache.clear()

        reloaded = cache.get_or_load("model-a", _factory)

        assert reloaded is not None
        assert cache.stats()["size"] == 1
        assert cache.stats()["misses"] == 1

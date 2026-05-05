from __future__ import annotations

from collections import OrderedDict

import numpy as np


class LRUEmbeddingCache:
    def __init__(
        self,
        max_entries: int = 10_000,
        dim: int | None = None,
        key_prefix: str = "",
    ):
        self.max_entries = max_entries
        self.dim = dim
        self.key_prefix = key_prefix
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        return f"{self.key_prefix}:{text}" if self.key_prefix else text

    def get(self, text: str) -> np.ndarray | None:
        key = self._make_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        key = self._make_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = embedding
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get_batch(self, texts: list[str]) -> tuple[list[np.ndarray | None], list[int]]:
        results: list[np.ndarray | None] = []
        uncached: list[int] = []
        for i, text in enumerate(texts):
            emb = self.get(text)
            results.append(emb)
            if emb is None:
                uncached.append(i)
        return results, uncached

    def put_batch(self, texts: list[str], embeddings: np.ndarray) -> None:
        for text, emb in zip(texts, embeddings, strict=False):
            self.put(text, emb)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def size(self) -> int:
        return len(self._cache)

    @property
    def stats(self) -> dict[str, int | float]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": self.hit_rate,
        }

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

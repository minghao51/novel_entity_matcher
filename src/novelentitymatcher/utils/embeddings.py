from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

__all__ = [
    "ModelCache",
    "batch_encode",
    "compute_embeddings",
    "cosine_sim",
    "get_cached_cross_encoder",
    "get_cached_sentence_transformer",
    "get_cached_setfit_model",
    "get_default_cache",
]


class ModelCache:
    """
    Configurable cache for SentenceTransformer models.

    Thread-safe LRU cache with memory-based eviction and optional TTL.
    """

    def __init__(
        self,
        max_memory_gb: float = 4.0,
        ttl_seconds: float | None = None,
    ):
        """
        Initialize the model cache.

        Args:
            max_memory_gb: Maximum memory to use for cached models (in GB).
            ttl_seconds: Optional time-to-live for cache entries in seconds.
        """
        self._cache: dict[str, Any] = {}
        self._access_times: dict[str, float] = {}
        self._max_memory_gb = max_memory_gb
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def get_or_load(self, model_name: str, factory: Callable[[], Any]) -> Any:
        """
        Get a model from cache or load it using the factory function.

        Args:
            model_name: Unique identifier for the model
            factory: Callable that returns the model instance

        Returns:
            The cached or newly created model
        """
        with self._lock:
            current_time = time.time()

            # Check if model is in cache and not expired
            if model_name in self._cache:
                if self._ttl_seconds is not None:
                    age = current_time - self._access_times.get(model_name, 0)
                    if age > self._ttl_seconds:
                        # Expired - remove from cache
                        del self._cache[model_name]
                        del self._access_times[model_name]
                    else:
                        self._hits += 1
                        self._access_times[model_name] = current_time
                        return self._cache[model_name]
                else:
                    self._hits += 1
                    self._access_times[model_name] = current_time
                    return self._cache[model_name]

            self._misses += 1

            # Load the model
            model = factory()

            # Add to cache
            self._cache[model_name] = model
            self._access_times[model_name] = current_time

            return model

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with hits, misses, size, hit_rate
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "size": len(self._cache),
                "hit_rate": hit_rate,
            }


# Global default cache instance (4GB limit)
_default_cache: ModelCache | None = None
_cache_lock = threading.Lock()


def get_default_cache() -> ModelCache:
    """Get or create the global default model cache."""
    global _default_cache
    with _cache_lock:
        if _default_cache is None:
            _default_cache = ModelCache(max_memory_gb=4.0)
        return _default_cache


def get_cached_sentence_transformer(
    model_name: str, cache: ModelCache | None = None, trust_remote_code: bool = False
) -> SentenceTransformer:
    """Get a SentenceTransformer from cache or load it.

    Args:
        model_name: HuggingFace model name or path.
        cache: Optional ModelCache instance. Uses global default if None.
        trust_remote_code: Whether to trust remote code (for custom models).

    Returns:
        Cached or newly loaded SentenceTransformer instance.
    """
    if cache is None:
        cache = get_default_cache()

    def factory() -> SentenceTransformer:
        return SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

    return cache.get_or_load(f"{model_name}:{trust_remote_code}", factory)


def get_cached_cross_encoder(
    model_name: str,
    device: str | None = None,
    cache: ModelCache | None = None,
) -> Any:
    """Get a CrossEncoder from cache or load it.

    Args:
        model_name: HuggingFace model name or path.
        device: Device to run model on (None for auto-detection).
        cache: Optional ModelCache instance. Uses global default if None.

    Returns:
        Cached or newly loaded CrossEncoder instance.
    """
    from sentence_transformers import CrossEncoder

    if cache is None:
        cache = get_default_cache()

    cache_key = f"crossencoder:{model_name}:{device}"

    def factory() -> CrossEncoder:
        return CrossEncoder(model_name, device=device)

    return cache.get_or_load(cache_key, factory)


def get_cached_setfit_model(
    model_name: str,
    labels: list[str] | None = None,
    cache: ModelCache | None = None,
) -> Any:
    """Get a SetFitModel from cache or load it.

    Args:
        model_name: HuggingFace model name or path.
        labels: Optional list of labels for the classifier.
        cache: Optional ModelCache instance. Uses global default if None.

    Returns:
        Cached or newly loaded SetFitModel instance.
    """
    from setfit import SetFitModel

    if cache is None:
        cache = get_default_cache()

    labels_key = ",".join(labels) if labels else "none"
    cache_key = f"setfit:{model_name}:{labels_key}"

    def factory() -> SetFitModel:
        return SetFitModel.from_pretrained(model_name, labels=labels)

    return cache.get_or_load(cache_key, factory)


def compute_embeddings(
    texts: list[str], model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2"
) -> np.ndarray:
    """Compute embeddings for a list of texts."""
    model = get_cached_sentence_transformer(model_name)
    result = model.encode(texts)
    return np.asarray(result)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    a = a.flatten()
    b = b.flatten()
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


def batch_encode(
    texts: list[str],
    model_name: str = "sentence-transformers/paraphrase-mpnet-base-v2",
    batch_size: int = 32,
) -> Iterator[np.ndarray]:
    """Encode texts in batches."""
    model = get_cached_sentence_transformer(model_name)
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        yield np.asarray(model.encode(batch))

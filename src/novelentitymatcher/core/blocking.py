"""Blocking strategies for efficient candidate filtering."""

from abc import ABC, abstractmethod
from hashlib import md5
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from rapidfuzz import fuzz, process
from sklearn.feature_extraction.text import TfidfVectorizer


class BlockingStrategy(ABC):
    """Abstract base class for blocking strategies."""

    @abstractmethod
    def block(
        self, query: str, entities: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """
        Return top_k candidate entities for the query.

        Args:
            query: Query text
            entities: List of all entities
            top_k: Maximum number of candidates to return

        Returns:
            List of candidate entities (top_k or fewer)
        """


class NoOpBlocking(BlockingStrategy):
    """
    Pass-through blocking for small datasets.

    Returns all entities up to top_k without any filtering.
    """

    def block(
        self, query: str, entities: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Return all entities or top_k if smaller."""
        if len(entities) <= top_k:
            return entities
        return entities[:top_k]


class BM25Blocking(BlockingStrategy):
    """
    Fast lexical blocking using BM25.

    Uses BM25 algorithm for efficient lexical matching.
    Good for keyword-heavy queries and proper nouns.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 blocking.

        Args:
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.k1 = k1
        self.b = b
        self.bm25: BM25Okapi | None = None
        self.cached_entities: list[dict[str, Any]] | None = None
        self._entity_hash: str | None = None

    def build_index(self, entities: list[dict[str, Any]]):
        """Build BM25 index from entities."""
        self.cached_entities = entities
        self._entity_hash = _compute_entity_hash(entities)

        tokenized_corpus = [
            self._tokenize(e.get("text", e.get("name", ""))) for e in entities
        ]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)

    def block(
        self, query: str, entities: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Return top_k candidates using BM25 scores."""
        current_hash = _compute_entity_hash(entities)

        if self.bm25 is None or self._entity_hash != current_hash:
            self.build_index(entities)

        tokenized_query = self._tokenize(query)
        assert self.bm25 is not None
        scores = self.bm25.get_scores(tokenized_query)

        # Get top_k indices
        top_k = min(top_k, len(scores))
        top_indices = np.argsort(scores)[-top_k:][::-1]

        return [entities[i] for i in top_indices]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple tokenization - lowercase and split."""
        return text.lower().split()


def _compute_entity_hash(entities: list[dict[str, Any]]) -> str:
    """Compute efficient content hash for entity list.

    Uses MD5 of sorted serialized entities for O(n) hashing
    instead of O(n) tuple construction + O(n) hash.
    """
    hasher = md5()
    for entity in sorted(entities, key=lambda e: e["id"]):
        hasher.update(str(entity["id"]).encode())
        hasher.update(entity.get("text", entity.get("name", "")).encode())
    return hasher.hexdigest()


class TFIDFBlocking(BlockingStrategy):
    """
    TF-IDF based blocking.

    Uses TF-IDF vectorization for lexical matching.
    Good for document-level similarity.

    Optimized with:
    - Vocabulary caching across rebuilds
    - Efficient content-based hashing (MD5)
    - Sparse matrix operations via sklearn
    """

    def __init__(self):
        """Initialize TF-IDF blocking."""
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: Any | None = None
        self.cached_entities: list[dict[str, Any]] | None = None
        self._entity_hash: str | None = None
        self._vocabulary: dict[str, int] | None = None

    def build_index(self, entities: list[dict[str, Any]]):
        """Build TF-IDF index from entities."""
        self.cached_entities = entities
        self._entity_hash = _compute_entity_hash(entities)

        texts = [e.get("text", e.get("name", "")) for e in entities]

        if self._vocabulary is None:
            self.vectorizer = TfidfVectorizer()
            self.matrix = self.vectorizer.fit_transform(texts)
            self._vocabulary = self.vectorizer.vocabulary_
        else:
            self.vectorizer = TfidfVectorizer(vocabulary=self._vocabulary)
            self.matrix = self.vectorizer.fit_transform(texts)

    def block(
        self, query: str, entities: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Return top_k candidates using TF-IDF scores."""
        current_hash = _compute_entity_hash(entities)

        if self.vectorizer is None or self._entity_hash != current_hash:
            self.build_index(entities)

        assert self.vectorizer is not None
        query_vec = self.vectorizer.transform([query])
        assert self.matrix is not None
        scores = (self.matrix @ query_vec.T).toarray().flatten()

        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [entities[i] for i in top_indices]


class FuzzyBlocking(BlockingStrategy):
    """
    Fuzzy string matching blocking.

    Uses RapidFuzz for approximate string matching.
    Good for catching typos and variations.
    """

    def __init__(self, score_cutoff: int = 70):
        """
        Initialize fuzzy blocking.

        Args:
            score_cutoff: Minimum similarity score (0-100)
        """
        self.score_cutoff = score_cutoff

    def block(
        self, query: str, entities: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Return top_k candidates using fuzzy matching."""
        texts = [e.get("text", e.get("name", "")) for e in entities]

        # Extract top matches with indices
        # process.extract returns list of (match, score, index) tuples
        results = process.extract(
            query, texts, scorer=fuzz.token_sort_ratio, limit=top_k
        )

        # Filter by score cutoff, preserving indices
        filtered = [
            (text, score, idx)
            for text, score, idx in results
            if score >= self.score_cutoff
        ]

        # Return matching entities using correct indices
        return [entities[idx] for _, _, idx in filtered]

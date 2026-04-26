"""
Cluster evidence extractor with configurable extraction methods.

Supports three evidence methods for benchmarking:
- ``"tfidf"``: TF-IDF keyword extraction (baseline).
- ``"centroid"``: Terms closest to cluster embedding centroid.
- ``"combined"``: Union of tfidf + centroid with deduplication.
"""

from __future__ import annotations

import math
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ...utils.logging_config import get_logger
from ..schemas.models import ClusterEvidence

logger = get_logger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
}


class ClusterEvidenceExtractor:
    """Extract compact evidence from a cluster of text samples.

    Args:
        method: Extraction method - ``"tfidf"``, ``"centroid"``, or ``"combined"``.
        max_keywords: Maximum keywords to return.
        max_examples: Maximum representative examples.
        token_budget: Soft token budget for representative examples.
    """

    def __init__(
        self,
        method: str = "tfidf",
        max_keywords: int = 8,
        max_examples: int = 4,
        token_budget: int = 256,
    ):
        if method not in ("tfidf", "centroid", "combined"):
            raise ValueError(
                f"Invalid evidence method: '{method}'. "
                "Must be 'tfidf', 'centroid', or 'combined'."
            )
        self.method = method
        self.max_keywords = max_keywords
        self.max_examples = max_examples
        self.token_budget = token_budget

    def extract(
        self,
        cluster_texts: list[str],
        cluster_embeddings: np.ndarray | None = None,
        reference_embeddings: np.ndarray | None = None,
    ) -> ClusterEvidence:
        """Extract evidence from a single cluster.

        Args:
            cluster_texts: Text samples in the cluster.
            cluster_embeddings: Embeddings for the cluster samples.
            reference_embeddings: Optional reference embeddings (not yet used).

        Returns:
            ClusterEvidence with keywords, examples, and metadata.
        """
        keywords = self._extract_keywords(cluster_texts, cluster_embeddings)
        representatives = self._select_representatives(
            cluster_texts, cluster_embeddings
        )

        return ClusterEvidence(
            keywords=keywords,
            representative_examples=representatives,
            sample_indices=list(range(len(cluster_texts))),
            metadata={
                "evidence_method": self.method,
                "sample_count": len(cluster_texts),
            },
            token_budget=self.token_budget,
        )

    def _extract_keywords(
        self,
        texts: list[str],
        embeddings: np.ndarray | None = None,
    ) -> list[str]:
        """Dispatch keyword extraction based on method."""
        if self.method == "centroid":
            return self._centroid_keywords(texts, embeddings)
        elif self.method == "combined":
            tfidf_kw = self._tfidf_keywords(texts)
            centroid_kw = self._centroid_keywords(texts, embeddings)
            seen: set[str] = set()
            combined: list[str] = []
            for kw in tfidf_kw + centroid_kw:
                normalized = kw.lower()
                if normalized not in seen:
                    seen.add(normalized)
                    combined.append(kw)
                    if len(combined) >= self.max_keywords:
                        break
            return combined
        else:
            return self._tfidf_keywords(texts)

    def _tfidf_keywords(self, texts: list[str]) -> list[str]:
        """TF-IDF based keyword extraction."""
        if not texts:
            return []
        joined = [" ".join(self._tokenize(t)) for t in texts]
        if not any(joined):
            return []
        try:
            vectorizer = TfidfVectorizer(
                stop_words=list(_STOPWORDS),
                token_pattern=r"[a-zA-Z0-9]+",
                max_features=1000,
            )
            matrix = vectorizer.fit_transform(joined)
        except ValueError:
            return []
        features = vectorizer.get_feature_names_out()
        avg_scores = np.asarray(matrix.mean(axis=0)).flatten()
        ranked = avg_scores.argsort()[::-1]
        return [features[i] for i in ranked[: self.max_keywords]]

    def _centroid_keywords(
        self, texts: list[str], embeddings: np.ndarray | None = None
    ) -> list[str]:
        """Keyword extraction based on proximity to the embedding centroid.

        When embeddings are provided, selects texts closest to the cluster
        centroid and extracts distinctive tokens from those representative
        samples. Falls back to TF-IDF centroid ranking when no embeddings.
        """
        if not texts:
            return []

        if embeddings is not None and len(embeddings) == len(texts):
            centroid = embeddings.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-12:
                emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                emb_norms = np.clip(emb_norms, a_min=1e-12, a_max=None)
                sims = (embeddings @ centroid) / (emb_norms.squeeze() * centroid_norm)
                top_indices = np.argsort(-sims)[: max(1, self.max_keywords)]
                representative_texts = [texts[int(i)] for i in top_indices]
                return self._tfidf_keywords(representative_texts)

        return self._tfidf_keywords(texts)

    def _select_representatives(
        self,
        texts: list[str],
        embeddings: np.ndarray | None = None,
    ) -> list[str]:
        """Select representative examples, preferring those closest to centroid."""
        if not texts:
            return []

        if embeddings is not None and len(embeddings) == len(texts):
            centroid = embeddings.mean(axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                return self._truncate(texts)

            emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            emb_norms = np.clip(emb_norms, a_min=1e-12, a_max=None)
            sims = (embeddings @ centroid) / (emb_norms.squeeze() * centroid_norm)
            top_indices = np.argsort(-sims)[: self.max_examples]

            representatives: list[str] = []
            tokens_used = 0
            for idx in top_indices:
                text = texts[int(idx)]
                token_est = max(1, math.ceil(len(text.split()) * 1.2))
                if tokens_used + token_est > self.token_budget and representatives:
                    break
                representatives.append(text)
                tokens_used += token_est
            return representatives

        return self._truncate(texts)

    def _truncate(self, texts: list[str]) -> list[str]:
        examples: list[str] = []
        tokens_used = 0
        for text in texts[: self.max_examples]:
            token_est = max(1, math.ceil(len(text.split()) * 1.2))
            if tokens_used + token_est > self.token_budget and examples:
                break
            examples.append(text)
            tokens_used += token_est
        return examples

    def _tokenize(self, text: str) -> list[str]:
        return [
            token
            for token in re.findall(r"[a-zA-Z0-9]+", text.lower())
            if len(token) > 2 and token not in _STOPWORDS
        ]

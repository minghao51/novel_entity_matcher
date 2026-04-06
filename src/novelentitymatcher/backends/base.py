from abc import ABC, abstractmethod
import heapq
from typing import Any, Dict, List


class EmbeddingBackend(ABC):
    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]:
        pass


class RerankerBackend(ABC):
    """Abstract base class for reranker backends."""

    @abstractmethod
    def score(self, query: str, docs: list[str]) -> List[float]:
        """Score query-document pairs."""
        pass

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5,
        text_field: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank candidates and return top_k.

        Default implementation using score(). Subclasses can override for optimization.
        """
        texts = [cand.get(text_field, cand.get("name", "")) for cand in candidates]
        scores = self.score(query, texts)

        scored = [
            {**candidate, "cross_encoder_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]

        return heapq.nlargest(top_k, scored, key=lambda x: x["cross_encoder_score"])

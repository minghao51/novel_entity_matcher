from __future__ import annotations

from typing import Any

import numpy as np


class UncertaintySampler:
    strategies: tuple[str, ...] = ("entropy", "margin", "least_confident")

    def __init__(self, strategy: str = "least_confident"):
        if strategy not in self.strategies:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {self.strategies}"
            )
        self._strategy = strategy

    @property
    def strategy(self) -> str:
        return self._strategy

    def sample(
        self,
        texts: list[str],
        confidences: list[float],
        predicted_classes: list[str],
        probabilities: np.ndarray | None = None,
        n_samples: int = 10,
    ) -> list[dict[str, Any]]:
        if n_samples <= 0:
            return []

        n = len(texts)
        if n == 0:
            return []

        scores = self._score(confidences, probabilities)
        top_indices = np.argsort(scores)[::-1][: min(n_samples, n)]

        return [
            {
                "text": texts[i],
                "confidence": confidences[i],
                "predicted_class": predicted_classes[i],
                "uncertainty_score": float(scores[i]),
                "strategy": self._strategy,
            }
            for i in top_indices
        ]

    def _score(
        self, confidences: list[float], probabilities: np.ndarray | None
    ) -> np.ndarray:
        if self._strategy == "least_confident":
            return np.asarray([1.0 - c for c in confidences])
        if self._strategy == "entropy" and probabilities is not None:
            return self._entropy(probabilities)
        if self._strategy == "margin" and probabilities is not None:
            return self._margin(probabilities)
        return np.asarray([1.0 - c for c in confidences])

    @staticmethod
    def _entropy(probabilities: np.ndarray | None) -> np.ndarray:
        if probabilities is None:
            return np.zeros(0)
        probs = np.asarray(probabilities, dtype=np.float64)
        probs = np.clip(probs, 1e-15, 1.0)
        return -np.sum(probs * np.log(probs), axis=1)

    @staticmethod
    def _margin(probabilities: np.ndarray | None) -> np.ndarray:
        if probabilities is None:
            return np.zeros(0)
        probs = np.asarray(probabilities, dtype=np.float64)
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margin = (
            sorted_probs[:, 0] - sorted_probs[:, 1]
            if sorted_probs.shape[1] >= 2
            else sorted_probs[:, 0]
        )
        return 1.0 - margin

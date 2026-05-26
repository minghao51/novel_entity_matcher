"""
Base protocol for novelty detection strategies.

All strategies must implement this protocol to be compatible
with the NoveltyDetector.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np


@dataclass(frozen=True)
class SignalInfo:
    score_key: str | None = None
    flag_key: str = ""
    weight_name: str = ""
    kind: str = "flag"


class NoveltyStrategy(ABC):
    """
    Base protocol for all novelty detection strategies.

    Each strategy is responsible for:
    1. Initializing with reference embeddings and labels
    2. Detecting novel samples from a batch of inputs
    3. Providing per-sample metrics for signal combination
    4. Specifying its weight for signal fusion
    """

    strategy_id: str
    maturity: Literal["production", "experimental", "internal"] = "experimental"
    score_keys: ClassVar[tuple[str, ...]] = ()
    signal_info: ClassVar[SignalInfo | None] = None
    default_weight: ClassVar[float] = 0.0

    @abstractmethod
    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
        config: Any,
    ) -> None:
        """
        Initialize strategy with reference data.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Class labels for known samples
            config: Strategy-specific configuration object
        """

    @abstractmethod
    def _detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs: Any,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        """Subclass implementation of novelty detection."""

    def detect(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        predicted_classes: list[str],
        confidences: np.ndarray,
        **kwargs: Any,
    ) -> tuple[set[int], dict[int, dict[str, Any]]]:
        if len(embeddings) == 0:
            return set(), {}
        if np.any(~np.isfinite(embeddings)):
            raise ValueError("Embeddings contain NaN or Inf values")
        return self._detect(texts, embeddings, predicted_classes, confidences, **kwargs)

    @property
    @abstractmethod
    def config_schema(self) -> type:
        """
        Return the config dataclass type for this strategy.

        This is used for validation and defaults.
        """

    def get_weight(self) -> float:
        """
        Return weight for signal combination.

        This weight determines how much this strategy contributes
        to the final novelty score.
        """
        return self.default_weight

    def save(self, path: str) -> None:
        """Persist trained strategy state to *path* (a directory)."""
        raise NotImplementedError(f"{type(self).__name__} does not implement save()")

    @classmethod
    def load(cls, path: str) -> "NoveltyStrategy":
        """Reconstruct a strategy from the directory at *path*."""
        raise NotImplementedError(f"{cls.__name__} does not implement load()")

    def get_config(self) -> Any:
        """
        Get the current configuration for this strategy.

        Override this if your strategy stores its config differently.
        """
        return getattr(self, "_config", None)

    @staticmethod
    def compute_class_means(
        embeddings: np.ndarray,
        labels: list[str],
    ) -> dict[str, np.ndarray]:
        class_means: dict[str, np.ndarray] = {}
        for label in set(labels):
            mask = np.array(labels) == label
            class_means[label] = embeddings[mask].mean(axis=0)
        return class_means

    @staticmethod
    def _resolve_class_mean(
        predicted_class: str,
        class_means: dict[str, np.ndarray],
        global_mean: np.ndarray,
    ) -> np.ndarray:
        if predicted_class in class_means:
            return class_means[predicted_class]
        return global_mean

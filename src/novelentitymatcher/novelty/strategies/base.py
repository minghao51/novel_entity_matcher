"""
Base protocol for novelty detection strategies.

All strategies must implement this protocol to be compatible
with the NoveltyDetector.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set, Any, Literal, Type
import numpy as np


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

    @abstractmethod
    def initialize(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: List[str],
        config: Any,
    ) -> None:
        """
        Initialize strategy with reference data.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Class labels for known samples
            config: Strategy-specific configuration object
        """
        pass

    @abstractmethod
    def detect(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        predicted_classes: List[str],
        confidences: np.ndarray,
        **kwargs,
    ) -> tuple[Set[int], Dict[int, Dict[str, Any]]]:
        """
        Detect novel samples.

        Args:
            texts: Input texts
            embeddings: Text embeddings
            predicted_classes: Predicted class for each sample
            confidences: Prediction confidence scores
            **kwargs: Additional strategy-specific parameters

        Returns:
            (flags, metrics) - flagged indices and per-sample metrics
            - flags: Set of indices flagged as novel
            - metrics: Dict mapping index to metric dict
        """
        pass

    @property
    @abstractmethod
    def config_schema(self) -> Type:
        """
        Return the config dataclass type for this strategy.

        This is used for validation and defaults.
        """
        pass

    @abstractmethod
    def get_weight(self) -> float:
        """
        Return weight for signal combination.

        This weight determines how much this strategy contributes
        to the final novelty score.
        """
        pass

    def get_config(self) -> Any:
        """
        Get the current configuration for this strategy.

        Override this if your strategy stores its config differently.
        """
        return getattr(self, "_config", None)

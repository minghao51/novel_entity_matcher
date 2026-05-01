"""
Core novelty detector with strategy orchestration.

This is the main entry point for novelty detection, using a strategy
pattern to support multiple detection algorithms.
"""

import hashlib
from typing import Any

import numpy as np

from ..config.base import DetectionConfig
from ..schemas import NovelSampleReport
from .metadata import MetadataBuilder
from .signal_combiner import SignalCombiner
from .strategies import StrategyRegistry


class NoveltyDetector:
    """
    Simplified novelty detector using registered strategies.

    This detector manages strategy initialization and orchestration,
    delegating signal combination and metadata building to specialized
    components.

    Responsibilities:
    - Strategy initialization and lifecycle
    - Strategy orchestration
    - Delegates signal combining to SignalCombiner
    - Delegates metadata creation to MetadataBuilder
    """

    def __init__(self, config: DetectionConfig):
        """
        Initialize the novelty detector.

        Args:
            config: Detection configuration
        """
        # Validate configuration
        config.validate_strategies()

        self.config = config
        self._strategies: dict[str, Any] = {}
        self._combiner = SignalCombiner(config)
        self._metadata_builder = MetadataBuilder()
        self._is_initialized = False
        self._reference_signature: str | None = None

    @staticmethod
    def _compute_reference_signature(
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> str:
        """Create a stable signature for the active reference corpus."""
        normalized = np.ascontiguousarray(reference_embeddings)
        digest = hashlib.sha256()
        digest.update(str(normalized.shape).encode("utf-8"))
        digest.update(str(normalized.dtype).encode("utf-8"))
        digest.update(normalized.tobytes())
        for label in reference_labels:
            digest.update(b"\0")
            digest.update(str(label).encode("utf-8"))
        return digest.hexdigest()

    def _initialize_strategies(
        self,
        reference_embeddings: np.ndarray,
        reference_labels: list[str],
    ) -> None:
        """
        Initialize all configured strategies.

        Args:
            reference_embeddings: Embeddings of known samples
            reference_labels: Class labels for known samples
        """
        self._strategies.clear()

        for strategy_id in self.config.strategies:
            # Get strategy class from registry
            strategy_cls = StrategyRegistry.get(strategy_id)

            # Create strategy instance
            strategy = strategy_cls()

            # Get strategy-specific config
            strategy_config = self.config.get_strategy_config(strategy_id)

            # Initialize the strategy
            strategy.initialize(
                reference_embeddings=reference_embeddings,
                reference_labels=reference_labels,
                config=strategy_config,
            )

            # Store initialized strategy
            self._strategies[strategy_id] = strategy

        self._is_initialized = True
        self._reference_signature = self._compute_reference_signature(
            reference_embeddings,
            reference_labels,
        )

    def detect_novel_samples(
        self,
        texts: list[str],
        confidences: np.ndarray,
        embeddings: np.ndarray,
        predicted_classes: list[str],
        reference_embeddings: np.ndarray | None = None,
        reference_labels: list[str] | None = None,
        **kwargs,
    ) -> NovelSampleReport:
        """
        Detect novel samples using configured strategies.

        Args:
            texts: Input texts
            confidences: Prediction confidence scores
            embeddings: Text embeddings
            predicted_classes: Predicted class for each sample
            reference_embeddings: Embeddings of known samples
            reference_labels: Class labels for known samples
            **kwargs: Additional strategy-specific parameters

        Returns:
            NovelSampleReport with detection results
        """
        if reference_embeddings is None or reference_labels is None:
            raise RuntimeError("reference embeddings and labels are required")

        if len(texts) == 0:
            return NovelSampleReport(
                novel_samples=[],
                detection_strategies=list(self.config.strategies),
                config=self.config.model_dump(),
                signal_counts=dict.fromkeys(self.config.strategies, 0),
            )

        reference_signature = self._compute_reference_signature(
            reference_embeddings,
            reference_labels,
        )

        # Initialize strategies if needed or if the reference corpus changed.
        if not self._is_initialized or self._reference_signature != reference_signature:
            self._initialize_strategies(reference_embeddings, reference_labels)

        # Run each strategy
        all_flags: set[int] = set()
        all_metrics: dict[int, dict[str, Any]] = {}
        strategy_outputs: dict[str, tuple[set[int], dict]] = {}

        for strategy_id, strategy in self._strategies.items():
            flags, metrics = strategy.detect(
                texts=texts,
                embeddings=embeddings,
                predicted_classes=predicted_classes,
                confidences=confidences,
                **kwargs,
            )
            strategy_outputs[strategy_id] = (flags, metrics)
            all_flags.update(flags)

            # Merge metrics
            for idx, metric_dict in metrics.items():
                if idx not in all_metrics:
                    all_metrics[idx] = {}
                all_metrics[idx].update(metric_dict)

        # Combine signals
        novel_indices, novelty_scores = self._combiner.combine(
            strategy_outputs=strategy_outputs,
            all_metrics=all_metrics,
        )

        # Build report
        report = self._metadata_builder.build_report(
            texts=texts,
            confidences=confidences,
            predicted_classes=predicted_classes,
            novel_indices=novel_indices,
            novelty_scores=novelty_scores,
            all_metrics=all_metrics,
            strategy_outputs=strategy_outputs,
            config=self.config,
        )

        return report

    def reset(self) -> None:
        """
        Reset the detector, clearing all initialized strategies.

        This allows the detector to be re-used with different reference data.
        """
        self._strategies.clear()
        self._is_initialized = False
        self._reference_signature = None

    def get_strategy(self, strategy_id: str) -> Any:
        """
        Get an initialized strategy by ID.

        Args:
            strategy_id: Strategy identifier

        Returns:
            Strategy instance if initialized

        Raises:
            ValueError: If strategy not found or not initialized
        """
        if strategy_id not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise ValueError(
                f"Strategy '{strategy_id}' not initialized. Available: {available}"
            )
        return self._strategies[strategy_id]

    def list_initialized_strategies(self) -> list[str]:
        """
        List all initialized strategies.

        Returns:
            List of strategy IDs
        """
        return list(self._strategies.keys())

    @property
    def is_initialized(self) -> bool:
        """Check if detector has been initialized with reference data."""
        return self._is_initialized

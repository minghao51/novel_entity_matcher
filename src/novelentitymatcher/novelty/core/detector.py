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
from .score_calibrator import OODScoreCalibrator
from .signal_combiner import SignalCombiner
from .strategies import StrategyRegistry

_STRATEGY_SCORE_KEYS: dict[str, list[str]] = {
    "confidence": ["confidence_score"],
    "uncertainty": ["uncertainty_score"],
    "knn_distance": ["knn_novelty_score"],
    "clustering": ["cluster_support_score"],
    "self_knowledge": ["self_knowledge_score"],
    "pattern": ["pattern_score"],
    "oneclass": ["oneclass_score"],
    "prototypical": ["prototypical_score"],
    "setfit": ["setfit_score"],
    "setfit_centroid": ["setfit_centroid_score"],
    "mahalanobis": ["mahalanobis_novelty_score"],
    "lof": ["lof_novelty_score"],
}


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

    def __init__(
        self,
        config: DetectionConfig,
        calibrator: OODScoreCalibrator | None = None,
    ):
        """
        Initialize the novelty detector.

        Args:
            config: Detection configuration
            calibrator: Optional OOD score calibrator for normalizing raw scores to [0,1]
        """
        # Validate configuration
        config.validate_strategies()

        self.config = config
        self._strategies: dict[str, Any] = {}
        self._combiner = SignalCombiner(config)
        self._metadata_builder = MetadataBuilder()
        self._calibrator = calibrator
        self._is_initialized = False
        self._reference_signature: str | None = None
        self._calibrator_reference_signature: str | None = None

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
        # Reset calibrator when reference corpus changes so it will be re-fitted
        # on the next detection batch.
        if self._calibrator is not None:
            self._calibrator.reset()
            self._calibrator_reference_signature = None

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

        # Calibrate raw scores before signal combination.
        # The calibrator is fitted once per reference corpus (on the first
        # detection batch) and reused for subsequent batches so that scores
        # are comparable across calls.
        if self._calibrator is not None:
            # Build score keys: hardcoded mapping + dynamically discovered
            # numeric keys from each strategy's raw metrics.
            score_keys_by_strategy: dict[str, list[str]] = {
                k: list(v) for k, v in _STRATEGY_SCORE_KEYS.items()
            }
            for strategy_id, (_flags, metrics) in strategy_outputs.items():
                discovered: set[str] = set()
                for metric_dict in metrics.values():
                    for key, val in metric_dict.items():
                        if isinstance(val, (int, float)) and not isinstance(val, bool):
                            discovered.add(key)
                existing = set(score_keys_by_strategy.get(strategy_id, []))
                score_keys_by_strategy[strategy_id] = sorted(existing | discovered)

            strategy_scores: dict[str, list[float]] = {}
            for strategy_id, score_keys in score_keys_by_strategy.items():
                if strategy_id not in strategy_outputs:
                    continue
                scores = []
                for idx_metrics in all_metrics.values():
                    for key in score_keys:
                        val = idx_metrics.get(key)
                        if val is not None:
                            scores.append(float(val))
                if scores:
                    strategy_scores[strategy_id] = scores

            if strategy_scores:
                score_arrays = {k: np.array(v) for k, v in strategy_scores.items()}
                # Fit once per reference corpus; reuse stats afterward.
                if self._calibrator_reference_signature != reference_signature:
                    self._calibrator.fit(score_arrays)
                    self._calibrator_reference_signature = reference_signature
                for strategy_id, score_keys in score_keys_by_strategy.items():
                    if strategy_id not in strategy_outputs:
                        continue
                    for idx_metrics in all_metrics.values():
                        for key in score_keys:
                            val = idx_metrics.get(key)
                            if val is not None:
                                idx_metrics[key] = float(
                                    self._calibrator.transform(
                                        strategy_id, np.array([val])
                                    )[0]
                                )

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
        """Reset the detector, clearing all initialized strategies and calibrator state."""
        self._strategies.clear()
        self._is_initialized = False
        self._reference_signature = None
        self._calibrator_reference_signature = None
        if self._calibrator is not None:
            self._calibrator.reset()

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

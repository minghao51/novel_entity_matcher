"""Tests for NoveltyDetector strategy lifecycle (get_strategy, list_initialized, reset)."""

import numpy as np
import pytest

from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import ConfidenceConfig, KNNConfig
from novelentitymatcher.novelty.core.detector import NoveltyDetector
from novelentitymatcher.novelty.strategies.base import NoveltyStrategy
from novelentitymatcher.novelty.core.strategies import StrategyRegistry


class TestNoveltyDetectorStrategyManagement:
    """Tests for strategy lifecycle methods on NoveltyDetector."""

    @pytest.fixture
    def detector(self):
        return NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                confidence=ConfidenceConfig(threshold=0.6),
                knn_distance=KNNConfig(distance_threshold=0.25),
            )
        )

    @pytest.fixture
    def initialized_detector(self, detector):
        """Detector that has been initialized with reference data."""
        detector.detect_novel_samples(
            texts=["a", "b"],
            confidences=np.array([0.9, 0.8]),
            embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            predicted_classes=["x", "y"],
            reference_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            reference_labels=["x", "y"],
        )
        return detector

    def test_is_initialized_false_before_detection(self, detector):
        """is_initialized should be False before any detection call."""
        assert detector.is_initialized is False

    def test_is_initialized_true_after_detection(self, initialized_detector):
        """is_initialized should be True after successful detection."""
        assert initialized_detector.is_initialized is True

    def test_get_strategy_returns_initialized_strategy(self, initialized_detector):
        """get_strategy should return the strategy instance after initialization."""
        strategy = initialized_detector.get_strategy("confidence")
        assert strategy is not None

    def test_get_strategy_unknown_id_raises_error(self, initialized_detector):
        """get_strategy should raise ValueError for unknown strategy_id."""
        with pytest.raises(ValueError, match="not initialized"):
            initialized_detector.get_strategy("unknown_strategy")

    def test_get_strategy_before_initialization_raises_error(self, detector):
        """get_strategy should raise ValueError before initialization."""
        with pytest.raises(ValueError, match="not initialized"):
            detector.get_strategy("confidence")

    def test_list_initialized_strategies_empty_before_init(self, detector):
        """list_initialized_strategies should return empty list before initialization."""
        assert detector.list_initialized_strategies() == []

    def test_list_initialized_strategies_returns_configured_after_init(
        self, initialized_detector
    ):
        """list_initialized_strategies should list all configured strategies after init."""
        strategies = initialized_detector.list_initialized_strategies()
        assert "confidence" in strategies
        assert "knn_distance" in strategies

    def test_list_initialized_strategies_only_includes_initialized(self):
        """list_initialized_strategies should only return strategies that were in config."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence"],
                confidence=ConfidenceConfig(threshold=0.6),
            )
        )
        detector.detect_novel_samples(
            texts=["a"],
            confidences=np.array([0.9]),
            embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            predicted_classes=["x"],
            reference_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            reference_labels=["x"],
        )

        assert detector.list_initialized_strategies() == ["confidence"]

    def test_reset_clears_strategies(self, initialized_detector):
        """reset should clear all initialized strategies."""
        initialized_detector.reset()

        assert initialized_detector.is_initialized is False
        assert initialized_detector.list_initialized_strategies() == []

    def test_reset_then_reinit_with_different_reference(self, initialized_detector):
        """reset should allow re-initialization with different reference data."""
        initialized_detector.reset()

        initialized_detector.detect_novel_samples(
            texts=["c", "d"],
            confidences=np.array([0.7, 0.6]),
            embeddings=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            predicted_classes=["z", "w"],
            reference_embeddings=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
            reference_labels=["z", "w"],
        )

        assert initialized_detector.is_initialized is True
        assert "confidence" in initialized_detector.list_initialized_strategies()

    def test_multiple_detections_do_not_reinitialize_if_reference_same(self):
        """Multiple detections with same reference should not reinitialize strategies."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence"],
                confidence=ConfidenceConfig(threshold=0.6),
            )
        )
        ref_emb = np.array([[1.0, 0.0]], dtype=np.float32)
        ref_labels = ["x"]

        detector.detect_novel_samples(
            texts=["a"],
            confidences=np.array([0.9]),
            embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
            predicted_classes=["x"],
            reference_embeddings=ref_emb,
            reference_labels=ref_labels,
        )
        first_strategy = detector.get_strategy("confidence")

        detector.detect_novel_samples(
            texts=["b"],
            confidences=np.array([0.8]),
            embeddings=np.array([[0.9, 0.1]], dtype=np.float32),
            predicted_classes=["x"],
            reference_embeddings=ref_emb,
            reference_labels=ref_labels,
        )
        second_strategy = detector.get_strategy("confidence")

        assert first_strategy is second_strategy

    def test_detector_reinitializes_when_reference_corpus_changes(self):
        """Detector should reinitialize strategies when reference corpus changes significantly."""
        unique_probe_id = "test_probe_reinit_ref"

        class ProbeStrategy(NoveltyStrategy):
            strategy_id = unique_probe_id
            _initialized_with: tuple = None

            def initialize(self, reference_embeddings, reference_labels, config):
                ProbeStrategy._initialized_with = (
                    float(np.sum(reference_embeddings)),
                    tuple(reference_labels),
                )

            def detect(
                self, texts, embeddings, predicted_classes, confidences, **kwargs
            ):
                return set(), {}

            @property
            def config_schema(self):
                return object

            def get_weight(self) -> float:
                return 1.0

        if StrategyRegistry.is_registered(unique_probe_id):
            StrategyRegistry.unregister(unique_probe_id)
        StrategyRegistry.register(ProbeStrategy)

        try:
            detector = NoveltyDetector(
                config=DetectionConfig(
                    strategies=[unique_probe_id],
                )
            )

            common_args = {
                "texts": ["sample"],
                "confidences": np.array([0.1], dtype=float),
                "embeddings": np.array([[0.0]], dtype=np.float32),
                "predicted_classes": ["physics"],
            }

            detector.detect_novel_samples(
                reference_embeddings=np.array([[1.0]], dtype=np.float32),
                reference_labels=["physics"],
                **common_args,
            )
            first_init = ProbeStrategy._initialized_with

            detector.detect_novel_samples(
                reference_embeddings=np.array([[5.0]], dtype=np.float32),
                reference_labels=["biology"],
                **common_args,
            )
            second_init = ProbeStrategy._initialized_with
        finally:
            StrategyRegistry.unregister(unique_probe_id)

        assert first_init == (1.0, ("physics",))
        assert second_init == (5.0, ("biology",))


class TestNoveltyDetectorSignatureComputation:
    """Tests for reference signature computation."""

    def test_signature_changes_with_different_embeddings_shape(self):
        """Signature should change when embedding shape changes."""
        detector = NoveltyDetector(config=DetectionConfig(strategies=[]))

        sig1 = detector._compute_reference_signature(
            np.array([[1.0, 0.0]], dtype=np.float32),
            ["a"],
        )
        sig2 = detector._compute_reference_signature(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            ["a"],
        )

        assert sig1 != sig2

    def test_signature_changes_with_different_labels(self):
        """Signature should change when reference labels change."""
        detector = NoveltyDetector(config=DetectionConfig(strategies=[]))

        sig1 = detector._compute_reference_signature(
            np.array([[1.0, 0.0]], dtype=np.float32),
            ["a"],
        )
        sig2 = detector._compute_reference_signature(
            np.array([[1.0, 0.0]], dtype=np.float32),
            ["b"],
        )

        assert sig1 != sig2

    def test_signature_same_for_identical_reference(self):
        """Signature should be identical for same reference data."""
        detector = NoveltyDetector(config=DetectionConfig(strategies=[]))

        emb = np.array([[1.0, 0.0]], dtype=np.float32)
        labels = ["a"]

        sig1 = detector._compute_reference_signature(emb, labels)
        sig2 = detector._compute_reference_signature(emb, labels)

        assert sig1 == sig2

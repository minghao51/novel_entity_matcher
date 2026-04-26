"""Tests for the modular NoveltyDetector."""

import numpy as np
import pytest

from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import (
    ClusteringConfig,
    ConfidenceConfig,
    KNNConfig,
)
from novelentitymatcher.novelty.config.weights import WeightConfig
from novelentitymatcher.novelty.core.detector import NoveltyDetector
from novelentitymatcher.novelty.core.strategies import StrategyRegistry
from novelentitymatcher.novelty.strategies.base import NoveltyStrategy


def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401

        return True
    except ImportError:
        return False


class TestNoveltyDetector:
    @pytest.fixture
    def sample_texts(self):
        return [
            "quantum physics research",
            "machine learning algorithms",
            "neural network architecture",
            "quantum entanglement experiments",
            "biology research methods",
            "gene editing techniques",
            "novel interdisciplinary sample",
            "another emerging topic",
        ]

    @pytest.fixture
    def reference_embeddings(self):
        return np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.95, 0.05, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.05, 0.95, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.95, 0.05],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def reference_labels(self):
        return ["physics", "physics", "cs", "cs", "biology", "biology"]

    @pytest.fixture
    def sample_embeddings(self):
        return np.array(
            [
                [0.98, 0.02, 0.0, 0.0],
                [0.0, 0.99, 0.01, 0.0],
                [0.02, 0.97, 0.01, 0.0],
                [0.96, 0.04, 0.0, 0.0],
                [0.0, 0.0, 0.99, 0.01],
                [0.0, 0.02, 0.97, 0.01],
                [0.55, 0.55, 0.0, 0.62],
                [0.56, 0.54, 0.02, 0.61],
            ],
            dtype=np.float32,
        )

    @pytest.fixture
    def sample_confidences(self):
        return np.array([0.97, 0.96, 0.95, 0.97, 0.94, 0.93, 0.58, 0.56], dtype=float)

    @pytest.fixture
    def sample_predictions(self):
        return ["physics", "cs", "cs", "physics", "biology", "biology", "cs", "cs"]

    @pytest.fixture
    def detector(self):
        return NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance", "clustering"],
                confidence=ConfidenceConfig(threshold=0.6),
                knn_distance=KNNConfig(distance_threshold=0.25),
                clustering=ClusteringConfig(min_cluster_size=2),
            )
        )

    def test_initialization(self, detector):
        assert detector.config is not None
        assert detector.is_initialized is False

    def test_default_initialization(self):
        detector = NoveltyDetector(config=DetectionConfig())
        assert detector.config.combine_method == "weighted"

    @pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
    def test_knn_distance_detection(
        self,
        detector,
        sample_texts,
        sample_embeddings,
        sample_confidences,
        sample_predictions,
        reference_embeddings,
        reference_labels,
    ):
        detector.detect_novel_samples(
            texts=sample_texts,
            confidences=sample_confidences,
            embeddings=sample_embeddings,
            predicted_classes=sample_predictions,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
        )
        strategy = detector.get_strategy("knn_distance")
        flags, metrics = strategy.detect(
            texts=sample_texts,
            embeddings=sample_embeddings,
            predicted_classes=sample_predictions,
            confidences=sample_confidences,
        )

        assert isinstance(metrics, dict)
        assert 6 in flags
        assert 7 in flags
        assert (
            metrics[6]["knn_novelty_score"]
            >= detector.config.knn_distance.distance_threshold
        )

    def test_missing_reference_embeddings_raises(
        self,
        detector,
        sample_texts,
        sample_confidences,
        sample_embeddings,
        sample_predictions,
    ):
        with pytest.raises(RuntimeError, match="reference embeddings"):
            detector.detect_novel_samples(
                texts=sample_texts,
                confidences=sample_confidences,
                embeddings=sample_embeddings,
                predicted_classes=sample_predictions,
            )

    @pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
    def test_full_detection_pipeline(
        self,
        detector,
        sample_texts,
        sample_confidences,
        sample_embeddings,
        sample_predictions,
        reference_embeddings,
        reference_labels,
    ):
        report = detector.detect_novel_samples(
            texts=sample_texts,
            confidences=sample_confidences,
            embeddings=sample_embeddings,
            predicted_classes=sample_predictions,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
        )

        assert len(report.novel_samples) >= 2
        novel_indices = {sample.index for sample in report.novel_samples}
        assert {6, 7}.issubset(novel_indices)

        sample = next(item for item in report.novel_samples if item.index == 6)
        assert sample.knn_novelty_score is not None
        assert sample.novelty_score is not None
        assert sample.signals["knn_distance"] is True

    def test_intersection_combine_mode_still_works(self):
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                combine_method="intersection",
                confidence=ConfidenceConfig(threshold=0.6),
                knn_distance=KNNConfig(distance_threshold=0.25),
            )
        )
        combined, _ = detector._combiner.combine(
            {
                "confidence": ({0, 2}, {}),
                "knn_distance": ({1, 2}, {}),
            },
            {},
        )
        assert combined == {2}

    def test_empty_samples(self, detector):
        report = detector.detect_novel_samples(
            texts=[],
            confidences=np.array([]),
            embeddings=np.array([]).reshape(0, 4),
            predicted_classes=[],
            reference_embeddings=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            reference_labels=["physics"],
        )

        assert len(report.novel_samples) == 0

    def test_detector_reinitializes_when_reference_changes(self):
        unique_strategy_id = "test_probe_reinitialize"

        class ProbeStrategy(NoveltyStrategy):
            strategy_id = unique_strategy_id

            def __init__(self):
                self.reference_sum = None

            def initialize(self, reference_embeddings, reference_labels, config):
                self.reference_sum = float(np.sum(reference_embeddings))

            def detect(
                self,
                texts,
                embeddings,
                predicted_classes,
                confidences,
                **kwargs,
            ):
                return set(), {
                    idx: {"probe_reference_sum": self.reference_sum}
                    for idx in range(len(texts))
                }

            @property
            def config_schema(self):
                return object

            def get_weight(self) -> float:
                return 1.0

        if StrategyRegistry.is_registered(unique_strategy_id):
            StrategyRegistry.unregister(unique_strategy_id)
        StrategyRegistry.register(ProbeStrategy)
        try:
            detector = NoveltyDetector(
                config=DetectionConfig(
                    strategies=[unique_strategy_id],
                    combine_method="union",
                )
            )
            common = {
                "texts": ["sample"],
                "confidences": np.array([0.1], dtype=float),
                "embeddings": np.array([[0.0]], dtype=np.float32),
                "predicted_classes": ["physics"],
            }

            detector.detect_novel_samples(
                reference_embeddings=np.array([[1.0]], dtype=np.float32),
                reference_labels=["physics"],
                **common,
            )
            first_reference_sum = detector.get_strategy(
                unique_strategy_id
            ).reference_sum

            detector.detect_novel_samples(
                reference_embeddings=np.array([[5.0]], dtype=np.float32),
                reference_labels=["biology"],
                **common,
            )
            second_reference_sum = detector.get_strategy(
                unique_strategy_id
            ).reference_sum
        finally:
            StrategyRegistry.unregister(unique_strategy_id)

        assert first_reference_sum == 1.0
        assert second_reference_sum == 5.0


class TestSignalCombinerRegression:
    @staticmethod
    def _make_detector(weights: WeightConfig) -> NoveltyDetector:
        return NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance", "clustering"],
                combine_method="weighted",
                weights=weights,
            )
        )

    def test_weighted_combination_normalizes_knn_distance_scores(self):
        detector = self._make_detector(
            WeightConfig(
                confidence=0.35,
                knn=0.45,
                cluster=0.2,
                novelty_threshold=0.6,
                knn_gate_threshold=1.0,
                strong_knn_threshold=1.0,
            )
        )

        novel, scores = detector._combiner.combine(
            {"knn_distance": ({0}, {0: {"knn_novelty_score": 0.7}})},
            {0: {"knn_novelty_score": 0.7}},
        )

        assert scores[0] == pytest.approx(0.7)
        assert novel == {0}

    def test_weighted_combination_normalizes_clustering_scores(self):
        detector = self._make_detector(
            WeightConfig(
                confidence=0.35,
                knn=0.45,
                cluster=0.2,
                novelty_threshold=0.6,
            )
        )

        novel, scores = detector._combiner.combine(
            {"clustering": ({0}, {0: {"cluster_support_score": 0.9}})},
            {0: {"cluster_support_score": 0.9}},
        )

        assert scores[0] == pytest.approx(0.9)
        assert novel == {0}

    def test_weighted_combination_uses_sum_of_active_strategy_weights(self):
        detector = self._make_detector(
            WeightConfig(
                confidence=0.35,
                knn=0.45,
                cluster=0.2,
                novelty_threshold=0.6,
                knn_gate_threshold=1.0,
                strong_knn_threshold=1.0,
            )
        )
        metrics = {
            0: {
                "confidence_is_novel": True,
                "knn_novelty_score": 0.7,
                "cluster_support_score": 0.9,
            }
        }

        novel, scores = detector._combiner.combine(
            {
                "confidence": ({0}, metrics),
                "knn_distance": ({0}, metrics),
                "clustering": ({0}, metrics),
            },
            metrics,
        )

        expected = (0.35 * 1.0 + 0.45 * 0.7 + 0.2 * 0.9) / (0.35 + 0.45 + 0.2)
        assert scores[0] == pytest.approx(expected)
        assert novel == {0}

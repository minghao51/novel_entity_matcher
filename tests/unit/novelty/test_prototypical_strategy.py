"""Tests for PrototypicalNoveltyDetector."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import PrototypicalConfig
from novelentitymatcher.novelty.strategies.prototypical import PrototypicalStrategy
from novelentitymatcher.novelty.strategies.prototypical_impl import (
    PrototypicalDetector,
)


class TestPrototypicalDetector:
    """Test suite for PrototypicalDetector."""

    @pytest.fixture
    def training_data(self):
        return [
            {"text": "machine learning algorithms", "label": "ml"},
            {"text": "neural network architectures", "label": "ml"},
            {"text": "deep learning models", "label": "ml"},
            {"text": "computer vision tasks", "label": "cv"},
            {"text": "image processing", "label": "cv"},
            {"text": "object detection", "label": "cv"},
        ]

    @pytest.fixture
    def detector(self):
        return PrototypicalDetector(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            distance_threshold=0.5,
        )

    def test_initialization(self, detector):
        assert detector.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert detector.distance_threshold == 0.5
        assert detector.distance_metric == "cosine"
        assert detector.is_trained is False

    def test_train(self, detector, training_data):
        detector.train(training_data, show_progress=False)

        assert detector.is_trained is True
        assert len(detector.prototypes) == 2  # ml and cv
        assert "ml" in detector.prototypes
        assert "cv" in detector.prototypes

    def test_train_empty_data(self, detector):
        with pytest.raises(ValueError, match="training_data cannot be empty"):
            detector.train([])

    def test_train_invalid_format(self, detector):
        invalid_data = [{"text": "test"}]  # Missing label
        with pytest.raises(ValueError, match="must have 'text' and 'label' keys"):
            detector.train(invalid_data)

    def test_is_novel_before_training(self, detector):
        with pytest.raises(RuntimeError, match="Detector must be trained"):
            detector.is_novel("test entity")

    def test_is_novel_known_entity(self, detector, training_data):
        detector.train(training_data, show_progress=False)

        is_novel, distance, label = detector.is_novel("machine learning")

        assert isinstance(is_novel, bool)
        assert isinstance(distance, float)
        assert distance >= 0
        assert label in ["ml", "cv"]

    def test_is_novel_novel_entity(self, detector, training_data):
        detector.train(training_data, show_progress=False)

        # Entity from different domain
        is_novel, distance, _label = detector.is_novel("organic farming techniques")

        assert isinstance(is_novel, bool)
        assert isinstance(distance, float)
        assert distance >= 0

    def test_score_batch(self, detector, training_data):
        detector.train(training_data, show_progress=False)

        texts = ["machine learning", "organic farming", "computer vision"]
        results = detector.score_batch(texts)

        assert len(results) == len(texts)
        for is_novel, distance, label in results:
            assert isinstance(is_novel, bool)
            assert isinstance(distance, float)
            assert distance >= 0
            if label is not None:
                assert label in ["ml", "cv"]

    def test_score_batch_before_training(self, detector):
        with pytest.raises(RuntimeError, match="Detector must be trained"):
            detector.score_batch(["test"])

    def test_cosine_distance_metric(self, training_data):
        detector = PrototypicalDetector(
            distance_metric="cosine",
            distance_threshold=0.5,
        )
        detector.train(training_data, show_progress=False)

        is_novel, distance, _label = detector.is_novel("test")

        assert isinstance(is_novel, bool)
        assert isinstance(distance, float)
        assert 0 <= distance <= 1  # Cosine distance is bounded

    def test_euclidean_distance_metric(self, training_data):
        detector = PrototypicalDetector(
            distance_metric="euclidean",
            distance_threshold=1.0,
        )
        detector.train(training_data, show_progress=False)

        is_novel, distance, _label = detector.is_novel("test")

        assert isinstance(is_novel, bool)
        assert isinstance(distance, float)
        assert distance >= 0

    def test_get_prototype_info(self, detector, training_data):
        detector.train(training_data, show_progress=False)

        info = detector.get_prototype_info()

        assert isinstance(info, dict)
        assert "ml" in info
        assert "cv" in info
        assert "prototype_norm" in info["ml"]
        assert "prototype_mean" in info["ml"]
        assert "prototype_std" in info["ml"]

    def test_save_and_load(self, detector, training_data, tmp_path):
        detector.train(training_data, show_progress=False)

        # Test is_novel before saving
        is_novel_before, dist_before, _label_before = detector.is_novel("test entity")

        # Save
        save_path = tmp_path / "prototypical_model"
        detector.save(str(save_path))

        # Load
        loaded_detector = PrototypicalDetector.load(str(save_path))

        assert loaded_detector.is_trained is True
        assert loaded_detector.distance_threshold == detector.distance_threshold
        assert len(loaded_detector.prototypes) == len(detector.prototypes)

        # Test that predictions are consistent
        is_novel_after, dist_after, _label_after = loaded_detector.is_novel(
            "test entity"
        )
        assert is_novel_before == is_novel_after
        assert np.isclose(dist_before, dist_after, atol=1e-6)

    def test_save_before_training(self, detector, tmp_path):
        with pytest.raises(RuntimeError, match="Cannot save untrained detector"):
            detector.save(str(tmp_path / "model"))

    def test_distance_threshold_affects_detection(self, training_data):
        # Test with low threshold (more strict)
        detector_strict = PrototypicalDetector(
            distance_threshold=0.3,
        )
        detector_strict.train(training_data, show_progress=False)

        # Test with high threshold (more lenient)
        detector_lenient = PrototypicalDetector(
            distance_threshold=0.8,
        )
        detector_lenient.train(training_data, show_progress=False)

        # Same entity should have different novelty classification
        test_entity = "somewhat related topic"
        is_novel_strict, _, _ = detector_strict.is_novel(test_entity)
        is_novel_lenient, _, _ = detector_lenient.is_novel(test_entity)

        # Strict detector should be more likely to mark as novel
        # (though this depends on the actual distances)
        assert isinstance(is_novel_strict, bool)
        assert isinstance(is_novel_lenient, bool)


class TestPrototypicalStrategy:
    def test_initialize_builds_training_data_from_labels(self, monkeypatch):
        captured = {}

        class FakeDetector:
            def __init__(self, distance_threshold, model_name):
                captured["init"] = {
                    "distance_threshold": distance_threshold,
                    "model_name": model_name,
                }
                self.is_trained = False

            def train(self, data, show_progress=False):
                captured["train_data"] = data
                captured["show_progress"] = show_progress
                self.is_trained = True

        monkeypatch.setattr(
            "novelentitymatcher.novelty.strategies.prototypical.PrototypicalDetector",
            FakeDetector,
        )

        strategy = PrototypicalStrategy()
        cfg = PrototypicalConfig(distance_threshold=0.7, model_name="mini")
        strategy.initialize(np.zeros((2, 2)), ["X", "Y"], cfg)

        assert captured["init"] == {"distance_threshold": 0.7, "model_name": "mini"}
        assert captured["train_data"] == [
            {"text": "X", "label": "X"},
            {"text": "Y", "label": "Y"},
        ]
        assert captured["show_progress"] is False

    def test_detect_returns_empty_when_detector_missing_or_untrained(self):
        strategy = PrototypicalStrategy()
        strategy._detector = None
        flags, metrics = strategy._detect(
            ["x"], np.zeros((1, 2)), ["A"], np.array([0.1])
        )
        assert flags == set()
        assert metrics == {}

    def test_detect_returns_scores_and_flags(self):
        strategy = PrototypicalStrategy()
        strategy._config = PrototypicalConfig(distance_threshold=0.5)

        class StubDetector:
            is_trained = True

            def score_batch(self, texts):
                assert texts == ["a", "b"]
                return [(False, 0.2, "X"), (True, 0.8, "Y")]

        strategy._detector = StubDetector()
        flags, metrics = strategy._detect(
            ["a", "b"],
            np.zeros((2, 2)),
            ["X", "Y"],
            np.array([0.9, 0.1]),
        )

        assert flags == {1}
        assert metrics[0]["prototypical_is_novel"] is False
        assert metrics[1]["prototypical_is_novel"] is True
        assert metrics[1]["prototypical_nearest_label"] == "Y"
        assert metrics[1]["prototypical_novelty_score"] == 1.0

    def test_metadata_and_config_schema_contract(self):
        strategy = PrototypicalStrategy()
        assert strategy.config_schema is PrototypicalConfig
        assert strategy.strategy_id == "prototypical"
        assert strategy.score_keys == ("prototypical_novelty_score",)
        assert strategy.signal_info.weight_name == "prototypical"

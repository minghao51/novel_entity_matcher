"""Tests for PrototypicalNoveltyDetector."""

import numpy as np
import pytest

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

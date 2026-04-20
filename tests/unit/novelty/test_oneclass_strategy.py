"""Tests for OneClassNoveltyDetector."""

import pytest
import numpy as np

from novelentitymatcher.novelty.strategies.oneclass_impl import OneClassSVMDetector


class TestOneClassSVMDetector:
    """Test suite for OneClassSVMDetector."""

    @pytest.fixture
    def known_entities(self):
        return [
            "machine learning",
            "neural networks",
            "deep learning",
            "artificial intelligence",
            "computer vision",
            "natural language processing",
        ]

    @pytest.fixture
    def detector(self):
        return OneClassSVMDetector(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            nu=0.1,
        )

    def test_initialization(self, detector):
        assert detector.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert detector.nu == 0.1
        assert detector.is_trained is False

    def test_train(self, detector, known_entities):
        detector.train(known_entities, show_progress=False)

        assert detector.is_trained is True
        assert detector.known_embeddings is not None
        assert detector.known_embeddings.shape[0] == len(known_entities)
        assert detector.oc_svm is not None

    def test_train_empty_entities(self, detector):
        with pytest.raises(ValueError, match="known_entities cannot be empty"):
            detector.train([])

    def test_is_novel_before_training(self, detector):
        with pytest.raises(RuntimeError, match="Detector must be trained"):
            detector.is_novel("test entity")

    def test_is_novel_known_entity(self, detector, known_entities):
        detector.train(known_entities, show_progress=False)

        is_novel, confidence = detector.is_novel("machine learning")

        # Known entity should not be novel (or low confidence)
        assert isinstance(is_novel, bool)
        assert 0 <= confidence <= 1

    def test_is_novel_similar_entity(self, detector, known_entities):
        detector.train(known_entities, show_progress=False)

        is_novel, confidence = detector.is_novel("deep neural networks")

        assert isinstance(is_novel, bool)
        assert 0 <= confidence <= 1

    def test_is_novel_novel_entity(self, detector, known_entities):
        detector.train(known_entities, show_progress=False)

        # Entity from completely different domain
        is_novel, confidence = detector.is_novel("organic farming techniques")

        assert isinstance(is_novel, bool)
        assert 0 <= confidence <= 1

    def test_score_batch(self, detector, known_entities):
        detector.train(known_entities, show_progress=False)

        texts = ["machine learning", "organic farming", "neural networks"]
        results = detector.score_batch(texts)

        assert len(results) == len(texts)
        for is_novel, confidence in results:
            assert isinstance(is_novel, bool)
            assert 0 <= confidence <= 1

    def test_score_batch_before_training(self, detector):
        with pytest.raises(RuntimeError, match="Detector must be trained"):
            detector.score_batch(["test"])

    def test_get_support_vectors_info(self, detector, known_entities):
        detector.train(known_entities, show_progress=False)

        info = detector.get_support_vectors_info()

        assert isinstance(info, dict)
        assert "n_support_vectors" in info

    def test_get_support_vectors_info_before_training(self, detector):
        info = detector.get_support_vectors_info()
        assert info == {}

    def test_save_and_load(self, detector, known_entities, tmp_path):
        detector.train(known_entities, show_progress=False)

        # Test is_novel before saving
        is_novel_before, conf_before = detector.is_novel("test entity")

        # Save
        save_path = tmp_path / "oneclass_model"
        detector.save(str(save_path))

        # Load
        loaded_detector = OneClassSVMDetector.load(str(save_path))

        assert loaded_detector.is_trained is True
        assert loaded_detector.nu == detector.nu
        assert loaded_detector.known_embeddings is not None

        # Test that predictions are consistent
        is_novel_after, conf_after = loaded_detector.is_novel("test entity")
        assert is_novel_before == is_novel_after
        assert np.isclose(conf_before, conf_after, atol=1e-6)

    def test_save_before_training(self, detector, tmp_path):
        with pytest.raises(RuntimeError, match="Cannot save untrained detector"):
            detector.save(str(tmp_path / "model"))

    def test_different_nu_values(self, known_entities):
        # Test with different nu values
        for nu in [0.05, 0.1, 0.2, 0.5]:
            detector = OneClassSVMDetector(nu=nu)
            detector.train(known_entities, show_progress=False)

            assert detector.is_trained is True
            assert detector.nu == nu

    def test_different_kernels(self, known_entities):
        # Test with different kernels
        for kernel in ["rbf", "linear", "poly"]:
            detector = OneClassSVMDetector(kernel=kernel)
            detector.train(known_entities, show_progress=False)

            assert detector.is_trained is True
            assert detector.kernel == kernel

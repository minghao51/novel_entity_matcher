"""Tests for OneClassNoveltyDetector."""

import numpy as np
import pytest

from novelentitymatcher.novelty.config.strategies import OneClassConfig
from novelentitymatcher.novelty.strategies.oneclass import OneClassStrategy
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


class TestOneClassStrategy:
    def test_initialize_wires_config_to_detector(self, monkeypatch):
        captured = {}

        class FakeDetector:
            def __init__(self, model_name, nu, kernel, gamma):
                captured["init"] = {
                    "model_name": model_name,
                    "nu": nu,
                    "kernel": kernel,
                    "gamma": gamma,
                }
                self.is_trained = False

            def train(self, labels, show_progress=False):
                captured["train_labels"] = labels
                captured["show_progress"] = show_progress
                self.is_trained = True

        monkeypatch.setattr(
            "novelentitymatcher.novelty.strategies.oneclass.OneClassSVMDetector",
            FakeDetector,
        )

        strategy = OneClassStrategy()
        cfg = OneClassConfig(model_name="m", nu=0.2, kernel="linear", gamma="auto")
        strategy.initialize(np.zeros((2, 3)), ["A", "B"], cfg)

        assert captured["init"] == {
            "model_name": "m",
            "nu": 0.2,
            "kernel": "linear",
            "gamma": "auto",
        }
        assert captured["train_labels"] == ["A", "B"]
        assert captured["show_progress"] is False

    def test_detect_returns_empty_when_untrained(self):
        strategy = OneClassStrategy()

        class StubDetector:
            is_trained = False

        strategy._detector = StubDetector()
        flags, metrics = strategy._detect(
            ["x"], np.zeros((1, 2)), ["A"], np.array([0.9])
        )
        assert flags == set()
        assert metrics == {}

    def test_detect_returns_expected_flags_and_metrics(self):
        strategy = OneClassStrategy()

        class StubDetector:
            is_trained = True

            def score_batch(self, texts):
                assert texts == ["known", "novel"]
                return [(False, 0.2), (True, 0.9)]

        strategy._detector = StubDetector()
        flags, metrics = strategy._detect(
            ["known", "novel"],
            np.zeros((2, 2)),
            ["A", "B"],
            np.array([0.9, 0.1]),
        )

        assert flags == {1}
        assert metrics[0]["oneclass_is_novel"] is False
        assert metrics[1]["oneclass_is_novel"] is True
        assert metrics[1]["oneclass_novelty_score"] == 0.9

    def test_metadata_and_config_schema_contract(self):
        strategy = OneClassStrategy()
        assert strategy.config_schema is OneClassConfig
        assert strategy.strategy_id == "oneclass"
        assert strategy.score_keys == ("oneclass_novelty_score",)
        assert strategy.signal_info.score_key == "oneclass_novelty_score"

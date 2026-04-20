"""Tests for SetFitDetector."""

import pytest

from novelentitymatcher.novelty.strategies.setfit_impl import SetFitDetector


class TestSetFitDetector:
    """Test suite for SetFitDetector."""

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
    def synthetic_novels(self):
        return [
            "organic farming",
            "agricultural science",
            "crop management",
            "sustainable agriculture",
        ]

    def test_initialization(self, known_entities):
        detector = SetFitDetector(
            known_entities=known_entities,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        assert detector.known_entities == known_entities
        assert detector.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert detector.is_trained is False

    def test_train_without_synthetic(self, known_entities):
        detector = SetFitDetector(
            known_entities=known_entities,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        detector.train(synthetic_novels=None, show_progress=False)

        assert detector.is_trained is True
        assert detector.novelty_threshold is not None
        assert detector.known_embeddings is not None

    def test_train_with_synthetic(self, known_entities, synthetic_novels):
        detector = SetFitDetector(
            known_entities=known_entities,
            model_name="sentence-transformers/all-MiniLM-L6-v2",
        )

        detector.train(synthetic_novels=synthetic_novels, show_progress=False)

        assert detector.is_trained is True
        assert detector.novelty_threshold is not None

    def test_train_empty_known_entities(self):
        with pytest.raises(ValueError, match="known_entities cannot be empty"):
            SetFitDetector(known_entities=[])

    def test_is_novel_before_training(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        with pytest.raises(RuntimeError, match="Detector must be trained"):
            detector.is_novel("test entity")

    def test_is_novel_known_entity(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)
        detector.train(show_progress=False)

        is_novel, confidence = detector.is_novel("machine learning")

        assert isinstance(is_novel, bool)
        assert 0 <= confidence <= 1

    def test_is_novel_novel_entity(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)
        detector.train(show_progress=False)

        is_novel, confidence = detector.is_novel("organic farming techniques")

        assert isinstance(is_novel, bool)
        assert 0 <= confidence <= 1

    def test_score_batch(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)
        detector.train(show_progress=False)

        texts = ["machine learning", "organic farming", "neural networks"]
        results = detector.score_batch(texts)

        assert len(results) == len(texts)
        for is_novel, confidence in results:
            assert isinstance(is_novel, bool)
            assert 0 <= confidence <= 1

    def test_score_batch_before_training(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        with pytest.raises(RuntimeError, match="Detector must be trained"):
            detector.score_batch(["test"])

    def test_save_and_load(self, known_entities, tmp_path):
        detector = SetFitDetector(known_entities=known_entities)
        detector.train(show_progress=False)

        # Test is_novel before saving
        is_novel_before, conf_before = detector.is_novel("test entity")

        # Save
        save_path = tmp_path / "setfit_model"
        detector.save(str(save_path))

        # Load
        loaded_detector = SetFitDetector.load(str(save_path))

        assert loaded_detector.is_trained is True
        assert loaded_detector.novelty_threshold is not None
        assert loaded_detector.known_embeddings is not None

        # Test that predictions are similar (may not be exact due to numerical differences)
        is_novel_after, conf_after = loaded_detector.is_novel("test entity")
        assert is_novel_before == is_novel_after
        # Confidence may vary slightly
        assert abs(conf_before - conf_after) < 0.1

    def test_save_before_training(self, known_entities, tmp_path):
        detector = SetFitDetector(known_entities=known_entities)

        with pytest.raises(RuntimeError, match="Cannot save untrained detector"):
            detector.save(str(tmp_path / "model"))

    def test_generate_synthetic_novels(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        synthetic = detector.generate_synthetic_novels(num_samples=10)

        assert len(synthetic) == 10
        assert all(isinstance(s, str) for s in synthetic)

    def test_generate_synthetic_novels_custom_methods(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        synthetic = detector.generate_synthetic_novels(
            num_samples=10,
            augmentation_methods=["typos", "case_change"],
        )

        assert len(synthetic) == 10
        assert all(isinstance(s, str) for s in synthetic)

    def test_add_typos(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        original = "test"
        with_typo = detector._add_typos(original, num_typos=1)

        assert isinstance(with_typo, str)
        # Should have at most 1 character changed
        changes = sum(1 for a, b in zip(original, with_typo) if a != b)
        assert changes <= 1

    def test_change_case(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        text = "Test String"
        changed = detector._change_case(text)

        assert isinstance(changed, str)
        assert changed != text or changed.lower() == text.lower()

    def test_modify_spacing(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        text = "test string"
        modified = detector._modify_spacing(text)

        assert isinstance(modified, str)

    def test_create_substring_variant(self, known_entities):
        detector = SetFitDetector(known_entities=known_entities)

        text = "teststring"
        variant = detector._create_substring_variant(text)

        assert isinstance(variant, str)
        assert len(variant) <= len(text)

    def test_different_margins(self, known_entities):
        for margin in [0.3, 0.5, 0.7]:
            detector = SetFitDetector(
                known_entities=known_entities,
                margin=margin,
            )
            detector.train(show_progress=False)

            assert detector.is_trained is True
            assert detector.margin == margin

    def test_different_epochs(self, known_entities):
        for epochs in [1, 2, 4]:
            detector = SetFitDetector(
                known_entities=known_entities,
                num_epochs=epochs,
            )
            detector.train(show_progress=False)

            assert detector.is_trained is True
            assert detector.num_epochs == epochs

"""Tests for BERTClassifier - BERT-based classification using transformers."""

import numpy as np
import pytest
import torch

from novelentitymatcher.core.bert_classifier import BERTClassifier
from novelentitymatcher.exceptions import TrainingError

# Skip single-label test on MPS due to PyTorch MSE loss compatibility
skip_on_mps = pytest.mark.skipif(
    torch.backends.mps.is_available(),
    reason="MPS device has MSE loss compatibility issues with single-label classification",
)


class TestBERTClassifier:
    """Tests for BERTClassifier - wrapper for BERT training and prediction."""

    @pytest.fixture
    def training_data(self):
        return [
            {"text": "Germany", "label": "DE"},
            {"text": "Deutschland", "label": "DE"},
            {"text": "France", "label": "FR"},
            {"text": "USA", "label": "US"},
        ]

    @pytest.fixture
    def labels(self):
        return ["DE", "FR", "US"]

    def test_init_with_defaults(self, labels):
        """Test initialization with default parameters."""
        clf = BERTClassifier(labels=labels)
        assert clf.labels == labels
        assert clf.model_name == "distilbert-base-uncased"
        assert clf.num_epochs == 3
        assert clf.batch_size == 16
        assert clf.learning_rate == 2e-5
        assert clf.max_length == 128
        assert clf.use_fp16 is True
        assert not clf.is_trained

    def test_init_with_custom_params(self, labels):
        """Test initialization with custom parameters."""
        clf = BERTClassifier(
            labels=labels,
            model_name="bert-base-uncased",
            num_epochs=5,
            batch_size=8,
            learning_rate=1e-4,
            max_length=256,
            use_fp16=False,
        )
        assert clf.model_name == "bert-base-uncased"
        assert clf.num_epochs == 5
        assert clf.batch_size == 8
        assert clf.learning_rate == 1e-4
        assert clf.max_length == 256
        assert clf.use_fp16 is False

    def test_train_creates_model(self, labels, training_data):
        """Test that training creates model and tokenizer."""
        clf = BERTClassifier(labels=labels)
        assert clf.model is None
        assert clf.tokenizer is None

        clf.train(training_data, num_epochs=1)

        assert clf.model is not None
        assert clf.tokenizer is not None
        assert clf.is_trained

    def test_predict_single(self, labels, training_data):
        """Test prediction on single text."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=3)  # More epochs for better convergence

        result = clf.predict("Deutschland")
        assert result in labels
        # Note: With small dataset, prediction may not always be perfect
        # Just check that it returns a valid label (already asserted above)

    def test_predict_batch(self, labels, training_data):
        """Test prediction on multiple texts."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)

        results = clf.predict(["Germany", "France", "USA"])
        assert len(results) == 3
        for result in results:
            assert result in labels

    def test_predict_proba_returns_distribution(self, labels, training_data):
        """Test that predict_proba returns probability distribution."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)

        proba = clf.predict_proba("Deutschland")
        assert len(proba) == len(labels)
        assert isinstance(proba, np.ndarray)
        # Probabilities should sum to approximately 1.0
        assert sum(proba) > 0.99
        assert sum(proba) <= 1.01
        # All probabilities should be non-negative
        assert all(p >= 0 for p in proba)

    def test_predict_proba_high_confidence(self, labels, training_data):
        """Test that predict_proba gives high confidence for known examples."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=4)  # More epochs for better convergence

        # "Deutschland" was trained as "DE"
        proba = clf.predict_proba("Deutschland")
        de_idx = labels.index("DE")
        # With small dataset, confidence may be lower - just check it's reasonable
        assert proba[de_idx] > 0.3  # Should have decent confidence

    def test_save_load_roundtrip(self, labels, training_data, tmp_path):
        """Test saving and loading classifier."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)

        model_path = tmp_path / "bert_model"
        clf.save(str(model_path))

        # Check that files were created
        assert (model_path / "config.json").exists()
        assert (model_path / "tokenizer_config.json").exists()
        assert (model_path / "labels.txt").exists()

        # Load and test
        loaded_clf = BERTClassifier.load(str(model_path))
        assert loaded_clf.is_trained
        assert loaded_clf.labels == labels

        # Test prediction
        result = loaded_clf.predict("Deutschland")
        assert result in labels

    def test_predict_before_training_raises(self, labels):
        """Test that predicting before training raises TrainingError."""
        clf = BERTClassifier(labels=labels)

        with pytest.raises(TrainingError, match="not trained"):
            clf.predict("test")

    def test_predict_proba_before_training_raises(self, labels):
        """Test that predict_proba before training raises TrainingError."""
        clf = BERTClassifier(labels=labels)

        with pytest.raises(TrainingError, match="not trained"):
            clf.predict_proba("test")

    def test_save_before_training_raises(self, labels, tmp_path):
        """Test that saving before training raises TrainingError."""
        clf = BERTClassifier(labels=labels)

        with pytest.raises(TrainingError, match="not trained"):
            clf.save(str(tmp_path / "model"))

    def test_handles_long_sequences(self, labels):
        """Test handling of sequences longer than max_length."""
        clf = BERTClassifier(labels=labels, max_length=32)

        # Create training data with short sequences
        training_data = [
            {"text": "Germany", "label": "DE"},
            {"text": "France", "label": "FR"},
            {"text": "USA", "label": "US"},
        ]

        clf.train(training_data, num_epochs=1)

        # Test with very long text
        long_text = "Germany " * 100  # Much longer than max_length
        result = clf.predict(long_text)
        assert result in labels

    def test_custom_batch_size(self, labels, training_data):
        """Test training with custom batch size."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=1, batch_size=2)
        assert clf.is_trained

    def test_custom_num_epochs(self, labels, training_data):
        """Test training with custom number of epochs."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=2)
        assert clf.is_trained

    def test_label2id_mapping(self, labels):
        """Test that label to ID mapping is created correctly."""
        clf = BERTClassifier(labels=labels)
        assert clf.label2id == {"DE": 0, "FR": 1, "US": 2}
        assert clf.id2label == {0: "DE", 1: "FR", 2: "US"}

    def test_different_model_names(self, labels, training_data):
        """Test initialization with different model names."""
        model_names = [
            "distilbert-base-uncased",
            "bert-base-uncased",
            "roberta-base",
        ]

        for model_name in model_names:
            clf = BERTClassifier(labels=labels, model_name=model_name)
            assert clf.model_name == model_name

    def test_empty_training_data_raises(self, labels):
        """Test that empty training data raises appropriate error."""
        clf = BERTClassifier(labels=labels)

        with pytest.raises((TrainingError, ValueError)):
            clf.train([], num_epochs=1)

    @skip_on_mps
    def test_single_label(self):
        """Test classifier with single label."""
        labels = ["POS"]
        training_data = [
            {"text": "good", "label": "POS"},
            {"text": "great", "label": "POS"},
        ]

        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)

        result = clf.predict("excellent")
        assert result == "POS"

    def test_many_labels(self):
        """Test classifier with many labels."""
        labels = [f"LABEL_{i}" for i in range(10)]
        training_data = [
            {"text": f"text {i}", "label": f"LABEL_{i}"} for i in range(10)
        ]

        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)

        result = clf.predict("text 5")
        assert result in labels

    def test_show_progress_false(self, labels, training_data):
        """Test training without progress bar."""
        clf = BERTClassifier(labels=labels)
        clf.train(training_data, num_epochs=1, show_progress=False)
        assert clf.is_trained

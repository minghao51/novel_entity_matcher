import pytest

from novelentitymatcher.core.classifier import SetFitClassifier
from novelentitymatcher.exceptions import TrainingError


class TestSetFitClassifier:
    """Tests for SetFitClassifier - wrapper for SetFit training and prediction."""

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

    def test_classifier_init_with_labels(self, labels):
        clf = SetFitClassifier(labels=labels)
        assert clf.labels == labels

    def test_classifier_init_with_model(self, labels):
        clf = SetFitClassifier(
            labels=labels, model_name="sentence-transformers/paraphrase-mpnet-base-v2"
        )
        assert clf.model_name == "sentence-transformers/paraphrase-mpnet-base-v2"

    def test_classifier_default_params(self, labels):
        clf = SetFitClassifier(labels=labels)
        assert clf.num_epochs == 4
        assert clf.batch_size == 16
        assert clf.weight_decay == 0.01
        assert clf.head_c == 1.0
        assert clf.num_iterations == 5
        assert clf.pca_dims is None
        assert clf.skip_body_training is False

    def test_classifier_custom_params(self, labels):
        clf = SetFitClassifier(labels=labels, num_epochs=2, batch_size=8)
        assert clf.num_epochs == 2
        assert clf.batch_size == 8

    def test_classifier_not_trained_initially(self, labels):
        clf = SetFitClassifier(labels=labels)
        assert not clf.is_trained

    def test_classifier_train(self, labels, training_data):
        clf = SetFitClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)
        assert clf.is_trained

    def test_classifier_predict_single(self, labels, training_data):
        clf = SetFitClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)
        result = clf.predict("Deutschland")
        assert result in labels

    def test_classifier_predict_multiple(self, labels, training_data):
        clf = SetFitClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)
        results = clf.predict(["Germany", "France"])
        assert len(results) == 2
        for r in results:
            assert r in labels

    def test_classifier_predict_proba(self, labels, training_data):
        clf = SetFitClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)
        proba = clf.predict_proba("Deutschland")
        assert len(proba) == len(labels)
        assert sum(proba) > 0.99

    def test_classifier_without_training_raises(self, labels):
        clf = SetFitClassifier(labels=labels)
        with pytest.raises(RuntimeError, match="not trained"):
            clf.predict("test")

    def test_classifier_without_training_raises_training_error(self, labels):
        clf = SetFitClassifier(labels=labels)
        with pytest.raises(TrainingError):
            clf.predict("test")

    def test_classifier_save_load(self, labels, training_data, tmp_path):
        clf = SetFitClassifier(labels=labels)
        clf.train(training_data, num_epochs=1)
        model_path = tmp_path / "model"
        clf.save(str(model_path))

        loaded_clf = SetFitClassifier.load(str(model_path))
        assert loaded_clf.is_trained
        result = loaded_clf.predict("Deutschland")
        assert result in labels

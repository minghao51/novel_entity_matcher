"""Tests for SetFitClassifier.add_class and retrain_head."""

import pytest

from novelentitymatcher.core.classifier import SetFitClassifier


@pytest.fixture
def trained_classifier():
    clf = SetFitClassifier(
        labels=["fruit", "vegetable"],
        model_name="sentence-transformers/paraphrase-mpnet-base-v2",
        skip_body_training=True,
    )
    clf.train(
        [
            {"text": "apple", "label": "fruit"},
            {"text": "banana", "label": "fruit"},
            {"text": "carrot", "label": "vegetable"},
            {"text": "broccoli", "label": "vegetable"},
        ]
    )
    return clf


class TestAddClass:
    def test_add_class_updates_labels(self, trained_classifier):
        trained_classifier.add_class("grain", ["rice", "wheat"])
        assert "grain" in trained_classifier.labels

    def test_add_class_new_class_predictable(self, trained_classifier):
        trained_classifier.add_class("grain", ["rice", "wheat"])
        pred = trained_classifier.predict("rice")
        assert pred == "grain"

    def test_add_class_keeps_old_classes(self, trained_classifier):
        trained_classifier.add_class("grain", ["rice", "wheat"])
        pred = trained_classifier.predict("apple")
        assert pred == "fruit"

    def test_add_class_duplicate_skips(self, trained_classifier):
        before_count = len(trained_classifier.labels)
        trained_classifier.add_class("fruit", ["orange"])
        assert len(trained_classifier.labels) == before_count

    def test_add_class_raises_if_not_trained(self):
        clf = SetFitClassifier(
            labels=["a", "b"],
            model_name="sentence-transformers/paraphrase-mpnet-base-v2",
            skip_body_training=True,
        )
        with pytest.raises(RuntimeError, match="Model not trained"):
            clf.add_class("c", ["example"])

    def test_retrain_head_updates_predictions(self, trained_classifier):
        trained_classifier.add_class("grain", ["rice", "wheat"])
        trained_classifier.retrain_head()

        pred = trained_classifier.predict("rice")
        assert pred == "grain"
        pred_old = trained_classifier.predict("apple")
        assert pred_old == "fruit"

    def test_retrain_head_no_pending_does_nothing(self, trained_classifier):
        trained_classifier.retrain_head()
        assert trained_classifier.predict("apple") == "fruit"

    def test_retrain_head_multiple_new_classes(self, trained_classifier):
        trained_classifier.add_class("grain", ["rice", "wheat"])
        trained_classifier.add_class("dairy", ["milk", "cheese"])
        trained_classifier.retrain_head()

        assert trained_classifier.predict("rice") == "grain"
        assert trained_classifier.predict("milk") == "dairy"
        assert trained_classifier.predict("apple") == "fruit"

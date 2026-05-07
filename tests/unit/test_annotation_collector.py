"""Tests for AnnotationCollector."""

import pytest

from novelentitymatcher import Matcher
from novelentitymatcher.novelty.active_learning.annotation import (
    AnnotationCollector,
    AnnotationResult,
)


@pytest.fixture
def matcher():
    entities = [
        {"id": "fruit", "name": "Fruit"},
        {"id": "vegetable", "name": "Vegetable"},
    ]
    m = Matcher(entities=entities, model="minilm", threshold=0.5)
    m.fit(
        texts=["apple", "banana", "carrot", "broccoli"],
        labels=["fruit", "fruit", "vegetable", "vegetable"],
    )
    return m


class TestAnnotationCollector:
    def test_existing_class_adds_training_data(self, matcher):
        collector = AnnotationCollector(matcher)
        anns = [
            AnnotationResult(text="orange", assigned_label="fruit", annotator="test"),
        ]
        stats = collector.apply_annotations(anns, retrain=False)
        assert stats["existing"] == 1
        assert stats["novel"] == 0

    def test_novel_class_adds_entity(self, matcher):
        collector = AnnotationCollector(matcher)
        anns = [
            AnnotationResult(text="wheat", assigned_label="novel:grain"),
        ]
        collector.apply_annotations(anns, retrain=False)
        count = sum(1 for e in matcher.entities if e["id"] == "grain")
        assert count == 1

    def test_retrain_with_new_class(self, matcher):
        collector = AnnotationCollector(matcher)
        anns = [
            AnnotationResult(text="rice", assigned_label="novel:grain"),
            AnnotationResult(text="wheat", assigned_label="novel:grain"),
        ]
        stats = collector.apply_annotations(anns, retrain=True)
        assert "grain" in stats["new_classes"]

    def test_unknown_existing_label_skipped(self, matcher):
        collector = AnnotationCollector(matcher)
        anns = [
            AnnotationResult(text="something", assigned_label="nonexistent"),
        ]
        stats = collector.apply_annotations(anns, retrain=False)
        assert stats["existing"] == 1

    def test_empty_annotations(self, matcher):
        collector = AnnotationCollector(matcher)
        stats = collector.apply_annotations([])
        assert stats["total"] == 0

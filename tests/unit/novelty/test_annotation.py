from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from novelentitymatcher.novelty.active_learning.annotation import (
    AnnotationCollector,
    AnnotationResult,
)


class TestAnnotationResult:
    def test_default_annotator(self):
        r = AnnotationResult(text="foo", assigned_label="bar")
        assert r.annotator == "human"
        assert r.metadata == {}


class TestAnnotationCollector:
    @pytest.fixture
    def matcher(self):
        m = MagicMock()
        m.entities = [
            {"id": "tech", "name": "Technology"},
            {"id": "health", "name": "Healthcare"},
        ]
        return m

    def test_novel_class_creates_entity(self, matcher):
        collector = AnnotationCollector(matcher)
        annotations = [
            AnnotationResult(text="Biotech Startup", assigned_label="novel:Biotech"),
        ]
        stats = collector.apply_annotations(annotations, retrain=False)
        matcher.add_entities.assert_called_once_with(
            [{"id": "Biotech", "name": "Biotech"}]
        )
        assert stats["novel"] == 1
        assert stats["existing"] == 0
        assert "Biotech" in stats["new_classes"]

    def test_existing_class_does_not_create_entity(self, matcher):
        collector = AnnotationCollector(matcher)
        annotations = [
            AnnotationResult(text="AI Company", assigned_label="tech"),
        ]
        stats = collector.apply_annotations(annotations, retrain=False)
        matcher.add_entities.assert_not_called()
        assert stats["existing"] == 1
        assert stats["novel"] == 0

    def test_mixed_annotations(self, matcher):
        collector = AnnotationCollector(matcher)
        annotations = [
            AnnotationResult(text="AI Company", assigned_label="tech"),
            AnnotationResult(text="Fintech Startup", assigned_label="novel:Fintech"),
        ]
        stats = collector.apply_annotations(annotations, retrain=False)
        assert stats["total"] == 2
        assert stats["novel"] == 1
        assert stats["existing"] == 1
        assert stats["new_classes"] == ["Fintech"]

    def test_duplicate_novel_class_not_added_twice(self, matcher):
        collector = AnnotationCollector(matcher)
        annotations = [
            AnnotationResult(text="A", assigned_label="novel:X"),
            AnnotationResult(text="B", assigned_label="novel:X"),
        ]
        stats = collector.apply_annotations(annotations, retrain=False)
        matcher.add_entities.assert_called_once_with([{"id": "X", "name": "X"}])
        assert stats["new_classes"] == ["X"]

    def test_empty_annotations(self, matcher):
        collector = AnnotationCollector(matcher)
        stats = collector.apply_annotations([], retrain=False)
        assert stats["total"] == 0
        matcher.add_entities.assert_not_called()

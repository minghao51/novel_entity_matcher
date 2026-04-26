"""Tests for existing_classes post-detection override in build_novel_match_result."""

from types import SimpleNamespace

import numpy as np

from novelentitymatcher.pipeline.discovery_support import build_novel_match_result
from novelentitymatcher.pipeline.match_result import (
    MatchRecord,
    MatchResultWithMetadata,
)


def _make_match_result(
    predicted_id: str = "physics",
    confidence: float = 0.9,
) -> MatchResultWithMetadata:
    return MatchResultWithMetadata(
        predictions=[predicted_id],
        confidences=np.array([confidence], dtype=float),
        embeddings=np.array([[0.1, 0.2, 0.3]]),
        records=[
            MatchRecord(
                text="test query",
                predicted_id=predicted_id,
                confidence=confidence,
                embedding=np.array([0.1, 0.2, 0.3]),
            )
        ],
        candidate_results=[[]],
    )


def _mock_detector(is_novel: bool = False, novelty_score: float = 0.1):
    sample = (
        SimpleNamespace(
            novelty_score=novelty_score,
            signals={"confidence": is_novel, "knn": is_novel},
        )
        if is_novel
        else None
    )

    report = SimpleNamespace(novel_samples=[sample] if sample else [])
    return SimpleNamespace(detect_novel_samples=lambda **kwargs: report)


def _reference_corpus():
    return {
        "embeddings": np.array([[0.1, 0.2, 0.3]]),
        "labels": ["physics"],
    }


class TestExistingClassesOverride:
    def test_none_existing_classes_no_override(self):
        result = build_novel_match_result(
            query="test",
            match_result=_make_match_result("physics", 0.9),
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=False),
            use_novelty_detector=True,
            acceptance_threshold=0.5,
            existing_classes=None,
        )
        assert result.is_match is True
        assert result.is_novel is False
        assert result.match_method == "accepted_known"
        assert result.id == "physics"

    def test_predicted_in_existing_classes_accepted(self):
        result = build_novel_match_result(
            query="test",
            match_result=_make_match_result("physics", 0.9),
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=False),
            use_novelty_detector=True,
            acceptance_threshold=0.5,
            existing_classes=["physics", "chemistry"],
        )
        assert result.is_match is True
        assert result.is_novel is False
        assert result.match_method == "accepted_known"
        assert result.id == "physics"

    def test_predicted_outside_existing_classes_rejected(self):
        result = build_novel_match_result(
            query="test",
            match_result=_make_match_result("biology", 0.9),
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=False),
            use_novelty_detector=True,
            acceptance_threshold=0.5,
            existing_classes=["physics", "chemistry"],
        )
        assert result.is_match is False
        assert result.is_novel is True
        assert result.match_method == "outside_existing_classes"
        assert result.id is None
        assert result.predicted_id == "biology"

    def test_predicted_unknown_not_overridden(self):
        result = build_novel_match_result(
            query="test",
            match_result=_make_match_result("unknown", 0.3),
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=False),
            use_novelty_detector=True,
            acceptance_threshold=0.5,
            existing_classes=["physics"],
        )
        assert result.is_match is False
        assert result.match_method == "no_match"

    def test_predicted_none_not_overridden(self):
        match_result = _make_match_result("something", 0.3)
        match_result.records[0].predicted_id = ""
        result = build_novel_match_result(
            query="test",
            match_result=match_result,
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=False),
            use_novelty_detector=True,
            acceptance_threshold=0.5,
            existing_classes=["physics"],
        )
        assert result.is_match is False

    def test_novelty_detector_and_outside_existing_classes(self):
        result = build_novel_match_result(
            query="test",
            match_result=_make_match_result("biology", 0.9),
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=True, novelty_score=0.8),
            use_novelty_detector=True,
            acceptance_threshold=0.5,
            existing_classes=["physics"],
        )
        assert result.is_match is False
        assert result.is_novel is True
        assert result.match_method == "outside_existing_classes"

    def test_below_threshold_outside_existing_classes(self):
        result = build_novel_match_result(
            query="test",
            match_result=_make_match_result("biology", 0.3),
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=False),
            use_novelty_detector=True,
            acceptance_threshold=0.5,
            existing_classes=["physics"],
        )
        assert result.is_match is False
        assert result.is_novel is True
        assert result.match_method == "outside_existing_classes"

    def test_no_novelty_detector_outside_existing_classes(self):
        result = build_novel_match_result(
            query="test",
            match_result=_make_match_result("biology", 0.9),
            reference_corpus=_reference_corpus(),
            detector=_mock_detector(is_novel=False),
            use_novelty_detector=False,
            acceptance_threshold=0.5,
            existing_classes=["physics"],
        )
        assert result.is_match is False
        assert result.is_novel is True
        assert result.match_method == "outside_existing_classes"

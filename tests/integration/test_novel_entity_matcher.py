"""Tests for the promoted NovelEntityMatcher orchestration API."""

from types import SimpleNamespace

import pytest

import novelentitymatcher
from novelentitymatcher import Matcher, NovelEntityMatcher
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import ConfidenceConfig


@pytest.fixture
def sample_entities():
    return [
        {"id": "physics", "name": "Quantum Physics"},
        {"id": "cs", "name": "Computer Science"},
        {"id": "biology", "name": "Molecular Biology"},
    ]


@pytest.fixture
def trained_matcher(sample_entities):
    matcher = Matcher(entities=sample_entities, model="minilm", threshold=0.6)
    matcher.fit(
        texts=[
            "quantum mechanics",
            "wave function",
            "machine learning",
            "data structures",
            "gene expression",
            "DNA replication",
        ],
        labels=["physics", "physics", "cs", "cs", "biology", "biology"],
    )
    return matcher


class TestNovelEntityMatcher:
    def test_root_export_is_available(self):
        assert NovelEntityMatcher is not None

    def test_match_returns_operational_result(self, trained_matcher):
        novelty_matcher = NovelEntityMatcher(
            matcher=trained_matcher,
            detection_config=DetectionConfig(
                strategies=["confidence"],
                confidence=ConfidenceConfig(threshold=0.7),
            ),
            auto_save=False,
        )

        result = novelty_matcher.match(
            "quantum superposition", return_alternatives=True
        )

        assert result.predicted_id is not None
        assert isinstance(result.is_novel, bool)
        assert isinstance(result.alternatives, list)

    @pytest.mark.asyncio
    async def test_discover_novel_classes_returns_report(self, trained_matcher):
        novelty_matcher = NovelEntityMatcher(
            matcher=trained_matcher,
            detection_config=DetectionConfig(
                strategies=["confidence"],
            ),
            auto_save=False,
        )

        report = await novelty_matcher.discover_novel_classes(
            queries=["quantum superposition", "random unrelated text here"],
            existing_classes=["physics", "cs", "biology"],
            run_llm_proposal=False,
        )

        assert report.discovery_id
        assert report.novel_sample_report is not None
        assert isinstance(report.discovery_clusters, list)

    def test_threshold_updates_delegate_to_inner_matcher(self, trained_matcher):
        novelty_matcher = NovelEntityMatcher(matcher=trained_matcher, auto_save=False)

        novelty_matcher.adjust_threshold(0.42)

        assert novelty_matcher.acceptance_threshold == 0.42
        assert trained_matcher.threshold == 0.42

    def test_removed_root_aliases_are_not_available(self):
        for attr in [
            "EntityMatcher",
            "EmbeddingMatcher",
            "HybridMatcher",
        ]:
            assert not hasattr(novelentitymatcher, attr)

    def test_below_threshold_known_prediction_is_not_novel_when_detector_disabled(
        self, trained_matcher
    ):
        novelty_matcher = NovelEntityMatcher(
            matcher=trained_matcher,
            auto_save=False,
            use_novelty_detector=False,
        )
        novelty_matcher.adjust_threshold(0.99)

        result = novelty_matcher.match("quantum superposition")

        assert result.predicted_id == "physics"
        assert result.is_match is False
        assert result.is_novel is False
        assert result.match_method == "below_acceptance_threshold"

    def test_below_threshold_known_prediction_is_not_novel_without_detector_signal(
        self, trained_matcher
    ):
        novelty_matcher = NovelEntityMatcher(
            matcher=trained_matcher,
            detection_config=DetectionConfig(
                strategies=["confidence"],
                confidence=ConfidenceConfig(threshold=0.0),
            ),
            auto_save=False,
        )
        novelty_matcher.adjust_threshold(0.99)
        novelty_matcher.detector.detect_novel_samples = lambda **kwargs: (
            SimpleNamespace(novel_samples=[])
        )

        result = novelty_matcher.match("quantum superposition")

        assert result.predicted_id == "physics"
        assert result.is_match is False
        assert result.is_novel is False
        assert result.match_method == "below_acceptance_threshold"

    def test_detector_flag_still_marks_prediction_as_novel(self, trained_matcher):
        novelty_matcher = NovelEntityMatcher(
            matcher=trained_matcher,
            auto_save=False,
        )
        novelty_matcher.detector.detect_novel_samples = lambda **kwargs: (
            SimpleNamespace(
                novel_samples=[
                    SimpleNamespace(
                        novelty_score=0.91,
                        signals={"confidence": True},
                    )
                ]
            )
        )

        result = novelty_matcher.match("quantum superposition")

        assert result.is_match is False
        assert result.is_novel is True
        assert result.match_method == "novelty_detector"
        assert result.novel_score == pytest.approx(0.91)
        assert result.signals == {"confidence": True}

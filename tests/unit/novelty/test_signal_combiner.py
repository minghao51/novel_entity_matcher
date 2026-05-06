"""Tests for SignalCombiner combination methods (union, intersection, voting)."""

import pytest

from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.weights import WeightConfig
from novelentitymatcher.novelty.core.detector import NoveltyDetector


class TestSignalCombinerUnion:
    """Tests for union combination method."""

    def test_union_combination_flags_if_any_strategy_flags(self):
        """Union should flag sample if any strategy flags it."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                combine_method="union",
            )
        )

        strategy_outputs = {
            "confidence": ({0, 2}, {}),
            "knn_distance": ({1, 2}, {}),
        }

        novel_indices, novelty_scores = detector._combiner.combine(strategy_outputs, {})

        assert novel_indices == {0, 1, 2}
        assert novelty_scores[0] == 1.0
        assert novelty_scores[1] == 1.0
        assert novelty_scores[2] == 1.0

    def test_union_combination_empty_when_no_flags(self):
        """Union should return empty set when no strategies flag anything."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                combine_method="union",
            )
        )

        novel_indices, novelty_scores = detector._combiner.combine({}, {})

        assert novel_indices == set()
        assert novelty_scores == {}

    def test_union_combination_single_strategy(self):
        """Union with single strategy should work correctly."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence"],
                combine_method="union",
            )
        )

        novel_indices, _novelty_scores = detector._combiner.combine(
            {"confidence": ({0, 1, 2}, {})},
            {},
        )

        assert novel_indices == {0, 1, 2}


class TestSignalCombinerIntersection:
    """Tests for intersection combination method."""

    def test_intersection_combination_flags_only_if_all_strategies_flag(self):
        """Intersection should only flag samples flagged by ALL strategies."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance", "clustering"],
                combine_method="intersection",
            )
        )

        strategy_outputs = {
            "confidence": ({0, 1, 2}, {}),
            "knn_distance": ({1, 2, 3}, {}),
            "clustering": ({2, 3, 4}, {}),
        }

        novel_indices, novelty_scores = detector._combiner.combine(strategy_outputs, {})

        assert novel_indices == {2}
        assert novelty_scores[2] == 1.0

    def test_intersection_combination_empty_when_no_overlap(self):
        """Intersection should return empty when no samples flagged by all."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                combine_method="intersection",
            )
        )

        strategy_outputs = {
            "confidence": ({0}, {}),
            "knn_distance": ({1}, {}),
        }

        novel_indices, novelty_scores = detector._combiner.combine(strategy_outputs, {})

        assert novel_indices == set()
        assert novelty_scores == {}

    def test_intersection_combination_empty_when_no_strategies(self):
        """Intersection should return empty set when no strategies provided."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=[],
                combine_method="intersection",
            )
        )

        novel_indices, novelty_scores = detector._combiner.combine({}, {})

        assert novel_indices == set()
        assert novelty_scores == {}


class TestSignalCombinerVoting:
    """Tests for voting combination method."""

    def test_voting_combination_flags_majority(self):
        """Voting should flag samples flagged by majority of strategies."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=[
                    "confidence",
                    "knn_distance",
                    "clustering",
                    "pattern",
                    "oneclass",
                ],
                combine_method="voting",
            )
        )

        strategy_outputs = {
            "confidence": ({0, 1}, {}),
            "knn_distance": ({0, 1}, {}),
            "clustering": ({0}, {}),
            "pattern": ({0}, {}),
            "oneclass": ({0}, {}),
        }

        novel_indices, novelty_scores = detector._combiner.combine(strategy_outputs, {})

        assert novel_indices == {0}
        assert novelty_scores[0] == 1.0

    def test_voting_combination_empty_when_no_majority(self):
        """Voting should return empty when no sample reaches majority."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance"],
                combine_method="voting",
            )
        )

        strategy_outputs = {
            "confidence": ({0}, {}),
            "knn_distance": ({1}, {}),
        }

        novel_indices, novelty_scores = detector._combiner.combine(strategy_outputs, {})

        assert novel_indices == set()
        assert novelty_scores == {0: 0.5, 1: 0.5}

    def test_voting_combination_score_is_fraction_of_votes(self):
        """Voting score should be fraction of strategies that flagged."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance", "clustering", "oneclass"],
                combine_method="voting",
            )
        )

        strategy_outputs = {
            "confidence": ({0}, {}),
            "knn_distance": ({0}, {}),
            "clustering": ({0}, {}),
            "oneclass": ({1}, {}),
        }

        _novel_indices, novelty_scores = detector._combiner.combine(
            strategy_outputs, {}
        )

        assert novelty_scores[0] == 3 / 4
        assert novelty_scores[1] == 1 / 4

    def test_voting_combination_empty_when_no_strategies(self):
        """Voting should return empty when no strategies provided."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=[],
                combine_method="voting",
            )
        )

        novel_indices, novelty_scores = detector._combiner.combine({}, {})

        assert novel_indices == set()
        assert novelty_scores == {}


class TestSignalCombinerWeighted:
    """Tests for weighted combination method (complements existing regression tests)."""

    def test_weighted_combination_empty_when_no_strategies(self):
        """Weighted should return empty when no strategies provided."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=[],
                combine_method="weighted",
            )
        )

        novel_indices, novelty_scores = detector._combiner.combine({}, {})

        assert novel_indices == set()
        assert novelty_scores == {}

    def test_weighted_combination_filters_by_threshold(self):
        """Weighted should only flag samples above novelty threshold."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["knn_distance"],
                combine_method="weighted",
                weights=WeightConfig(
                    confidence=0.0,
                    knn=1.0,
                    novelty_threshold=0.7,
                    knn_gate_threshold=1.0,
                    strong_knn_threshold=1.0,
                ),
            )
        )

        strategy_outputs = {
            "knn_distance": (
                {0, 1},
                {
                    0: {"knn_novelty_score": 0.5},
                    1: {"knn_novelty_score": 0.8},
                },
            )
        }
        all_metrics = {
            0: {"knn_novelty_score": 0.5},
            1: {"knn_novelty_score": 0.8},
        }

        novel_indices, _novelty_scores = detector._combiner.combine(
            strategy_outputs, all_metrics
        )

        assert 1 in novel_indices
        assert 0 not in novel_indices

    def test_weighted_combination_includes_mahalanobis_and_lof_scores(self):
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["mahalanobis", "lof"],
                combine_method="weighted",
                weights=WeightConfig(
                    confidence=0.0,
                    uncertainty=0.0,
                    knn=0.0,
                    cluster=0.0,
                    self_knowledge=0.0,
                    pattern=0.0,
                    oneclass=0.0,
                    prototypical=0.0,
                    setfit=0.0,
                    setfit_centroid=0.0,
                    mahalanobis=0.7,
                    lof=0.3,
                    novelty_threshold=0.6,
                    knn_gate_threshold=1.0,
                    strong_knn_threshold=1.0,
                    strong_uncertainty_threshold=1.0,
                ),
                allowed_maturities=["production", "experimental"],
            )
        )

        strategy_outputs = {
            "mahalanobis": ({0}, {0: {"mahalanobis_novelty_score": 0.9}}),
            "lof": ({0}, {0: {"lof_novelty_score": 0.5}}),
        }
        all_metrics = {
            0: {
                "mahalanobis_novelty_score": 0.9,
                "lof_novelty_score": 0.5,
            }
        }

        novel_indices, novelty_scores = detector._combiner.combine(
            strategy_outputs, all_metrics
        )

        assert 0 in novel_indices
        assert novelty_scores[0] == pytest.approx(0.78)

    def test_weighted_combination_includes_phase67_strategies(self):
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["energy_ood", "mixture_gaussian", "react_energy"],
                combine_method="weighted",
                weights=WeightConfig(
                    confidence=0.0,
                    uncertainty=0.0,
                    knn=0.0,
                    cluster=0.0,
                    self_knowledge=0.0,
                    pattern=0.0,
                    oneclass=0.0,
                    prototypical=0.0,
                    setfit=0.0,
                    setfit_centroid=0.0,
                    mahalanobis=0.0,
                    lof=0.0,
                    energy_ood=0.2,
                    mixture_gaussian=0.3,
                    react_energy=0.5,
                    novelty_threshold=0.6,
                    knn_gate_threshold=1.0,
                    strong_knn_threshold=1.0,
                    strong_uncertainty_threshold=1.0,
                ),
                allowed_maturities=["production", "experimental"],
            )
        )

        strategy_outputs = {
            "energy_ood": ({0}, {0: {"energy_is_novel": True}}),
            "mixture_gaussian": ({0}, {0: {"mixture_gaussian_is_novel": True}}),
            "react_energy": ({0}, {0: {"react_energy_is_novel": True}}),
        }
        all_metrics = {
            0: {
                "energy_is_novel": True,
                "mixture_gaussian_is_novel": True,
                "react_energy_is_novel": True,
            }
        }

        novel_indices, novelty_scores = detector._combiner.combine(
            strategy_outputs, all_metrics
        )

        assert 0 in novel_indices
        assert novelty_scores[0] == pytest.approx(1.0)


class TestSignalCombinerEdgeCases:
    """Tests for edge cases in signal combining."""

    def test_unknown_combine_method_raises_error(self):
        """Unknown combine_method should raise ValueError."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence"],
                combine_method="unknown_method",
            )
        )

        with pytest.raises(ValueError, match="Unknown combine_method"):
            detector._combiner.combine({}, {})

    def test_combiner_preserves_strategy_output_order(self):
        """Combination should not depend on strategy ordering."""
        detector = NoveltyDetector(
            config=DetectionConfig(
                strategies=["confidence", "knn_distance", "clustering"],
                combine_method="union",
            )
        )

        outputs_abc = {
            "confidence": ({0, 1}, {}),
            "knn_distance": ({1, 2}, {}),
            "clustering": ({2, 0}, {}),
        }

        novel_indices, _ = detector._combiner.combine(outputs_abc, {})

        assert novel_indices == {0, 1, 2}

"""Tests for PatternBasedNoveltyStrategy."""

import pytest

from novelentitymatcher.novelty.config.strategies import PatternConfig
from novelentitymatcher.novelty.strategies.pattern import PatternStrategy
from novelentitymatcher.novelty.strategies.pattern_impl import (
    PatternScorer,
    score_batch_novelty,
)


class TestPatternScorer:
    """Test suite for PatternScorer."""

    @pytest.fixture
    def known_entities(self):
        return [
            "Apple Inc",
            "Microsoft Corporation",
            "Google LLC",
            "Amazon.com Inc",
            "Tesla Inc",
        ]

    @pytest.fixture
    def strategy(self, known_entities):
        return PatternScorer(known_entities)

    def test_initialization(self, strategy, known_entities):
        assert strategy.known_entities == known_entities
        assert strategy.patterns is not None
        assert "char_ngrams" in strategy.patterns
        assert "char_4grams" in strategy.patterns
        assert "capitalization" in strategy.patterns
        assert "length_range" in strategy.patterns

    def test_initialization_empty_entities(self):
        with pytest.raises(ValueError, match="known_entities cannot be empty"):
            PatternScorer([])

    def test_score_novelty_known_entity(self, strategy):
        # Known entity should have low novelty score
        score = strategy.score_novelty("Apple Inc")
        assert 0 <= score <= 1
        # Known entity should have relatively low novelty
        assert score < 0.7

    def test_score_novelty_similar_entity(self, strategy):
        # Similar entity should have moderate novelty
        score = strategy.score_novelty("Apple Corp")
        assert 0 <= score <= 1

    def test_score_novelty_novel_entity(self, strategy):
        # Novel entity with different patterns should have high novelty
        score = strategy.score_novelty("xyz123")
        assert 0 <= score <= 1
        # Should have relatively high novelty
        assert score > 0.3

    def test_score_novelty_empty_string(self, strategy):
        # Empty string should be maximally novel
        score = strategy.score_novelty("")
        assert score == 1.0

    def test_score_batch_novelty(self, strategy):
        entities = ["Apple Inc", "xyz123", "Microsoft Corporation", "novel_entity"]
        scores = score_batch_novelty(entities, strategy)

        assert len(scores) == len(entities)
        for score in scores:
            assert 0 <= score <= 1

    def test_char_ngrams_extraction(self, strategy):
        ngrams = strategy._get_char_ngrams(["test"], n=3)
        assert "tes" in ngrams
        assert "est" in ngrams
        assert len(ngrams) == 2

    def test_char_4grams_extraction(self, strategy):
        ngrams = strategy._get_char_ngrams(["test"], n=4)
        assert "test" in ngrams
        assert len(ngrams) == 1

    def test_has_numbers(self, strategy):
        entities_with_numbers = ["abc123", "xyz456", "test"]
        fraction = strategy._has_numbers(entities_with_numbers)
        assert fraction == 2 / 3

    def test_capitalization_patterns(self, strategy):
        entities = ["Title Case", "UPPERCASE", "lowercase", "MixedCase"]
        patterns = strategy._get_capitalization_patterns(entities)

        assert "title_case" in patterns
        assert "uppercase" in patterns
        assert "lowercase" in patterns
        assert "mixed" in patterns

    def test_prefix_distribution(self, strategy):
        entities = ["Apple", "Application", "Apply"]
        prefixes = strategy._get_prefix_suffix_distribution(entities, prefix=True, n=3)

        assert "App" in prefixes
        assert prefixes["App"] == 3

    def test_suffix_distribution(self, strategy):
        entities = ["testing", "running", "jumping"]
        suffixes = strategy._get_prefix_suffix_distribution(entities, prefix=False, n=3)

        assert "ing" in suffixes
        assert suffixes["ing"] == 3

    def test_length_range(self, strategy, known_entities):
        min_len, max_len = strategy.patterns["length_range"]

        # Check that range is correct
        assert min_len == min(len(e) for e in known_entities)
        assert max_len == max(len(e) for e in known_entities)

    def test_score_novelty_consistency(self, strategy):
        # Scoring the same entity twice should give the same result
        entity = "Test Entity"
        score1 = strategy.score_novelty(entity)
        score2 = strategy.score_novelty(entity)

        assert score1 == score2


class TestPatternStrategy:
    def test_initialize_uses_reference_labels(self, monkeypatch):
        captured = {}

        class FakeScorer:
            def __init__(self, known_entities):
                captured["known_entities"] = known_entities

        monkeypatch.setattr(
            "novelentitymatcher.novelty.strategies.pattern.PatternScorer",
            FakeScorer,
        )

        strategy = PatternStrategy()
        strategy.initialize([], ["A", "B"], PatternConfig(threshold=0.4))
        assert captured["known_entities"] == ["A", "B"]

    def test_detect_applies_threshold_and_metrics_schema(self):
        strategy = PatternStrategy()
        strategy._config = PatternConfig(threshold=0.5)

        class StubScorer:
            def score_novelty(self, text):
                return {"known": 0.2, "novel": 0.8}[text]

        strategy._pattern_scorer = StubScorer()
        flags, metrics = strategy._detect(
            ["known", "novel"],
            [],
            ["A", "B"],
            [],
        )

        assert flags == {1}
        assert metrics[0]["pattern_is_novel"] is False
        assert metrics[1]["pattern_is_novel"] is True
        assert metrics[1]["pattern_text"] == "novel"
        assert metrics[1]["pattern_novelty_score"] == 0.8

    def test_metadata_and_config_schema_contract(self):
        strategy = PatternStrategy()
        assert strategy.config_schema is PatternConfig
        assert strategy.strategy_id == "pattern"
        assert strategy.score_keys == ("pattern_novelty_score",)
        assert strategy.signal_info.flag_key == "pattern_is_novel"

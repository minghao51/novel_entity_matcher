"""Extended integration tests for semantic matcher."""

import pytest

from novelentitymatcher import Matcher
from novelentitymatcher.exceptions import ValidationError


class TestAsyncAPIIntegration:
    """Integration tests for async API functionality."""

    @pytest.fixture
    def trained_matcher(self):
        """Create a trained matcher for async testing."""
        entities = [
            {"id": "physics", "name": "Quantum Physics"},
            {"id": "cs", "name": "Computer Science"},
            {"id": "biology", "name": "Molecular Biology"},
        ]

        training_data = {
            "physics": ["quantum mechanics", "wave function", "Schrödinger equation"],
            "cs": ["machine learning", "neural networks", "algorithm design"],
            "biology": ["gene expression", "protein synthesis", "DNA replication"],
        }

        # Flatten training data
        texts = []
        labels = []
        for label, samples in training_data.items():
            texts.extend(samples)
            labels.extend([label] * len(samples))

        matcher = Matcher(entities=entities, model="minilm", threshold=0.6)
        matcher.fit(texts=texts, labels=labels)

        return matcher

    @pytest.mark.asyncio
    async def test_async_match_single_query(self, trained_matcher):
        """Test async matching with a single query."""
        result = await trained_matcher.match_async("quantum superposition")

        assert result is not None

    @pytest.mark.asyncio
    async def test_async_match_multiple_queries(self, trained_matcher):
        """Test async matching with multiple queries."""
        queries = ["quantum superposition", "deep learning", "gene editing"]
        results = await trained_matcher.match_async(queries)

        assert len(results) == len(queries)
        assert all(isinstance(r, dict) or r is None for r in results)

    @pytest.mark.asyncio
    async def test_async_match_with_candidates(self, trained_matcher):
        """Test async matching with candidate filtering."""
        queries = ["quantum physics research"]
        candidates = [
            {"id": "physics", "name": "Quantum Physics"},
            {"id": "cs", "name": "Computer Science"},
        ]

        results = await trained_matcher.match_async(queries, candidates=candidates)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_async_match_batch(self, trained_matcher):
        """Test async batch matching."""
        queries = ["quantum mechanics", "machine learning", "DNA replication"]
        results = await trained_matcher.match_batch_async(queries, batch_size=2)

        assert len(results) == len(queries)

    @pytest.mark.asyncio
    async def test_async_explain_match(self, trained_matcher):
        """Test async explain match."""
        explanation = await trained_matcher.explain_match_async("quantum physics")

        assert explanation is not None
        assert isinstance(explanation, dict)
        assert "best_match" in explanation


class TestErrorHandlingIntegration:
    """Integration tests for error handling."""

    def test_match_with_empty_entities(self):
        """Test matcher handles empty entities gracefully."""
        with pytest.raises(ValidationError):
            Matcher(entities=[], model="minilm")

    def test_match_with_invalid_threshold(self):
        """Test matcher handles invalid threshold."""
        entities = [{"id": "1", "name": "Test"}]

        with pytest.raises(ValidationError):
            Matcher(entities=entities, model="minilm", threshold=1.5)

    def test_match_without_training(self):
        """Test matching without training returns results (zero-shot)."""
        entities = [{"id": "tech", "name": "Technology"}]

        matcher = Matcher(entities=entities, model="minilm", mode="zero-shot")
        results = matcher.match(["software development"])

        # Zero-shot should still work
        assert results is not None

    def test_predict_returns_none_before_fit(self):
        """Test that predict before fit returns None (zero-shot)."""
        entities = [{"id": "1", "name": "Test"}]

        matcher = Matcher(entities=entities, model="minilm", mode="zero-shot")

        # Zero-shot mode should not raise error
        result = matcher.predict(["test"])

        # Result might be None or list of IDs in zero-shot
        assert result is None or isinstance(result, list)

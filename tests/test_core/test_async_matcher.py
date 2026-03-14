import pytest
import asyncio
from semanticmatcher.core.matcher import Matcher


class TestMatcherAsyncLifecycle:
    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
        ]

    @pytest.mark.asyncio
    async def test_async_context_manager(self, sample_entities):
        """Test that Matcher can be used as async context manager"""
        async with Matcher(entities=sample_entities) as matcher:
            assert matcher is not None
            assert matcher._async_executor is not None

        # Executor should be shut down after context exit
        # This is verified by not getting an error

    @pytest.mark.asyncio
    async def test_aclose_explicit(self, sample_entities):
        """Test explicit aclose() method"""
        matcher = Matcher(entities=sample_entities)
        await matcher.fit_async()
        await matcher.aclose()
        # Should not raise

    @pytest.mark.asyncio
    async def test_multiple_async_fits(self, sample_entities):
        """Test that async executor is reused across multiple async calls"""
        async with Matcher(entities=sample_entities) as matcher:
            executor_id_before = id(matcher._async_executor)
            await matcher.fit_async()
            executor_id_after = id(matcher._async_executor)
            assert executor_id_before == executor_id_after


class TestMatcherFitAsync:
    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
        ]

    @pytest.fixture
    def training_data(self):
        return [
            {"text": "Germany", "label": "DE"},
            {"text": "USA", "label": "US"},
        ]

    @pytest.mark.asyncio
    async def test_fit_async_zero_shot(self, sample_entities):
        """Test async fit in zero-shot mode"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            assert matcher._active_matcher is not None
            assert matcher._training_mode == "zero-shot"

    @pytest.mark.asyncio
    async def test_fit_async_with_training_data(self, sample_entities, training_data):
        """Test async fit with training data"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async(training_data, num_epochs=1)
            assert matcher._active_matcher is not None
            assert matcher._has_training_data

    @pytest.mark.asyncio
    async def test_fit_async_explicit_mode(self, sample_entities, training_data):
        """Test async fit with explicit mode"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async(training_data, mode="head-only", num_epochs=1)
            assert matcher._training_mode == "head-only"

    @pytest.mark.asyncio
    async def test_fit_async_returns_self(self, sample_entities):
        """Test that fit_async returns self for method chaining"""
        async with Matcher(entities=sample_entities) as matcher:
            result = await matcher.fit_async()
            assert result is matcher


class TestMatcherMatchAsync:
    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
        ]

    @pytest.mark.asyncio
    async def test_match_async_single(self, sample_entities):
        """Test async match with single query"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            result = await matcher.match_async("USA")
            assert result is not None
            assert result["id"] == "US"
            assert result["score"] > 0.7

    @pytest.mark.asyncio
    async def test_match_async_multiple(self, sample_entities):
        """Test async match with multiple queries"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            results = await matcher.match_async(["USA", "Germany"])
            assert len(results) == 2
            assert results[0]["id"] == "US"
            assert results[1]["id"] == "DE"

    @pytest.mark.asyncio
    async def test_match_async_top_k(self, sample_entities):
        """Test async match with top_k parameter"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            results = await matcher.match_async("United States", top_k=2)
            assert isinstance(results, list)
            assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_match_async_below_threshold(self, sample_entities):
        """Test async match returns None when below threshold"""
        async with Matcher(entities=sample_entities, threshold=0.99) as matcher:
            await matcher.fit_async()
            result = await matcher.match_async("TotallyNotACountry123")
            assert result is None


class TestMatcherMatchBatchAsync:
    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        ]

    @pytest.mark.asyncio
    async def test_match_batch_async_basic(self, sample_entities):
        """Test basic async batch matching"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            queries = ["USA", "Germany", "France"]
            results = await matcher.match_batch_async(queries)
            assert len(results) == 3
            assert results[0]["id"] == "US"
            assert results[1]["id"] == "DE"
            assert results[2]["id"] == "FR"

    @pytest.mark.asyncio
    async def test_match_batch_async_with_progress(self, sample_entities):
        """Test batch matching with progress callback"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()

            progress_updates = []

            async def progress_callback(completed, total):
                progress_updates.append((completed, total))

            queries = ["USA", "Germany"] * 5
            results = await matcher.match_batch_async(
                queries,
                batch_size=2,
                on_progress=progress_callback
            )

            assert len(results) == 10
            assert len(progress_updates) > 0
            # Verify progress was reported
            assert all(c <= t for c, t in progress_updates)

    @pytest.mark.asyncio
    async def test_match_batch_async_top_k(self, sample_entities):
        """Test batch matching with top_k"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            queries = ["USA", "Germany"]
            results = await matcher.match_batch_async(queries, top_k=2)
            assert len(results) == 2
            # Each result should be a list when top_k > 1
            assert isinstance(results[0], list)

    @pytest.mark.asyncio
    async def test_match_batch_async_threshold(self, sample_entities):
        """Test batch matching with threshold filtering"""
        async with Matcher(entities=sample_entities, threshold=0.99) as matcher:
            await matcher.fit_async()
            queries = ["USA", "TotallyNotACountry"]
            results = await matcher.match_batch_async(queries)
            assert len(results) == 2
            # First might not match due to high threshold
            # Second definitely won't match
            assert results[1] is None


class TestMatcherExplainAsync:
    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
        ]

    @pytest.mark.asyncio
    async def test_explain_match_async(self, sample_entities):
        """Test async explain_match"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            explanation = await matcher.explain_match_async("USA", top_k=3)

            assert explanation["query"] == "USA"
            assert "matched" in explanation
            assert "best_match" in explanation
            assert "top_k" in explanation
            assert explanation["threshold"] == 0.7

    @pytest.mark.asyncio
    async def test_diagnose_async(self, sample_entities):
        """Test async diagnose"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            diagnosis = await matcher.diagnose_async("USA")

            assert diagnosis["query"] == "USA"
            assert diagnosis["matcher_ready"] is True
            assert "matched" in diagnosis


class TestEntityMatcherAsync:
    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
        ]

    @pytest.fixture
    def training_data(self):
        return [
            {"text": "Germany", "label": "DE"},
            {"text": "USA", "label": "US"},
        ]

    @pytest.mark.asyncio
    async def test_entity_matcher_train_async(self, sample_entities, training_data):
        """Test EntityMatcher.train_async"""
        from semanticmatcher.core.matcher import EntityMatcher

        matcher = EntityMatcher(entities=sample_entities)
        await matcher.train_async(training_data, num_epochs=1)
        assert matcher.is_trained

    @pytest.mark.asyncio
    async def test_entity_matcher_match_async(self, sample_entities, training_data):
        """Test EntityMatcher.match_async"""
        from semanticmatcher.core.matcher import EntityMatcher

        matcher = EntityMatcher(entities=sample_entities)
        await matcher.train_async(training_data, num_epochs=1)
        result = await matcher.match_async("USA")
        assert result["id"] == "US"

    @pytest.mark.asyncio
    async def test_entity_matcher_predict_async(self, sample_entities, training_data):
        """Test EntityMatcher.predict_async"""
        from semanticmatcher.core.matcher import EntityMatcher

        matcher = EntityMatcher(entities=sample_entities)
        await matcher.train_async(training_data, num_epochs=1)
        result = await matcher.predict_async("USA")
        assert result == "US"


class TestEmbeddingMatcherAsync:
    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
        ]

    @pytest.mark.asyncio
    async def test_embedding_matcher_build_index_async(self, sample_entities):
        """Test EmbeddingMatcher.build_index_async"""
        from semanticmatcher.core.matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher(entities=sample_entities)
        await matcher.build_index_async()
        assert matcher.embeddings is not None

    @pytest.mark.asyncio
    async def test_embedding_matcher_match_async(self, sample_entities):
        """Test EmbeddingMatcher.match_async"""
        from semanticmatcher.core.matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher(entities=sample_entities)
        await matcher.build_index_async()
        result = await matcher.match_async("USA")
        assert result["id"] == "US"



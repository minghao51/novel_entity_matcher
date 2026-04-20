import pytest
import asyncio
import threading
import time
from types import MethodType
from novelentitymatcher.core.matcher import Matcher


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
    async def test_match_async_metadata_matches_sync_contract(self, sample_entities):
        """Async metadata contract should mirror the sync metadata shape."""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()
            async_result = await matcher.match_async(
                ["USA", "Germany"],
                return_metadata=True,
                top_k=2,
            )
            sync_result = matcher.match(
                ["USA", "Germany"],
                return_metadata=True,
                top_k=2,
            )

            assert len(async_result.records) == len(sync_result.records) == 2
            assert (
                len(async_result.candidate_results)
                == len(sync_result.candidate_results)
                == 2
            )
            assert async_result.metadata["top_k"] == sync_result.metadata["top_k"] == 2

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
                queries, batch_size=2, on_progress=progress_callback
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
        from novelentitymatcher.core.matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher(entities=sample_entities)
        await matcher.build_index_async()
        assert matcher.embeddings is not None

    @pytest.mark.asyncio
    async def test_embedding_matcher_match_async(self, sample_entities):
        """Test EmbeddingMatcher.match_async"""
        from novelentitymatcher.core.matcher import EmbeddingMatcher

        matcher = EmbeddingMatcher(entities=sample_entities)
        await matcher.build_index_async()
        result = await matcher.match_async("USA")
        assert result["id"] == "US"


class TestAsyncCancellation:
    @pytest.fixture
    def sample_entities(self):
        return [{"id": str(i), "name": f"Entity {i}"} for i in range(10)]

    @pytest.mark.asyncio
    async def test_match_batch_async_cancellation(self, sample_entities):
        """Test that batch matching can be cancelled"""
        from novelentitymatcher.core.matcher import Matcher

        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()

            # Create a large batch
            queries = [f"Entity {i}" for i in range(1000)]

            # Create a task that can be cancelled
            task = asyncio.create_task(
                matcher.match_batch_async(queries, batch_size=10)
            )

            # Cancel immediately
            task.cancel()

            # Should raise CancelledError
            with pytest.raises(asyncio.CancelledError):
                await task

    @pytest.mark.asyncio
    async def test_fit_async_cancellation(self, sample_entities):
        """Test that fit can be cancelled"""
        training_data = [{"text": f"Entity {i}", "label": str(i)} for i in range(10)]

        async with Matcher(entities=sample_entities) as matcher:
            # Create a task that can be cancelled
            task = asyncio.create_task(
                matcher.fit_async(training_data, mode="full", num_epochs=10)
            )

            # Wait a bit then cancel
            await asyncio.sleep(0.1)
            task.cancel()

            # Should raise CancelledError
            with pytest.raises(asyncio.CancelledError):
                await task


class TestAsyncConcurrency:
    @pytest.fixture
    def sample_entities(self):
        return [{"id": str(i), "name": f"Entity {i}"} for i in range(10)]

    @pytest.mark.asyncio
    async def test_concurrent_matchers(self, sample_entities):
        """Test multiple matchers running concurrently"""

        async def match_entity(entity_id):
            async with Matcher(entities=sample_entities) as matcher:
                await matcher.fit_async()
                return await matcher.match_async(f"Entity {entity_id}")

        # Run multiple matchers concurrently
        results = await asyncio.gather(
            match_entity(0),
            match_entity(1),
            match_entity(2),
        )

        assert len(results) == 3
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_queries_on_same_matcher(self, sample_entities):
        """Test multiple queries on the same matcher concurrently"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()

            async def match_query(query):
                return await matcher.match_async(query)

            # Run multiple queries concurrently
            results = await asyncio.gather(
                match_query("Entity 0"),
                match_query("Entity 1"),
                match_query("Entity 2"),
            )

            assert len(results) == 3
            assert all(r is not None for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_batch_operations(self, sample_entities):
        """Test concurrent batch operations"""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()

            async def match_batch(start_idx):
                queries = [f"Entity {i}" for i in range(start_idx, start_idx + 5)]
                return await matcher.match_batch_async(queries, batch_size=2)

            # Run multiple batch operations concurrently
            results = await asyncio.gather(
                match_batch(0),
                match_batch(5),
                match_batch(2),
            )

            assert len(results) == 3
            assert all(len(r) == 5 for r in results)

    @pytest.mark.asyncio
    async def test_concurrent_match_async_autofit_only_runs_once(self, sample_entities):
        """Concurrent async callers should not duplicate lazy fit work."""
        matcher = Matcher(entities=sample_entities)
        original_fit = matcher.fit
        fit_calls = 0

        def wrapped_fit(self, *args, **kwargs):
            nonlocal fit_calls
            fit_calls += 1
            return original_fit(*args, **kwargs)

        matcher.fit = MethodType(wrapped_fit, matcher)

        try:
            results = await asyncio.gather(
                matcher.match_async("Entity 0"),
                matcher.match_async("Entity 1"),
                matcher.match_async("Entity 2"),
            )
        finally:
            await matcher.aclose()

        assert len(results) == 3
        assert all(result is not None for result in results)
        assert fit_calls == 1

    @pytest.mark.asyncio
    async def test_concurrent_temporary_threshold_operations_restore_state(
        self, sample_entities
    ):
        """Temporary threshold overrides should not leak after concurrent calls."""
        async with Matcher(entities=sample_entities, threshold=0.95) as matcher:
            await matcher.fit_async()

            explanation_task = matcher.explain_match_async("Entity 0", top_k=3)
            batch_task = matcher.match_batch_async(
                ["Entity 0", "Entity 1", "missing"],
                threshold=0.0,
                batch_size=1,
            )

            explanation, batch_results = await asyncio.gather(
                explanation_task,
                batch_task,
            )

            assert explanation["threshold"] == 0.95
            assert matcher.threshold == 0.95
            assert len(batch_results) == 3

    @pytest.mark.asyncio
    async def test_explain_match_async_threshold_override_is_call_local(
        self, sample_entities
    ):
        """Concurrent explain calls must not lower threshold for plain matches."""
        async with Matcher(entities=sample_entities, threshold=0.95) as matcher:
            await matcher.fit_async()

            seen_thresholds = []
            gate = threading.Event()

            def instrumented_match(*args, **kwargs):
                threshold = kwargs.get(
                    "threshold_override", matcher.embedding_matcher.threshold
                )
                seen_thresholds.append(threshold)
                if len(seen_thresholds) == 1:
                    gate.wait(timeout=1.0)
                texts = kwargs.get("texts", args[0] if args else None)
                result = {"id": "probe", "score": threshold, "text": "probe"}
                if isinstance(texts, list):
                    return [result for _ in texts]
                return result

            matcher.embedding_matcher.match = instrumented_match

            explain_task = asyncio.create_task(
                matcher.explain_match_async("Entity 0", top_k=1)
            )
            while len(seen_thresholds) < 1:
                await asyncio.sleep(0.01)

            plain_task = asyncio.create_task(matcher.match_async("Entity 1", top_k=1))
            while len(seen_thresholds) < 2:
                await asyncio.sleep(0.01)

            gate.set()
            plain_result = await plain_task
            explanation = await explain_task

            assert seen_thresholds == [0.0, 0.95]
            assert plain_result["score"] == 0.95
            assert explanation["threshold"] == 0.95
            assert matcher.threshold == 0.95

    @pytest.mark.asyncio
    async def test_match_batch_async_threshold_override_is_call_local(
        self, sample_entities
    ):
        """Batch threshold overrides must not leak into concurrent plain matches."""
        async with Matcher(entities=sample_entities, threshold=0.92) as matcher:
            await matcher.fit_async()

            seen_thresholds = []
            gate = threading.Event()

            def instrumented_match(*args, **kwargs):
                threshold = kwargs.get(
                    "threshold_override", matcher.embedding_matcher.threshold
                )
                seen_thresholds.append(threshold)
                if len(seen_thresholds) == 1:
                    gate.wait(timeout=1.0)
                texts = kwargs.get("texts", args[0] if args else None)
                result = {"id": "probe", "score": threshold, "text": "probe"}
                if isinstance(texts, list):
                    return [result for _ in texts]
                return result

            matcher.embedding_matcher.match = instrumented_match

            batch_task = asyncio.create_task(
                matcher.match_batch_async(["Entity 0"], threshold=0.0, batch_size=1)
            )
            while len(seen_thresholds) < 1:
                await asyncio.sleep(0.01)

            plain_task = asyncio.create_task(matcher.match_async("Entity 1", top_k=1))
            while len(seen_thresholds) < 2:
                await asyncio.sleep(0.01)

            gate.set()
            plain_result = await plain_task
            batch_result = await batch_task

            assert seen_thresholds == [0.0, 0.92]
            assert plain_result["score"] == 0.92
            assert batch_result[0]["score"] == 0.0
            assert matcher.threshold == 0.92

    @pytest.mark.asyncio
    async def test_match_batch_async_cancellation_keeps_executor_usable(
        self, sample_entities
    ):
        """Cancelling one batch should not tear down the shared executor."""
        async with Matcher(entities=sample_entities) as matcher:
            await matcher.fit_async()

            original_match = matcher.embedding_matcher.match
            entered = threading.Event()
            release = threading.Event()

            def slow_match(*args, **kwargs):
                entered.set()
                release.wait(timeout=1.0)
                return original_match(*args, **kwargs)

            matcher.embedding_matcher.match = slow_match

            task = asyncio.create_task(
                matcher.match_batch_async(["Entity 0", "Entity 1"], batch_size=1)
            )
            while not entered.wait(timeout=0.05):
                await asyncio.sleep(0.01)

            start = time.perf_counter()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            cancelled_after = time.perf_counter() - start

            assert cancelled_after < 0.2
            assert matcher._async_executor is not None

            release.set()
            result = await matcher.match_async("Entity 0")
            assert result["id"] == "0"

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

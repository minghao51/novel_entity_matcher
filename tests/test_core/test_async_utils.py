import pytest
from semanticmatcher.core.async_utils import AsyncExecutor


class TestAsyncExecutor:
    def test_init_default_workers(self):
        executor = AsyncExecutor()
        assert executor._executor is not None
        executor.shutdown()

    def test_init_custom_workers(self):
        executor = AsyncExecutor(max_workers=4)
        assert executor._executor is not None
        executor.shutdown()

    @pytest.mark.asyncio
    async def test_run_in_thread(self):
        executor = AsyncExecutor()

        def sync_func(x):
            return x * 2

        result = await executor.run_in_thread(sync_func, 5)
        assert result == 10
        executor.shutdown()

    @pytest.mark.asyncio
    async def test_run_in_thread_batch(self):
        executor = AsyncExecutor()

        def sync_batch(items):
            return [x * 2 for x in items]

        items = [1, 2, 3, 4, 5]
        results = await executor.run_in_thread_batch(sync_batch, items, batch_size=2)
        assert results == [2, 4, 6, 8, 10]
        executor.shutdown()

    @pytest.mark.asyncio
    async def test_shutdown(self):
        executor = AsyncExecutor()
        executor.shutdown()
        # Should not raise
        executor.shutdown()

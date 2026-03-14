import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, Optional
import functools


class AsyncExecutor:
    """
    Manages async execution of sync operations.

    Runs CPU-bound or blocking sync operations in a thread pool,
    allowing async code to proceed without blocking the event loop.
    """

    def __init__(self, max_workers: Optional[int] = None):
        """
        Initialize the async executor.

        Args:
            max_workers: Maximum number of worker threads. Defaults to CPU_COUNT * 2,
                capped at 32 for I/O bound workloads.
        """
        if max_workers is None:
            # Default to CPU count * 2 for I/O bound work
            max_workers = min(32, (os.cpu_count() or 1) * 2)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    async def run_in_thread(self, func: Callable, *args, **kwargs) -> Any:
        """
        Run a sync function in a thread pool.

        Args:
            func: Synchronous function to execute
            *args: Positional arguments to pass to func
            **kwargs: Keyword arguments to pass to func

        Returns:
            The return value of func
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor,
            functools.partial(func, *args, **kwargs)
        )

    async def run_in_thread_batch(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 32
    ) -> List[Any]:
        """
        Run sync function on batches concurrently.

        Splits items into batches and runs func on each batch in parallel,
        then flattens the results.

        Args:
            func: Function that takes a list and returns a list
            items: Items to process in batches
            batch_size: Size of each batch

        Returns:
            Flattened list of results from all batches
        """
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        tasks = [self.run_in_thread(func, batch) for batch in batches]
        results = await asyncio.gather(*tasks)
        return [item for batch in results for item in batch]

    def shutdown(self):
        """Clean up resources by shutting down the thread pool. Idempotent."""
        if hasattr(self, '_executor') and self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

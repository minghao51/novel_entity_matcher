import asyncio
import inspect
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .matcher import Matcher

logger = logging.getLogger(__name__)


class BatchEngine:
    """Batch async matching operations for the Matcher facade."""

    def __init__(self, facade: "Matcher") -> None:
        self._facade = facade

    async def match_batch(
        self,
        queries: list[str],
        threshold: float | None = None,
        top_k: int = 1,
        batch_size: int = 32,
        on_progress: Callable[[int, int], None] | None = None,
        **kwargs,
    ) -> list[Any]:
        executor = self._facade._ensure_async_executor()
        return await self._match_impl(
            executor,
            queries,
            top_k,
            batch_size,
            on_progress,
            threshold_override=threshold,
            **kwargs,
        )

    async def _match_impl(
        self,
        executor: Any,
        queries: list[str],
        top_k: int,
        batch_size: int,
        on_progress: Callable[[int, int], None] | None,
        threshold_override: float | None = None,
        **kwargs,
    ) -> list[Any]:
        total = len(queries)
        results: list[Any] = []
        completed = 0

        for index in range(0, total, batch_size):
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelled():
                raise asyncio.CancelledError()

            batch = queries[index : index + batch_size]
            try:
                batch_results = await executor.run_in_thread(
                    self._facade.match,
                    batch,
                    top_k,
                    _threshold_override=threshold_override,
                    **kwargs,
                )

                if isinstance(batch_results, dict):
                    batch_results = [batch_results]
                elif not isinstance(batch_results, list):
                    batch_results = list(batch_results)

                results.extend(batch_results)
            except Exception:
                logger.exception("Batch %d failed", index // batch_size)
                results.extend([None] * len(batch))

            completed += len(batch)

            if on_progress:
                if inspect.iscoroutinefunction(on_progress):
                    await on_progress(completed, total)
                else:
                    on_progress(completed, total)

        if any(r is None for r in results):
            logger.warning(
                "Batch matching completed with %d failed results out of %d",
                sum(1 for r in results if r is None),
                total,
            )

        return results

"""Performance monitoring utilities for semantic matchers."""

import functools
import time
from contextlib import contextmanager
from typing import Callable, Dict, List, Optional

from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)

from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


def track_performance(func: Callable) -> Callable:
    """
    Decorator to track method performance metrics.

    Tracks:
        - Number of calls
        - Total time
        - Average time per call
        - Last call duration

    Usage:
        @track_performance
        def match(self, query, top_k=5):
            ...

        # Access metrics
        matcher._metrics  # {'calls': 10, 'total_time': 1.5, 'avg_time': 0.15}
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.time()
        result = func(self, *args, **kwargs)
        elapsed = time.time() - start

        if not hasattr(self, "_metrics"):
            self._metrics = {
                "calls": 0,
                "total_time": 0.0,
                "avg_time": 0.0,
                "last_time": 0.0,
            }

        self._metrics["calls"] += 1
        self._metrics["total_time"] += elapsed
        self._metrics["avg_time"] = self._metrics["total_time"] / self._metrics["calls"]
        self._metrics["last_time"] = elapsed

        return result

    return wrapper


class PerformanceMonitor:
    """
    Simple performance tracking for matchers and other components.

    Provides detailed metrics for different operations.

    Example:
        monitor = PerformanceMonitor()

        with monitor.track("match_operation"):
            result = matcher.match(query)

        logger.info(monitor.summary())
    """

    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}

    def record(self, operation: str, duration: float):
        """Record a timing for an operation."""
        if operation not in self.metrics:
            self.metrics[operation] = []
        self.metrics[operation].append(duration)

    @contextmanager
    def track(self, operation: str):
        """Context manager for tracking operation timing."""
        start = time.time()
        try:
            yield
        finally:
            self.record(operation, time.time() - start)

    @staticmethod
    def _stats(timings: List[float]) -> Dict[str, float]:
        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Return summary statistics for all tracked operations.

        Returns:
            Dict mapping operation names to statistics:
                - count: Number of recordings
                - total: Total time
                - mean: Average time
                - min: Minimum time
                - max: Maximum time
        """
        return {
            op: self._stats(timings) for op, timings in self.metrics.items() if timings
        }

    def reset(self):
        """Clear all recorded metrics."""
        self.metrics.clear()

    def get_operation_metrics(self, operation: str) -> Optional[Dict[str, float]]:
        """Get metrics for a specific operation."""
        timings = self.metrics.get(operation, [])
        return self._stats(timings) if timings else None

    def to_dict(self) -> Dict[str, List[float]]:
        """Return raw metrics as dictionary."""
        return dict(self.metrics)

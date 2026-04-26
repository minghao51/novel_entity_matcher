"""Performance monitoring utilities for semantic matchers."""

import functools
import json
import time
from collections.abc import Callable
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


def track_performance(func: Callable[..., Any]) -> Callable[..., Any]:
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
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
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

    def __init__(self) -> None:
        self.metrics: dict[str, list[float]] = {}

    def record(self, operation: str, duration: float) -> None:
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
    def _stats(timings: list[float]) -> dict[str, float]:
        return {
            "count": len(timings),
            "total": sum(timings),
            "mean": sum(timings) / len(timings),
            "min": min(timings),
            "max": max(timings),
        }

    def summary(self) -> dict[str, dict[str, float]]:
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

    def reset(self) -> None:
        """Clear all recorded metrics."""
        self.metrics.clear()

    def get_operation_metrics(self, operation: str) -> dict[str, float] | None:
        """Get metrics for a specific operation."""
        timings = self.metrics.get(operation, [])
        return self._stats(timings) if timings else None

    def to_dict(self) -> dict[str, list[float]]:
        """Return raw metrics as dictionary."""
        return {k: list(v) for k, v in self.metrics.items()}

    def _write_csv(self, output_path: Path) -> None:
        """Write metrics to CSV file."""
        import csv

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["operation", "duration_ms", "timestamp"])
            for operation, timings in self.metrics.items():
                for timing in timings:
                    writer.writerow(
                        [operation, timing * 1000, datetime.now().isoformat()]
                    )

    def export_json(self, filepath: str) -> Path:
        """Export all metrics to JSON file.

        Args:
            filepath: Path to output JSON file

        Returns:
            Path to saved file
        """
        output_path = Path(filepath)
        data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.summary(),
            "raw_timings": self.to_dict(),
        }
        output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        logger.info("Exported metrics to %s", output_path)
        return output_path

    def export_csv(self, filepath: str) -> Path:
        """Export metrics to CSV file.

        Args:
            filepath: Path to output CSV file

        Returns:
            Path to saved file
        """
        output_path = Path(filepath)
        self._write_csv(output_path)
        logger.info("Exported metrics to %s", output_path)
        return output_path

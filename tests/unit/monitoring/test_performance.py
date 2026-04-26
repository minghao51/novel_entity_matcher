"""Unit tests for monitoring.performance module."""

import json
import time

import pytest

from novelentitymatcher.monitoring.performance import (
    PerformanceMonitor,
    track_performance,
)


class TestTrackPerformance:
    def test_decorator_tracks_metrics(self):
        class DummyService:
            @track_performance
            def do_work(self):
                time.sleep(0.01)
                return 42

        service = DummyService()
        result = service.do_work()
        assert result == 42
        assert hasattr(service, "_metrics")
        assert service._metrics["calls"] == 1
        assert service._metrics["total_time"] > 0
        assert service._metrics["avg_time"] > 0
        assert service._metrics["last_time"] > 0

    def test_decorator_multiple_calls(self):
        class DummyService:
            @track_performance
            def do_work(self):
                return 1

        service = DummyService()
        service.do_work()
        service.do_work()
        service.do_work()
        assert service._metrics["calls"] == 3
        assert service._metrics["avg_time"] == pytest.approx(
            service._metrics["total_time"] / 3
        )

    def test_decorator_preserves_function_name(self):
        class DummyService:
            @track_performance
            def named_method(self):
                pass

        assert DummyService.named_method.__name__ == "named_method"

    def test_decorator_without_self_raises(self):
        @track_performance
        def standalone():
            pass

        with pytest.raises(TypeError):
            standalone()


class TestPerformanceMonitor:
    def test_init_creates_empty_metrics(self):
        monitor = PerformanceMonitor()
        assert monitor.metrics == {}

    def test_record_adds_timing(self):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        assert "op1" in monitor.metrics
        assert monitor.metrics["op1"] == [0.1]

    def test_record_multiple_timings(self):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        monitor.record("op1", 0.2)
        assert monitor.metrics["op1"] == [0.1, 0.2]

    def test_track_context_manager(self):
        monitor = PerformanceMonitor()
        with monitor.track("slow_op"):
            time.sleep(0.01)
        assert "slow_op" in monitor.metrics
        assert len(monitor.metrics["slow_op"]) == 1
        assert monitor.metrics["slow_op"][0] > 0

    def test_track_context_manager_exception_still_records(self):
        monitor = PerformanceMonitor()
        with pytest.raises(ValueError):
            with monitor.track("failing_op"):
                time.sleep(0.01)
                raise ValueError("boom")
        assert "failing_op" in monitor.metrics
        assert len(monitor.metrics["failing_op"]) == 1

    def test_summary_empty(self):
        monitor = PerformanceMonitor()
        assert monitor.summary() == {}

    def test_summary_single_operation(self):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        monitor.record("op1", 0.3)
        summary = monitor.summary()
        assert "op1" in summary
        stats = summary["op1"]
        assert stats["count"] == 2
        assert stats["total"] == pytest.approx(0.4)
        assert stats["mean"] == pytest.approx(0.2)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.3)

    def test_summary_multiple_operations(self):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        monitor.record("op2", 0.5)
        summary = monitor.summary()
        assert set(summary.keys()) == {"op1", "op2"}

    def test_summary_filters_empty_timings(self):
        monitor = PerformanceMonitor()
        monitor.metrics["empty_op"] = []
        assert "empty_op" not in monitor.summary()

    def test_reset_clears_metrics(self):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        monitor.reset()
        assert monitor.metrics == {}

    def test_get_operation_metrics_existing(self):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        stats = monitor.get_operation_metrics("op1")
        assert stats is not None
        assert stats["count"] == 1

    def test_get_operation_metrics_missing(self):
        monitor = PerformanceMonitor()
        assert monitor.get_operation_metrics("missing") is None

    def test_get_operation_metrics_empty_timings(self):
        monitor = PerformanceMonitor()
        monitor.metrics["op"] = []
        assert monitor.get_operation_metrics("op") is None

    def test_to_dict_returns_copy(self):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        d = monitor.to_dict()
        assert d == {"op1": [0.1]}
        d["op1"].append(0.2)
        assert monitor.metrics["op1"] == [0.1]

    def test_export_json(self, tmp_path):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        filepath = tmp_path / "metrics.json"
        result_path = monitor.export_json(str(filepath))
        assert result_path == filepath
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert "timestamp" in data
        assert "metrics" in data
        assert data["metrics"]["op1"]["count"] == 1
        assert "raw_timings" in data

    def test_export_csv(self, tmp_path):
        monitor = PerformanceMonitor()
        monitor.record("op1", 0.1)
        monitor.record("op1", 0.2)
        filepath = tmp_path / "metrics.csv"
        result_path = monitor.export_csv(str(filepath))
        assert result_path == filepath
        assert filepath.exists()
        lines = filepath.read_text().strip().split("\n")
        assert lines[0] == "operation,duration_ms,timestamp"
        assert len(lines) == 3  # header + 2 records

    def test_static_stats_method(self):
        stats = PerformanceMonitor._stats([0.1, 0.2, 0.3])
        assert stats["count"] == 3
        assert stats["total"] == pytest.approx(0.6)
        assert stats["mean"] == pytest.approx(0.2)
        assert stats["min"] == pytest.approx(0.1)
        assert stats["max"] == pytest.approx(0.3)

    def test_static_stats_single_value(self):
        stats = PerformanceMonitor._stats([0.5])
        assert stats["count"] == 1
        assert stats["mean"] == pytest.approx(0.5)

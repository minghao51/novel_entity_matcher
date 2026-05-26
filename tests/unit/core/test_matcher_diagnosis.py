from unittest.mock import AsyncMock, MagicMock

import pytest

from novelentitymatcher.core.matcher_diagnosis import DiagnosisEngine
from novelentitymatcher.exceptions import TrainingError


def _make_facade(**overrides):
    facade = MagicMock()
    facade.threshold = 0.5
    facade._training_mode = "zero-shot"
    facade._active_matcher = MagicMock()
    facade.normalize = False
    for k, v in overrides.items():
        setattr(facade, k, v)
    return facade


class TestBuildExplanation:
    def test_returns_matched_when_score_above_threshold(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        results = [{"id": "a", "score": 0.9}]
        explanation = engine.build_explanation("query", results, None)
        assert explanation["matched"] is True
        assert explanation["best_match"] == {"id": "a", "score": 0.9}

    def test_returns_unmatched_when_score_below_threshold(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        results = [{"id": "a", "score": 0.3}]
        explanation = engine.build_explanation("query", results, None)
        assert explanation["matched"] is False
        assert explanation["best_match"]["score"] == 0.3

    def test_handles_none_results(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        explanation = engine.build_explanation("query", None, None)
        assert explanation["matched"] is False
        assert explanation["best_match"] is None
        assert explanation["top_k"] == []

    def test_handles_empty_list_results(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        explanation = engine.build_explanation("query", [], None)
        assert explanation["matched"] is False
        assert explanation["top_k"] == []

    def test_single_dict_result_wrapped_in_list(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        result = {"id": "a", "score": 0.8}
        explanation = engine.build_explanation("query", result, None)
        assert explanation["top_k"] == [result]

    def test_includes_query_and_normalized(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        explanation = engine.build_explanation(
            "Hello", [{"id": "a", "score": 0.9}], "hello"
        )
        assert explanation["query"] == "Hello"
        assert explanation["query_normalized"] == "hello"
        assert explanation["threshold"] == 0.5
        assert explanation["mode"] == "zero-shot"


class TestExplain:
    def test_raises_when_matcher_not_ready(self):
        facade = _make_facade(_active_matcher=None)
        engine = DiagnosisEngine(facade)
        with pytest.raises(TrainingError, match="not ready"):
            engine.explain("query")

    def test_returns_explanation_on_success(self):
        facade = _make_facade()
        facade.match.return_value = [{"id": "a", "score": 0.9}]
        engine = DiagnosisEngine(facade)
        result = engine.explain("query")
        assert result["matched"] is True
        assert result["query"] == "query"

    async def test_explain_async_raises_when_not_ready(self):
        facade = _make_facade(_active_matcher=None)
        engine = DiagnosisEngine(facade)
        with pytest.raises(TrainingError, match="not ready"):
            await engine.explain_async("query")

    async def test_explain_async_returns_explanation(self):
        facade = _make_facade()
        facade.match_async = AsyncMock(return_value=[{"id": "a", "score": 0.9}])
        executor = MagicMock()
        facade._ensure_async_executor.return_value = executor
        engine = DiagnosisEngine(facade)
        result = await engine.explain_async("query")
        assert result["matched"] is True


class TestDiagnose:
    def test_returns_issue_when_matcher_not_ready(self):
        facade = _make_facade(_active_matcher=None)
        engine = DiagnosisEngine(facade)
        result = engine.diagnose("query")
        assert result["issue"] == "Matcher not ready"
        assert "suggestion" in result

    def test_returns_diagnosis_with_explanation(self):
        facade = _make_facade()
        facade.match.return_value = [{"id": "a", "score": 0.9}]
        engine = DiagnosisEngine(facade)
        result = engine.diagnose("query")
        assert result["matched"] is True
        assert result["matcher_ready"] is True

    def test_diagnosis_header_structure(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        header = engine._build_diagnosis_header("test")
        assert header["query"] == "test"
        assert header["matcher_ready"] is True
        assert header["active_matcher"] == "MagicMock"

    def test_diagnosis_header_no_active_matcher(self):
        facade = _make_facade(_active_matcher=None)
        engine = DiagnosisEngine(facade)
        header = engine._build_diagnosis_header("test")
        assert header["matcher_ready"] is False
        assert header["active_matcher"] is None

    async def test_diagnose_async_returns_issue_when_not_ready(self):
        facade = _make_facade(_active_matcher=None)
        engine = DiagnosisEngine(facade)
        result = await engine.diagnose_async("query")
        assert result["issue"] == "Matcher not ready"

    async def test_diagnose_async_returns_diagnosis(self):
        facade = _make_facade()
        facade.match_async = AsyncMock(return_value=[{"id": "a", "score": 0.9}])
        executor = MagicMock()
        facade._ensure_async_executor.return_value = executor
        engine = DiagnosisEngine(facade)
        result = await engine.diagnose_async("query")
        assert result["matched"] is True


class TestFailureHints:
    def test_hint_score_below_threshold(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        diagnosis = {}
        explanation = {
            "matched": False,
            "best_match": {"score": 0.3},
            "threshold": 0.5,
        }
        engine._add_failure_hints(diagnosis, explanation)
        assert "below threshold" in diagnosis["issue"]
        assert "suggestion" in diagnosis

    def test_hint_no_candidates(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        diagnosis = {}
        explanation = {
            "matched": False,
            "best_match": None,
            "threshold": 0.5,
        }
        engine._add_failure_hints(diagnosis, explanation)
        assert diagnosis["issue"] == "No candidates found"
        assert "suggestion" in diagnosis

    def test_no_hints_when_matched(self):
        facade = _make_facade()
        engine = DiagnosisEngine(facade)
        diagnosis = {}
        explanation = {
            "matched": True,
            "best_match": {"score": 0.9},
            "threshold": 0.5,
        }
        engine._add_failure_hints(diagnosis, explanation)
        assert "issue" not in diagnosis

from unittest.mock import AsyncMock, MagicMock

from novelentitymatcher.core.matcher_modes import (
    HybridMode,
    TrainingMode,
    ZeroShotMode,
)


def _make_mock_matcher():
    matcher = MagicMock()
    matcher.threshold = 0.5
    matcher._training_mode = "zero-shot"
    matcher._requested_model = "test-model"
    matcher._training_model_name = "train-model"
    matcher._has_training_data = False
    matcher.logger = MagicMock()
    matcher.embedding_matcher = MagicMock()
    matcher.entity_matcher = MagicMock()
    matcher.bert_matcher = MagicMock()
    matcher.hybrid_matcher = MagicMock()
    return matcher


class TestZeroShotMode:
    def test_fit_builds_index(self):
        matcher = _make_mock_matcher()
        mode = ZeroShotMode(matcher)
        mode.fit()
        matcher.embedding_matcher.build_index.assert_called_once()

    def test_match_delegates_to_embedding_matcher(self):
        matcher = _make_mock_matcher()
        matcher.embedding_matcher.match.return_value = [{"id": "a", "score": 0.9}]
        mode = ZeroShotMode(matcher)
        result = mode.match("hello", top_k=5, threshold_override=None)
        matcher.embedding_matcher.match.assert_called_once()
        assert result == [{"id": "a", "score": 0.9}]

    def test_match_uses_override_threshold(self):
        matcher = _make_mock_matcher()
        matcher.embedding_matcher.match.return_value = []
        mode = ZeroShotMode(matcher)
        mode.match("hello", top_k=5, threshold_override=0.8)
        call_kwargs = matcher.embedding_matcher.match.call_args
        assert call_kwargs.kwargs["threshold_override"] == 0.8

    async def test_match_async_delegates(self):
        matcher = _make_mock_matcher()
        matcher.embedding_matcher.match_async = AsyncMock(
            return_value=[{"id": "a", "score": 0.9}]
        )
        mode = ZeroShotMode(matcher)
        result = await mode.match_async("hello", top_k=5, threshold_override=None)
        matcher.embedding_matcher.match_async.assert_called_once()
        assert result == [{"id": "a", "score": 0.9}]


class TestTrainingMode:
    def test_init_stores_classifier_type(self):
        matcher = _make_mock_matcher()
        mode = TrainingMode(matcher, "setfit")
        assert mode._classifier_type == "setfit"

    def test_fit_trains_entity_matcher_for_setfit(self):
        matcher = _make_mock_matcher()
        mode = TrainingMode(matcher, "setfit")
        training_data = [{"text": "a", "label": "x"}]
        mode.fit(training_data=training_data)
        matcher.entity_matcher.train.assert_called_once()
        assert matcher._has_training_data is True

    def test_fit_trains_bert_matcher_for_bert(self):
        matcher = _make_mock_matcher()
        mode = TrainingMode(matcher, "bert")
        training_data = [{"text": "a", "label": "x"}]
        mode.fit(training_data=training_data)
        matcher.bert_matcher.train.assert_called_once()

    def test_match_delegates_to_resolved_matcher(self):
        matcher = _make_mock_matcher()
        matcher.entity_matcher.match.return_value = [{"id": "b", "score": 0.7}]
        mode = TrainingMode(matcher, "setfit")
        result = mode.match("query", top_k=3, threshold_override=None)
        matcher.entity_matcher.match.assert_called_once()
        assert result == [{"id": "b", "score": 0.7}]

    async def test_match_async_delegates(self):
        matcher = _make_mock_matcher()
        matcher.entity_matcher.match_async = AsyncMock(
            return_value=[{"id": "b", "score": 0.7}]
        )
        mode = TrainingMode(matcher, "setfit")
        await mode.match_async("query", top_k=3, threshold_override=None)
        matcher.entity_matcher.match_async.assert_called_once()


class TestHybridMode:
    def test_fit_returns_hybrid_matcher(self):
        matcher = _make_mock_matcher()
        mode = HybridMode(matcher)
        result = mode.fit()
        assert result is matcher.hybrid_matcher
        assert matcher._has_training_data is False

    def test_match_single_text(self):
        matcher = _make_mock_matcher()
        matcher.hybrid_matcher.match.return_value = [
            {"id": "c", "score": 0.95},
            {"id": "d", "score": 0.3},
        ]
        mode = HybridMode(matcher)
        result = mode.match("hello", top_k=1, threshold_override=0.5)
        matcher.hybrid_matcher.match.assert_called_once()
        assert result == {"id": "c", "score": 0.95}

    def test_match_single_text_no_results(self):
        matcher = _make_mock_matcher()
        matcher.hybrid_matcher.match.return_value = []
        mode = HybridMode(matcher)
        result = mode.match("hello", top_k=1, threshold_override=0.5)
        assert result is None

    def test_match_bulk_texts(self):
        matcher = _make_mock_matcher()
        matcher.hybrid_matcher.match_bulk.return_value = [
            [{"id": "c", "score": 0.95}],
            [{"id": "e", "score": 0.85}],
        ]
        mode = HybridMode(matcher)
        result = mode.match(["hello", "world"], top_k=5, threshold_override=0.5)
        matcher.hybrid_matcher.match_bulk.assert_called_once()
        assert len(result) == 2

    async def test_match_async_single_text(self):
        matcher = _make_mock_matcher()
        executor = MagicMock()
        executor.run_in_thread = AsyncMock(return_value=[{"id": "c", "score": 0.95}])
        matcher._ensure_async_executor.return_value = executor
        mode = HybridMode(matcher)
        result = await mode.match_async("hello", top_k=1, threshold_override=0.5)
        assert result == {"id": "c", "score": 0.95}

    async def test_match_async_bulk(self):
        matcher = _make_mock_matcher()
        executor = MagicMock()
        executor.run_in_thread = AsyncMock(return_value=[[{"id": "c", "score": 0.95}]])
        matcher._ensure_async_executor.return_value = executor
        mode = HybridMode(matcher)
        result = await mode.match_async(["hello"], top_k=5, threshold_override=0.5)
        assert len(result) == 1

import warnings

import numpy as np
import pytest

from novelentitymatcher.core.embedding_matcher import EmbeddingMatcher
from novelentitymatcher.core.matcher import Matcher, _EntityMatcher
from novelentitymatcher.exceptions import ModeError, TrainingError


class TestEmbeddingMatcher:
    """Tests for EmbeddingMatcher - similarity-based matching."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
            {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        ]

    def test_embedding_matcher_init(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        assert matcher.entities == sample_entities

    def test_embedding_matcher_default_model(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        assert matcher.model_name == "sentence-transformers/paraphrase-mpnet-base-v2"

    def test_embedding_matcher_custom_model(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities, model_name="BAAI/bge-m3")
        assert matcher.model_name == "BAAI/bge-m3"

    def test_embedding_matcher_resolves_dynamic_alias_before_loading(
        self, sample_entities, monkeypatch
    ):
        loaded_models = []

        class FakeModel:
            def __init__(self, model_name):
                loaded_models.append(model_name)

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts, batch_size=None):
                if isinstance(texts, str):
                    texts = [texts]
                return np.ones((len(texts), 2), dtype=float)

        monkeypatch.setattr(
            "novelentitymatcher.core.matcher.SentenceTransformer", FakeModel
        )

        matcher = EmbeddingMatcher(
            entities=sample_entities,
            model_name="mpnet",
            normalize=False,
            threshold=0.0,
        )
        matcher.build_index()

        assert loaded_models == ["sentence-transformers/all-mpnet-base-v2"]

    def test_embedding_matcher_build_index(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        assert matcher.embeddings is not None
        assert len(matcher.embeddings) > 0

    def test_embedding_matcher_match(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        result = matcher.match("Deutschland")
        assert result is not None
        assert result["id"] == "DE"
        assert "score" in result

    def test_embedding_matcher_match_below_threshold(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities, threshold=0.99)
        matcher.build_index()
        result = matcher.match("xyzunknown123")
        assert result is None

    def test_embedding_matcher_match_multiple(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        results = matcher.match(["Deutschland", "America"])
        assert len(results) == 2
        assert results[0]["id"] == "DE"
        assert results[1]["id"] == "US"

    def test_embedding_matcher_with_aliases(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        result = matcher.match("Deutchland")
        assert result["id"] == "DE"

    def test_embedding_matcher_top_k_deduplicates_alias_hits(
        self, sample_entities, monkeypatch
    ):
        vectors = {
            "Germany": [1.0, 0.0],
            "Deutschland": [1.0, 0.0],
            "Deutchland": [1.0, 0.0],
            "France": [0.0, 1.0],
            "Frankreich": [0.0, 1.0],
            "United States": [0.7, 0.7],
            "USA": [0.7, 0.7],
            "America": [0.7, 0.7],
        }

        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                encoded = [vectors.get(text, [1.0, 0.0]) for text in texts]
                return np.array(encoded, dtype=float)

        monkeypatch.setattr(
            "novelentitymatcher.core.matcher.SentenceTransformer", FakeModel
        )

        matcher = EmbeddingMatcher(
            entities=sample_entities, normalize=False, threshold=0.0
        )
        matcher.build_index()
        results = matcher.match("Germany", top_k=2)

        assert [r["id"] for r in results] == ["DE", "US"]

    def test_embedding_matcher_empty_candidates_returns_empty(
        self, sample_entities, monkeypatch
    ):
        class FakeModel:
            def __init__(self, *_args, **_kwargs):
                pass

            def get_sentence_embedding_dimension(self):
                return 2

            def encode(self, texts):
                if isinstance(texts, str):
                    texts = [texts]
                return np.ones((len(texts), 2), dtype=float)

        monkeypatch.setattr(
            "novelentitymatcher.core.matcher.SentenceTransformer", FakeModel
        )

        matcher = EmbeddingMatcher(
            entities=sample_entities, normalize=False, threshold=0.0
        )
        matcher.build_index()

        assert matcher.match("Germany", candidates=[]) is None
        assert matcher.match("Germany", candidates=[], top_k=2) == []


class TestUnifiedMatcher:
    """Tests for the unified Matcher class with auto-selection."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
            {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        ]

    @pytest.fixture
    def training_data_small(self):
        """Small training set (< 3 examples per entity) for head-only mode."""
        return [
            {"text": "Germany", "label": "DE"},
            {"text": "USA", "label": "US"},
            {"text": "France", "label": "FR"},
        ]

    @pytest.fixture
    def training_data_full(self):
        """Full training set (≥ 3 examples per entity) for full training mode."""
        return [
            {"text": "Germany", "label": "DE"},
            {"text": "Deutschland", "label": "DE"},
            {"text": "Deutchland", "label": "DE"},
            {"text": "France", "label": "FR"},
            {"text": "Frankreich", "label": "FR"},
            {"text": "USA", "label": "US"},
            {"text": "America", "label": "US"},
            {"text": "United States", "label": "US"},
        ]

    @staticmethod
    def _build_trained_matcher(sample_entities, threshold=0.4):
        class FakeClassifier:
            labels = ["DE", "FR", "US"]

            def predict_proba(self, text):
                assert text == "deutschland"
                return np.array([0.82, 0.75, 0.41], dtype=float)

        from novelentitymatcher.core.matcher import _EntityMatcher

        entity_matcher = _EntityMatcher(
            entities=sample_entities,
            threshold=threshold,
            normalize=True,
        )
        entity_matcher.classifier = FakeClassifier()
        entity_matcher.is_trained = True

        matcher = Matcher(entities=sample_entities, mode="full", threshold=threshold)
        matcher._entity_matcher = entity_matcher
        matcher._active_matcher = entity_matcher
        matcher._training_mode = "full"
        return matcher

    def test_matcher_init(self, sample_entities):
        """Test basic initialization."""
        matcher = Matcher(entities=sample_entities)
        assert matcher.entities == sample_entities
        assert matcher._training_mode == "auto"

    def test_matcher_invalid_mode_raises(self, sample_entities):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="Invalid mode"):
            Matcher(entities=sample_entities, mode="invalid_mode")

    def test_matcher_zero_shot_mode(self, sample_entities):
        """Test zero-shot mode is selected when mode is explicitly set."""
        matcher = Matcher(entities=sample_entities, mode="zero-shot")
        assert matcher._training_mode == "zero-shot"

    def test_matcher_head_only_mode(self, sample_entities):
        """Test head-only mode is accepted."""
        matcher = Matcher(entities=sample_entities, mode="head-only")
        assert matcher._training_mode == "head-only"

    def test_matcher_full_mode(self, sample_entities):
        """Test full training mode is accepted."""
        matcher = Matcher(entities=sample_entities, mode="full")
        assert matcher._training_mode == "full"

    def test_matcher_component_injection_still_works(self, sample_entities):
        matcher = Matcher(entities=sample_entities, mode="full")
        injected = object()

        matcher._entity_matcher = injected

        assert matcher.entity_matcher is injected

    def test_matcher_set_threshold_updates_existing_components(self, sample_entities):
        matcher = Matcher(entities=sample_entities, mode="full", threshold=0.6)
        entity_matcher = matcher.entity_matcher

        matcher.set_threshold(0.25)

        assert matcher.threshold == 0.25
        assert entity_matcher.threshold == 0.25

    def test_matcher_hybrid_mode(self, sample_entities):
        """Test hybrid mode is accepted."""
        matcher = Matcher(entities=sample_entities, mode="hybrid")
        assert matcher._training_mode == "hybrid"

    def test_matcher_auto_detect_zero_shot(self, sample_entities):
        """Test auto-detection selects zero-shot without training data."""
        matcher = Matcher(entities=sample_entities)
        assert matcher._detect_training_mode(None) == "zero-shot"

    def test_matcher_auto_detect_head_only(self, sample_entities, training_data_small):
        """Test auto-detection selects head-only with minimal training data."""
        matcher = Matcher(entities=sample_entities)
        assert matcher._detect_training_mode(training_data_small) == "head-only"

    def test_matcher_auto_detect_full(self, sample_entities, training_data_full):
        """Test auto-detection selects full training with sufficient data."""
        matcher = Matcher(entities=sample_entities)
        assert matcher._detect_training_mode(training_data_full) == "full"

    def test_matcher_fit_zero_shot(self, sample_entities):
        """Test fit() with zero-shot mode."""
        matcher = Matcher(entities=sample_entities, mode="zero-shot")
        matcher.fit()
        assert matcher._active_matcher is not None
        assert matcher._training_mode == "zero-shot"

    def test_matcher_fit_auto_with_small_data(
        self, sample_entities, training_data_small
    ):
        """Test fit() with auto-detection and small training set."""
        matcher = Matcher(entities=sample_entities)
        matcher.fit(training_data_small)
        assert matcher._has_training_data
        assert matcher._active_matcher is not None
        # Should auto-detect to head-only or full (currently defaults to training)

    def test_matcher_fit_with_full_training(self, sample_entities, training_data_full):
        """Test fit() with full training data."""
        matcher = Matcher(entities=sample_entities)
        matcher.fit(training_data_full, num_epochs=1)
        assert matcher._has_training_data
        assert matcher._active_matcher is not None

    def test_matcher_fit_mode_override_to_zero_shot(
        self, sample_entities, training_data_full
    ):
        """Test fit() with mode override to zero-shot."""
        matcher = Matcher(entities=sample_entities)
        matcher.fit(training_data_full, mode="zero-shot")
        assert matcher._training_mode == "zero-shot"

    def test_matcher_fit_mode_override_to_full(
        self, sample_entities, training_data_small
    ):
        """Test fit() with mode override to full training."""
        matcher = Matcher(entities=sample_entities)
        matcher.fit(training_data_small, mode="full", num_epochs=1)
        assert matcher._training_mode == "full"

    def test_matcher_training_default_uses_training_safe_backbone(
        self, sample_entities, training_data_small, monkeypatch
    ):
        trained_models = []

        def fake_train(self, training_data, **kwargs):
            trained_models.append(self.model_name)
            self.is_trained = True

        monkeypatch.setattr(_EntityMatcher, "train", fake_train)

        matcher = Matcher(entities=sample_entities)
        matcher.fit(training_data_small, mode="full", show_progress=False)

        assert trained_models == ["sentence-transformers/all-mpnet-base-v2"]

    def test_matcher_explicit_static_model_falls_back_for_training(
        self, sample_entities, training_data_small, monkeypatch
    ):
        trained_models = []

        def fake_train(self, training_data, **kwargs):
            trained_models.append(self.model_name)
            self.is_trained = True

        monkeypatch.setattr(_EntityMatcher, "train", fake_train)

        matcher = Matcher(entities=sample_entities, model="potion-8m")
        matcher.fit(training_data_small, mode="head-only", show_progress=False)

        assert trained_models == ["sentence-transformers/all-mpnet-base-v2"]

    def test_matcher_fit_without_training_data_raises(self, sample_entities):
        """Test fit() without training data for non-zero-shot mode raises."""
        matcher = Matcher(entities=sample_entities, mode="full")
        with pytest.raises(ValueError, match="training_data is required"):
            matcher.fit()

    def test_matcher_fit_without_training_data_raises_for_bert(self, sample_entities):
        """Test fit() without training data in bert mode raises updated guidance."""
        matcher = Matcher(entities=sample_entities, mode="bert")
        with pytest.raises(ValueError, match="'head-only', 'full', and 'bert'"):
            matcher.fit()

    def test_matcher_fit_hybrid_initializes_pipeline(
        self, sample_entities, monkeypatch
    ):
        class FakeHybridMatcher:
            def __init__(
                self,
                entities,
                blocking_strategy,
                retriever_model,
                reranker_model,
                normalize,
            ):
                self.entities = entities
                self.blocking_strategy = blocking_strategy
                self.retriever_model = retriever_model
                self.reranker_model = reranker_model
                self.normalize = normalize

        monkeypatch.setattr(
            "novelentitymatcher.core.hybrid.HybridMatcher", FakeHybridMatcher
        )

        matcher = Matcher(
            entities=sample_entities,
            mode="hybrid",
            model="mpnet",
            reranker_model="bge-m3",
        )
        matcher.fit(training_data=[{"text": "ignored", "label": "DE"}])

        assert isinstance(matcher._active_matcher, FakeHybridMatcher)
        assert matcher._active_matcher.retriever_model == (
            "sentence-transformers/all-mpnet-base-v2"
        )
        assert matcher._active_matcher.reranker_model == "bge-m3"

    def test_matcher_invalid_mode_uses_mode_error(self, sample_entities):
        """Test invalid mode raises the custom compatibility error type."""
        with pytest.raises(ModeError):
            Matcher(entities=sample_entities, mode="invalid_mode")

    def test_matcher_match_zero_shot(self, sample_entities):
        """Test match() in zero-shot mode."""
        matcher = Matcher(entities=sample_entities, mode="zero-shot")
        matcher.fit()
        result = matcher.match("Deutschland")
        assert result is not None
        assert result["id"] == "DE"
        assert "score" in result

    def test_matcher_match_with_training(self, sample_entities, training_data_full):
        """Test match() after training."""
        matcher = Matcher(entities=sample_entities, mode="full", threshold=0.5)
        matcher.fit(training_data_full, num_epochs=1)
        result = matcher.match("Deutschland")
        assert result is not None
        assert result["id"] == "DE"

    def test_matcher_match_with_training_honors_top_k(self, sample_entities):
        """Test trained mode returns ranked top-k results."""
        matcher = self._build_trained_matcher(sample_entities)

        results = matcher.match("Deutschland", top_k=2)

        assert len(results) == 2
        assert [result["id"] for result in results] == ["DE", "FR"]
        assert [result["score"] for result in results] == [0.82, 0.75]

    def test_matcher_match_with_training_filters_candidates(self, sample_entities):
        """Test trained mode applies candidate filtering before top-k truncation."""
        matcher = self._build_trained_matcher(sample_entities)
        candidates = [sample_entities[1], sample_entities[2]]

        results = matcher.match("Deutschland", top_k=2, candidates=candidates)

        assert [result["id"] for result in results] == ["FR", "US"]
        assert [result["score"] for result in results] == [0.75, 0.41]

    def test_matcher_match_with_training_applies_threshold(self, sample_entities):
        """Test trained mode still filters out results below the threshold."""
        matcher = self._build_trained_matcher(sample_entities, threshold=0.8)

        results = matcher.match("Deutschland", top_k=3)

        assert [result["id"] for result in results] == ["DE"]
        assert [result["score"] for result in results] == [0.82]

    def test_matcher_match_auto_fit(self, sample_entities):
        """Test match() triggers auto-fit if not yet fitted."""
        matcher = Matcher(entities=sample_entities)
        # Don't call fit() explicitly
        result = matcher.match("Deutschland")
        assert result is not None
        assert result["id"] == "DE"

    def test_matcher_match_multiple(self, sample_entities):
        """Test match() with multiple inputs."""
        matcher = Matcher(entities=sample_entities, mode="zero-shot")
        matcher.fit()
        results = matcher.match(["Deutschland", "America", "France"])
        assert len(results) == 3
        assert results[0]["id"] == "DE"
        assert results[1]["id"] == "US"
        assert results[2]["id"] == "FR"

    def test_matcher_predict(self, sample_entities):
        """Test predict() convenience method."""
        matcher = Matcher(entities=sample_entities, mode="zero-shot")
        matcher.fit()
        result = matcher.predict("Deutschland")
        assert result == "DE"

    def test_matcher_predict_multiple(self, sample_entities):
        """Test predict() with multiple inputs."""
        matcher = Matcher(entities=sample_entities, mode="zero-shot")
        matcher.fit()
        results = matcher.predict(["Deutschland", "America"])
        assert results == ["DE", "US"]

    def test_matcher_match_hybrid_single(self, sample_entities):
        class FakeHybridMatcher:
            def match(
                self,
                query,
                blocking_top_k=1000,
                retrieval_top_k=50,
                final_top_k=5,
            ):
                assert query == "Deutschland"
                assert blocking_top_k == 20
                assert retrieval_top_k == 10
                assert final_top_k == 2
                return [
                    {"id": "DE", "score": 0.91, "cross_encoder_score": 0.4},
                    {"id": "FR", "score": 0.32, "cross_encoder_score": 0.9},
                ]

        matcher = Matcher(entities=sample_entities, mode="hybrid", threshold=0.5)
        matcher._hybrid_matcher = FakeHybridMatcher()
        matcher._active_matcher = matcher._hybrid_matcher
        matcher._training_mode = "hybrid"

        results = matcher.match(
            "Deutschland",
            top_k=2,
            blocking_top_k=20,
            retrieval_top_k=10,
        )

        assert results == [{"id": "DE", "score": 0.91, "cross_encoder_score": 0.4}]

    def test_matcher_match_hybrid_multiple(self, sample_entities):
        class FakeHybridMatcher:
            def match_bulk(
                self,
                queries,
                blocking_top_k=1000,
                retrieval_top_k=50,
                final_top_k=5,
                n_jobs=-1,
                chunk_size=None,
            ):
                assert queries == ["Deutschland", "America"]
                assert blocking_top_k == 30
                assert retrieval_top_k == 8
                assert final_top_k == 1
                assert n_jobs == 2
                assert chunk_size == 4
                return [
                    [{"id": "DE", "score": 0.87}],
                    [{"id": "US", "score": 0.22}],
                ]

        matcher = Matcher(entities=sample_entities, mode="hybrid", threshold=0.5)
        matcher._hybrid_matcher = FakeHybridMatcher()
        matcher._active_matcher = matcher._hybrid_matcher
        matcher._training_mode = "hybrid"

        results = matcher.match(
            ["Deutschland", "America"],
            blocking_top_k=30,
            retrieval_top_k=8,
            n_jobs=2,
            chunk_size=4,
        )

        assert results == [{"id": "DE", "score": 0.87}, None]

    def test_matcher_set_threshold_updates_underlying_matchers(self, sample_entities):
        matcher = Matcher(entities=sample_entities, mode="zero-shot")
        matcher._embedding_matcher = type(
            "FakeEmbeddingMatcher", (), {"threshold": 0.7}
        )()
        matcher._entity_matcher = type("FakeEntityMatcher", (), {"threshold": 0.7})()

        returned = matcher.set_threshold(0.42)

        assert returned is matcher
        assert matcher.threshold == 0.42
        assert matcher._embedding_matcher.threshold == 0.42
        assert matcher._entity_matcher.threshold == 0.42

    def test_matcher_get_training_info(self, sample_entities):
        matcher = self._build_trained_matcher(sample_entities, threshold=0.55)
        matcher._detected_mode = "full"
        matcher._has_training_data = True

        info = matcher.get_training_info()

        assert info == {
            "mode": "full",
            "detected_mode": "full",
            "is_trained": True,
            "active_matcher": "_EntityMatcher",
            "has_training_data": True,
            "threshold": 0.55,
        }

    def test_matcher_get_statistics(self, sample_entities):
        matcher = self._build_trained_matcher(sample_entities, threshold=0.55)
        matcher._embedding_matcher = type(
            "FakeEmbeddingMatcher", (), {"embeddings": np.ones((2, 2))}
        )()

        stats = matcher.get_statistics()

        assert stats["num_entities"] == 3
        assert stats["threshold"] == 0.55
        assert stats["training_mode"] == "full"
        assert stats["is_trained"] is True
        assert stats["has_embeddings"] is True
        assert stats["classifier_trained"] is True

    def test_matcher_explain_match_temporarily_lowers_threshold(self, sample_entities):
        class ThresholdAwareMatcher:
            def __init__(self, threshold):
                self.threshold = threshold
                self.seen_thresholds = []

            def match(self, _query, top_k=5, threshold_override=None):
                effective_threshold = (
                    self.threshold if threshold_override is None else threshold_override
                )
                self.seen_thresholds.append(effective_threshold)
                candidates = [
                    {"id": "FR", "score": 0.65, "text": "France"},
                    {"id": "US", "score": 0.25, "text": "United States"},
                ]
                results = [
                    candidate
                    for candidate in candidates
                    if candidate["score"] >= effective_threshold
                ]
                return results[:top_k]

        matcher = Matcher(entities=sample_entities, mode="zero-shot", threshold=0.8)
        fake = ThresholdAwareMatcher(threshold=0.8)
        matcher._active_matcher = fake
        matcher._embedding_matcher = fake
        matcher._training_mode = "zero-shot"

        explanation = matcher.explain_match("France", top_k=2)

        assert explanation["best_match"]["id"] == "FR"
        assert [result["id"] for result in explanation["top_k"]] == ["FR", "US"]
        assert explanation["matched"] is False
        assert matcher.threshold == 0.8
        assert fake.threshold == 0.8
        assert fake.seen_thresholds == [0.0]

    def test_matcher_explain_match_hybrid_uses_wrapper_threshold(self, sample_entities):
        class FakeHybridMatcher:
            def match(
                self,
                _query,
                blocking_top_k=1000,
                retrieval_top_k=50,
                final_top_k=5,
            ):
                return [
                    {"id": "FR", "score": 0.62, "cross_encoder_score": 0.95},
                    {"id": "US", "score": 0.24, "cross_encoder_score": 0.8},
                ][:final_top_k]

        matcher = Matcher(entities=sample_entities, mode="hybrid", threshold=0.8)
        matcher._hybrid_matcher = FakeHybridMatcher()
        matcher._active_matcher = matcher._hybrid_matcher
        matcher._training_mode = "hybrid"

        explanation = matcher.explain_match("France", top_k=2)

        assert [result["id"] for result in explanation["top_k"]] == ["FR", "US"]
        assert explanation["matched"] is False
        assert matcher.threshold == 0.8

    def test_matcher_diagnose_uses_explanation_results(self, sample_entities):
        matcher = self._build_trained_matcher(sample_entities, threshold=0.9)

        diagnosis = matcher.diagnose("Deutschland")

        assert diagnosis["matcher_ready"] is True
        assert diagnosis["issue"] == "Score 0.82 below threshold 0.9"
        assert "matcher.set_threshold(0.8)" in diagnosis["suggestion"]

    def test_matcher_explain_match_requires_fit(self, sample_entities):
        matcher = Matcher(entities=sample_entities)

        with pytest.raises(TrainingError, match="Matcher not ready"):
            matcher.explain_match("Germany")

    def test_matcher_model_alias_resolution(self, sample_entities):
        """Test that model aliases are resolved correctly."""
        matcher = Matcher(entities=sample_entities, model="mpnet")
        assert matcher.model_name == "sentence-transformers/all-mpnet-base-v2"

    def test_matcher_model_alias_bge_base(self, sample_entities):
        """Test BGE base alias resolution."""
        matcher = Matcher(entities=sample_entities, model="bge-base")
        assert matcher.model_name == "BAAI/bge-base-en-v1.5"

    def test_removed_entity_matcher_alias_is_not_exported(self, sample_entities):
        import novelentitymatcher

        with pytest.raises(AttributeError):
            _ = novelentitymatcher.EntityMatcher

    def test_removed_embedding_matcher_alias_is_not_exported(self, sample_entities):
        import novelentitymatcher

        with pytest.raises(AttributeError):
            _ = novelentitymatcher.EmbeddingMatcher

    def test_matcher_no_deprecation_for_new_api(self, sample_entities):
        """Test that new Matcher class doesn't show deprecation warning."""
        from novelentitymatcher import Matcher as NewMatcher

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = NewMatcher(entities=sample_entities)
            # No warnings expected
            assert len(w) == 0


class TestMatcherBERTMode:
    """Tests for BERT mode in Matcher."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
            {"id": "US", "name": "United States", "aliases": ["USA"]},
        ]

    @pytest.fixture
    def training_data_rich(self):
        """Rich training data that should trigger BERT auto-detection."""
        data = []
        for entity_id, country in [("DE", "Germany"), ("FR", "France"), ("US", "USA")]:
            # Include the base name
            data.append({"text": country, "label": entity_id})
            # Include aliases
            if country == "Germany":
                data.append({"text": "Deutschland", "label": entity_id})
            elif country == "France":
                data.append({"text": "Frankreich", "label": entity_id})
            elif country == "USA":
                data.append({"text": "America", "label": entity_id})
            # Add variants
            for i in range(47):  # Total 50 examples per entity
                data.append({"text": f"{country} variant {i}", "label": entity_id})
        return data

    def test_matcher_bert_mode_initialization(self, sample_entities):
        """Test BERT mode initialization."""
        matcher = Matcher(entities=sample_entities, mode="bert")
        assert matcher._training_mode == "bert"
        assert matcher._bert_model_name == "distilbert-base-uncased"

    def test_matcher_bert_mode_fit(self, sample_entities, training_data_rich):
        """Test fit() with BERT mode."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.5)
        matcher.fit(training_data_rich, num_epochs=1)
        assert matcher._has_training_data
        assert matcher._active_matcher is not None
        assert matcher._training_mode == "bert"

    def test_matcher_bert_mode_match(self, sample_entities, training_data_rich):
        """Test match() after BERT training."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.3)
        matcher.fit(training_data_rich, num_epochs=2)
        result = matcher.match("Germany")
        assert result is not None
        assert result["id"] == "DE"

    def test_matcher_bert_mode_with_candidates(
        self, sample_entities, training_data_rich
    ):
        """Test BERT mode with candidate filtering."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.3)
        matcher.fit(training_data_rich, num_epochs=1)

        candidates = [sample_entities[0], sample_entities[1]]  # DE and FR only
        result = matcher.match("Germany", candidates=candidates)
        assert result is not None
        assert result["id"] in ["DE", "FR"]

    def test_matcher_auto_detect_bert_for_rich_data(
        self, sample_entities, training_data_rich
    ):
        """Test auto-detection selects BERT for rich datasets."""
        matcher = Matcher(entities=sample_entities)
        detected = matcher._detect_training_mode(training_data_rich)
        assert detected == "bert"

    def test_matcher_bert_mode_predict(self, sample_entities, training_data_rich):
        """Test predict() convenience method with BERT mode."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.3)
        matcher.fit(training_data_rich, num_epochs=2)
        result = matcher.predict("Germany")
        assert result == "DE"

    def test_matcher_bert_mode_predict_batch(self, sample_entities, training_data_rich):
        """Test predict() with batch in BERT mode."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.3)
        matcher.fit(training_data_rich, num_epochs=2)
        results = matcher.predict(["Germany", "France", "USA"])
        assert len(results) == 3
        assert results[0] == "DE"
        assert results[1] == "FR"
        assert results[2] == "US"

    def test_matcher_set_threshold_updates_bert_matcher(self, sample_entities):
        """Test set_threshold updates BERT matcher."""
        matcher = Matcher(entities=sample_entities, mode="bert")
        matcher._bert_matcher = type("FakeBERTMatcher", (), {"threshold": 0.7})()

        returned = matcher.set_threshold(0.42)

        assert returned is matcher
        assert matcher.threshold == 0.42
        assert matcher._bert_matcher.threshold == 0.42

    def test_matcher_get_statistics_includes_bert(self, sample_entities):
        """Test get_statistics includes BERT classifier status."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.55)
        fake_bert_matcher = type(
            "FakeBERTMatcher",
            (),
            {
                "is_trained": True,
                "classifier": type("FakeClassifier", (), {"is_trained": True})(),
            },
        )()
        matcher._bert_matcher = fake_bert_matcher
        matcher._active_matcher = fake_bert_matcher
        matcher._training_mode = "bert"
        matcher._has_training_data = True

        stats = matcher.get_statistics()

        assert stats["num_entities"] == 3
        assert stats["training_mode"] == "bert"
        assert stats["is_trained"] is True
        assert stats.get("bert_classifier_trained") is True

    def test_matcher_bert_interface_compatibility(self, sample_entities):
        """Test that BERT mode has same interface as other modes."""
        matcher = Matcher(entities=sample_entities, mode="bert")

        # Should support all the same methods
        assert hasattr(matcher, "fit")
        assert hasattr(matcher, "match")
        assert hasattr(matcher, "predict")
        assert hasattr(matcher, "set_threshold")
        assert hasattr(matcher, "get_training_info")
        assert hasattr(matcher, "get_statistics")

    def test_matcher_bert_model_alias_resolution(self, sample_entities):
        """Test BERT model alias resolution."""
        matcher = Matcher(entities=sample_entities, mode="bert", model="distilbert")
        assert matcher._bert_model_name == "distilbert-base-uncased"

    def test_matcher_bert_mode_default_model_uses_bert_backbone(self, sample_entities):
        """Test bert mode uses the BERT default instead of the SetFit default."""
        matcher = Matcher(entities=sample_entities, mode="bert", model="default")
        assert matcher._bert_model_name == "distilbert-base-uncased"
        assert matcher._training_model_name == "sentence-transformers/all-mpnet-base-v2"

    def test_matcher_bert_mode_non_bert_model_warns_and_falls_back(
        self, sample_entities, training_data_rich, monkeypatch, caplog
    ):
        """Test explicit bert mode falls back to the default BERT backbone."""
        trained_models = []

        def fake_train(self, training_data, **kwargs):
            trained_models.append(self.model_name)
            self.is_trained = True

        monkeypatch.setattr(_EntityMatcher, "train", fake_train)

        matcher = Matcher(
            entities=sample_entities,
            mode="bert",
            model="mpnet",
            verbose=True,
        )
        with caplog.at_level("WARNING"):
            matcher.fit(training_data_rich, show_progress=False)

        assert matcher._bert_model_name == "distilbert-base-uncased"
        assert trained_models == ["distilbert-base-uncased"]
        assert "Using non-BERT model 'mpnet' with bert mode" in caplog.text

    def test_matcher_auto_detected_bert_uses_bert_backbone(
        self, sample_entities, training_data_rich, monkeypatch
    ):
        """Test auto-detected bert mode resolves the BERT backbone consistently."""
        trained_models = []

        def fake_train(self, training_data, **kwargs):
            trained_models.append(self.model_name)
            self.is_trained = True

        monkeypatch.setattr(_EntityMatcher, "train", fake_train)

        matcher = Matcher(entities=sample_entities, model="default")
        matcher.fit(training_data_rich, show_progress=False)

        assert matcher._training_mode == "bert"
        assert matcher._bert_model_name == "distilbert-base-uncased"
        assert trained_models == ["distilbert-base-uncased"]

    def test_matcher_explain_match_with_bert(self, sample_entities, training_data_rich):
        """Test explain_match works with BERT mode."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.3)
        matcher.fit(training_data_rich, num_epochs=2)

        explanation = matcher.explain_match("Germany", top_k=2)

        assert explanation["query"] == "Germany"
        assert explanation["matched"] is True
        assert explanation["best_match"] is not None
        assert explanation["best_match"]["id"] == "DE"
        assert len(explanation["top_k"]) <= 2
        assert explanation["mode"] == "bert"

    def test_matcher_diagnose_with_bert(self, sample_entities, training_data_rich):
        """Test diagnose works with BERT mode."""
        matcher = Matcher(entities=sample_entities, mode="bert", threshold=0.3)
        matcher.fit(training_data_rich, num_epochs=2)

        diagnosis = matcher.diagnose("Germany")

        assert diagnosis["matcher_ready"] is True
        assert diagnosis["active_matcher"] == "_EntityMatcher"
        assert diagnosis["matched"] is True

import pytest
import numpy as np
import warnings
from semanticmatcher.core.matcher import EntityMatcher, EmbeddingMatcher, Matcher


class TestEntityMatcher:
    """Tests for EntityMatcher - SetFit-based entity matching."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
            {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
        ]

    @pytest.fixture
    def training_data(self):
        return [
            {"text": "Germany", "label": "DE"},
            {"text": "Deutschland", "label": "DE"},
            {"text": "Deutchland", "label": "DE"},
            {"text": "France", "label": "FR"},
            {"text": "Frankreich", "label": "FR"},
            {"text": "USA", "label": "US"},
            {"text": "America", "label": "US"},
        ]

    def test_entity_matcher_init(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities)
        assert matcher.entities == sample_entities

    def test_entity_matcher_with_model(self, sample_entities):
        matcher = EntityMatcher(
            entities=sample_entities,
            model_name="sentence-transformers/paraphrase-mpnet-base-v2",
        )
        assert matcher.model_name == "sentence-transformers/paraphrase-mpnet-base-v2"

    def test_entity_matcher_default_threshold(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities)
        assert matcher.threshold == 0.7

    def test_entity_matcher_custom_threshold(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities, threshold=0.5)
        assert matcher.threshold == 0.5

    def test_entity_matcher_without_training_raises(self, sample_entities):
        matcher = EntityMatcher(entities=sample_entities)
        with pytest.raises(RuntimeError, match="not trained"):
            matcher.predict("Germany")

    def test_entity_matcher_train(self, sample_entities, training_data):
        matcher = EntityMatcher(entities=sample_entities)
        matcher.train(training_data, num_epochs=1)
        assert matcher.is_trained

    def test_entity_matcher_predict_single(self, sample_entities, training_data):
        matcher = EntityMatcher(entities=sample_entities)
        matcher.train(training_data, num_epochs=1)
        result = matcher.predict("Deutchland")
        assert result == "DE"

    def test_entity_matcher_predict_multiple(self, sample_entities, training_data):
        matcher = EntityMatcher(entities=sample_entities)
        matcher.train(training_data, num_epochs=1)
        results = matcher.predict(["Deutchland", "America", "France"])
        assert results == ["DE", "US", "FR"]

    def test_entity_matcher_predict_below_threshold(
        self, sample_entities, training_data
    ):
        matcher = EntityMatcher(entities=sample_entities, threshold=0.99)
        matcher.train(training_data, num_epochs=1)
        result = matcher.predict("UnknownCountry123")
        assert result is None


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
            "semanticmatcher.core.matcher.SentenceTransformer", FakeModel
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
            "semanticmatcher.core.matcher.SentenceTransformer", FakeModel
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

    def test_matcher_fit_auto_with_small_data(self, sample_entities, training_data_small):
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

    def test_matcher_fit_mode_override_to_zero_shot(self, sample_entities, training_data_full):
        """Test fit() with mode override to zero-shot."""
        matcher = Matcher(entities=sample_entities)
        matcher.fit(training_data_full, mode="zero-shot")
        assert matcher._training_mode == "zero-shot"

    def test_matcher_fit_mode_override_to_full(self, sample_entities, training_data_small):
        """Test fit() with mode override to full training."""
        matcher = Matcher(entities=sample_entities)
        matcher.fit(training_data_small, mode="full", num_epochs=1)
        assert matcher._training_mode == "full"

    def test_matcher_fit_without_training_data_raises(self, sample_entities):
        """Test fit() without training data for non-zero-shot mode raises."""
        matcher = Matcher(entities=sample_entities, mode="full")
        with pytest.raises(ValueError, match="training_data is required"):
            matcher.fit()

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

    def test_matcher_model_alias_resolution(self, sample_entities):
        """Test that model aliases are resolved correctly."""
        matcher = Matcher(entities=sample_entities, model="mpnet")
        assert matcher.model_name == "sentence-transformers/all-mpnet-base-v2"

    def test_matcher_model_alias_bge_base(self, sample_entities):
        """Test BGE base alias resolution."""
        matcher = Matcher(entities=sample_entities, model="bge-base")
        assert matcher.model_name == "BAAI/bge-base-en-v1.5"

    def test_matcher_backward_compatibility_entity_matcher(self, sample_entities):
        """Test that importing old classes shows deprecation warning."""
        # Import through the public API - warning happens at import time
        # The import itself triggers the warning
        import semanticmatcher

        # Clear any previous warnings
        semanticmatcher.__dict__.pop('EntityMatcher', None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Access through __getattr__ which triggers deprecation
            OldEntityMatcher = semanticmatcher.EntityMatcher
            # Should see deprecation warning
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_matcher_backward_compatibility_embedding_matcher(self, sample_entities):
        """Test EmbeddingMatcher deprecation warning."""
        import semanticmatcher

        # Clear cached import
        semanticmatcher.__dict__.pop('EmbeddingMatcher', None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Access through __getattr__ which triggers deprecation
            OldEmbeddingMatcher = semanticmatcher.EmbeddingMatcher
            assert len(w) >= 1
            assert any(issubclass(warning.category, DeprecationWarning) for warning in w)

    def test_matcher_no_deprecation_for_new_api(self, sample_entities):
        """Test that new Matcher class doesn't show deprecation warning."""
        from semanticmatcher import Matcher as NewMatcher

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _ = NewMatcher(entities=sample_entities)
            # No warnings expected
            assert len(w) == 0

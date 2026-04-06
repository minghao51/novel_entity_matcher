# Testing Patterns

**Analysis Date:** 2026-04-06

## Test Framework

**Runner:**
- pytest >= 8.4.2
- Config: `pyproject.toml` → `[tool.pytest.ini_options]`

**Assertion Library:**
- Built-in pytest assertions (`assert`, `pytest.raises`, `pytest.fixture`)

**Run Commands:**
```bash
pytest                     # Run all tests
pytest -xvs                # Verbose, stop on first failure
pytest -m "not slow"       # Skip slow tests
pytest tests/test_core/    # Run specific directory
pytest -k "matcher"        # Run tests matching "matcher"
```

**Test Markers:**
- `integration` — tests that depend on external services or network access
- `slow` — tests that are expensive to run in default CI
- `hf` — Hugging Face model-backed tests
- `llm` — tests that make actual LLM API calls (require API key, slow)
- `llm_mocked` — tests that involve LLM logic but use mocks instead of real API calls
- `e2e` — end-to-end / feature tests that exercise multiple components

**Async Support:**
- pytest-asyncio >= 1.2.0
- `asyncio_mode = "auto"` in pytest config

## Test File Organization

**Location:**
- Separate `tests/` directory at project root, mirroring `src/` structure

**Naming:**
- `test_*.py` prefix for all test files
- Test subdirectories match source modules: `tests/test_core/`, `tests/test_utils/`, `tests/test_backends/`, `tests/test_ingestion/`

**Structure:**
```
tests/
├── conftest.py                          # Shared fixtures
├── __init__.py
├── test_core/
│   ├── test_matcher.py
│   ├── test_classifier.py
│   ├── test_async_matcher.py
│   ├── test_bert_classifier.py
│   ├── test_normalizer.py
│   ├── test_async_utils.py
│   └── ...
├── test_utils/
│   ├── test_logging_config.py
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_benchmarks.py
│   └── test_validation.py
├── test_backends/
│   ├── test_huggingface.py
│   ├── test_litellm.py
│   ├── test_reranker_contracts.py
│   └── ...
├── test_ingestion/
│   ├── test_cli.py
│   └── test_timezones.py
├── test_integration.py
├── test_novelty_detector.py
├── test_llm_proposer.py
└── ...
```

## Test Structure

**Suite Organization:**
```python
class TestEmbeddingMatcher:
    """Tests for EmbeddingMatcher - similarity-based matching."""

    @pytest.fixture
    def sample_entities(self):
        return [
            {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
            {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        ]

    def test_embedding_matcher_init(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        assert matcher.entities == sample_entities

    def test_embedding_matcher_match(self, sample_entities):
        matcher = EmbeddingMatcher(entities=sample_entities)
        matcher.build_index()
        result = matcher.match("Deutschland")
        assert result is not None
        assert result["id"] == "DE"
```

**Patterns:**
- Class-based test suites: `Test*` classes grouping related tests
- Method-based tests: `test_*` methods with descriptive names
- Fixtures for setup: `@pytest.fixture` for reusable test data
- Docstrings on classes and sometimes methods describing test purpose
- Arrange-Act-Assert pattern within test methods

**Global Fixtures (conftest.py):**
```python
@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the global model cache before each test."""
    cache = get_default_cache()
    cache.clear()
    yield
    cache.clear()
```

## Mocking

**Framework:** `unittest.mock` (Mock, patch, monkeypatch)

**Patterns:**
```python
# Monkeypatch for class replacement
def test_embedding_matcher_resolves_dynamic_alias(self, sample_entities, monkeypatch):
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
    # ... test logic ...
    assert loaded_models == ["sentence-transformers/all-mpnet-base-v2"]
```

```python
# Patch for function mocking (LLM tests)
from unittest.mock import Mock, patch

@patch("novelentitymatcher.novelty.proposal.llm.litellm")
def test_propose_classes(mock_litellm, proposer, sample_novel_samples, mock_llm_response):
    mock_litellm.completion.return_value = Mock(
        choices=[Mock(message=Mock(content=mock_llm_response))]
    )
    # ... test logic ...
```

**What to Mock:**
- External model loading (SentenceTransformer) with fake encode methods
- LLM API calls (litellm.completion)
- Network/external service calls
- Global caches (autouse fixture clears model cache)

**What NOT to Mock:**
- Core business logic — test actual behavior
- Numpy operations — use real arrays with synthetic data
- Config/dataclass creation — use real instances

## Fixtures and Factories

**Test Data:**
```python
@pytest.fixture
def sample_entities(self):
    return [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland", "Deutchland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
    ]

@pytest.fixture
def reference_embeddings(self):
    return np.array(
        [[1.0, 0.0, 0.0, 0.0], [0.95, 0.05, 0.0, 0.0], ...],
        dtype=np.float32,
    )

@pytest.fixture
def detector(self):
    return NoveltyDetector(
        config=DetectionConfig(
            strategies=["confidence", "knn_distance", "clustering"],
            confidence=ConfidenceConfig(threshold=0.6),
            knn_distance=KNNConfig(distance_threshold=0.25),
        )
    )
```

**Location:**
- Fixtures defined in test classes as methods with `@pytest.fixture`
- Shared fixtures in `tests/conftest.py` (currently: `clear_model_cache`)
- Synthetic numpy arrays with `dtype=np.float32` for embedding tests

## Coverage

**Requirements:** None enforced (no coverage threshold in config)

**View Coverage:**
```bash
pytest --cov=novelentitymatcher --cov-report=html
pytest --cov=novelentitymatcher --cov-report=term-missing
```

## Test Types

**Unit Tests:**
- Test individual classes and functions in isolation
- Use synthetic/small data (3-6 entities, small embedding arrays)
- Mock external dependencies (models, APIs)
- Located in `tests/test_core/`, `tests/test_utils/`, `tests/test_backends/`

**Integration Tests:**
- Test complete pipelines end-to-end
- Use `@pytest.mark.integration` for external service tests
- `tests/test_integration.py`, `tests/test_integration_extended.py`
- Exercise multiple components together (Matcher + NoveltyDetector + storage)

**E2E Tests:**
- Marked with `@pytest.mark.e2e`
- Full pipeline tests with real data flows
- `tests/test_pipeline_orchestrator.py`, `tests/test_discovery_pipeline.py`

**LLM Tests:**
- `@pytest.mark.llm` — real API calls (slow, require API key)
- `@pytest.mark.llm_mocked` — mocked LLM responses (fast, CI-safe)
- `tests/test_llm_proposer.py` uses mocking pattern

## Common Patterns

**Async Testing:**
```python
# pytest-asyncio with auto mode — async tests just work
async def test_async_match(self, sample_entities):
    matcher = await AsyncMatcher(entities=sample_entities)
    result = await matcher.match("Deutschland")
    assert result["id"] == "DE"
```

**Error Testing:**
```python
def test_classifier_without_training_raises(self, labels):
    clf = SetFitClassifier(labels=labels)
    with pytest.raises(RuntimeError, match="not trained"):
        clf.predict("test")

def test_matcher_invalid_mode(self):
    with pytest.raises(ModeError, match="Invalid mode"):
        Matcher(entities=[], mode="invalid")
```

**Logging Test Helpers:**
```python
def _reset_novelentitymatcher_logging():
    logger = logging.getLogger("novelentitymatcher")
    logger.handlers.clear()
    logger.setLevel(logging.NOTSET)
    logging_config._logging_configured = False
    return logger
```

---

*Testing analysis: 2026-04-06*

# Testing Patterns

**Analysis Date:** 2026-04-30

## Framework & Setup

### Test Runner

- **Framework:** pytest (>= 9.0.3)
- **Config:** `pyproject.toml` `[tool.pytest.ini_options]` (line 203-218)
- **Async support:** `pytest-asyncio` (>= 1.2.0) with `asyncio_mode = "auto"`
- **Import mode:** `--import-mode=importlib`

### Key pytest settings

```toml
testpaths = ["tests"]
addopts = ["--strict-markers", "-ra", "--durations=10", "--import-mode=importlib"]
```

### Registered Markers

Defined in `pyproject.toml:206-217`:

| Marker | Purpose |
|--------|---------|
| `unit` | Fast isolated tests with no external dependencies |
| `integration` | Tests that depend on external services or network access |
| `slow` | Expensive tests for non-default CI |
| `e2e` | End-to-end tests exercising multiple components |
| `hf` | Hugging Face model-backed tests |
| `llm` | Tests making actual LLM API calls (require key, slow) |
| `llm_mocked` | LLM logic tests using mocks instead of real API |
| `serial` | Tests that cannot run in parallel |
| `network` | Tests requiring internet access |
| `smoke` | Critical path tests |

### Run Commands

```bash
uv run pytest                          # Run all tests
uv run pytest tests/unit/              # Unit tests only
uv run pytest tests/integration/       # Integration tests only
uv run pytest -m unit                  # By marker
uv run pytest -m "not slow"            # Exclude slow tests
uv run pytest -m smoke                 # Smoke tests only
uv run pytest --durations=10           # Show 10 slowest (default from addopts)
```

## Test Structure

### Directory Layout

```
tests/
├── conftest.py                          # Global fixtures + auto-marking
├── fixtures/
│   ├── sample_countries.json            # (currently empty)
│   └── sample_texts.json                # (currently empty)
├── unit/
│   ├── backends/
│   ├── benchmarks/
│   ├── core/
│   │   ├── test_async_utils.py
│   │   └── test_normalizer.py
│   ├── ingestion/
│   ├── monitoring/
│   ├── novelty/
│   │   ├── test_ann_index.py
│   │   ├── test_conformal.py
│   │   ├── test_novelty_detector.py
│   │   ├── test_novelty_detector_lifecycle.py
│   │   ├── test_oneclass_strategy.py
│   │   ├── test_pattern_strategy.py
│   │   ├── test_prototypical_strategy.py
│   │   ├── test_setfit_novelty.py
│   │   └── test_signal_combiner.py
│   ├── pipeline/
│   ├── utils/
│   ├── test_config.py
│   ├── test_discovery_pipeline.py
│   ├── test_llm_proposer.py
│   ├── test_packaging.py
│   ├── test_phase2_features.py
│   ├── test_security_logging.py
│   ├── test_smoke_paths.py
│   └── test_smoke_static_embedding.py
└── integration/
    ├── backends/
    ├── core/
    ├── utils/
    ├── test_async_sync_parity.py
    ├── test_discovery_pipeline_extended.py
    ├── test_integration.py
    ├── test_integration_extended.py
    └── test_novel_entity_matcher.py
```

### File Naming

- Test files: `test_<module_or_feature>.py`
- Test directories mirror `src/novelentitymatcher/` package structure
- E.g., `tests/unit/novelty/test_novelty_detector.py` tests `src/novelentitymatcher/novelty/core/detector.py`

### Auto-Marking

`tests/conftest.py:15-28` automatically applies markers based on file path:
- Files under `tests/unit/` → `@pytest.mark.unit`
- Files under `tests/integration/` → `@pytest.mark.integration` + `@pytest.mark.slow`
- Async tests get `@pytest.mark.anyio` added automatically

## Test Patterns

### Test Class Organization

Tests are organized into classes named `Test<Feature>`:

```python
class TestTextNormalizer:
    """Tests for TextNormalizer - text normalization utilities."""

    def test_normalizer_init_defaults(self):
        normalizer = TextNormalizer()
        assert normalizer.lowercase is True
```

- Each class has a docstring describing what it tests
- Methods are plain functions (no `self` when using standalone test functions)
- Descriptive test names: `test_<what>_<condition>_<expected>`

### Standalone Test Functions

Many tests use simple standalone functions without classes:

```python
def test_config_loads_default_and_nested_access(tmp_path, monkeypatch):
    ...
```

### Fixtures

**Global autouse fixture** (`conftest.py:6-12`):
```python
@pytest.fixture(autouse=True)
def clear_model_cache():
    """Clear the global model cache before each test."""
    cache = get_default_cache()
    cache.clear()
    yield
    cache.clear()
```

**Per-test-class fixtures** (defined inside test classes):

```python
class TestNoveltyDetector:
    @pytest.fixture
    def sample_texts(self):
        return ["quantum physics research", ...]

    @pytest.fixture
    def reference_embeddings(self):
        return np.array([...], dtype=np.float32)
```

### Mocking

**`monkeypatch`** (preferred over `unittest.mock`):

```python
def test_static_embedding_backend_model2vec_path(monkeypatch):
    monkeypatch.setattr("model2vec.StaticModel", _MockModel2Vec)
    # or:
    monkeypatch.setattr(Config, "_default_config_candidates", lambda self: [default_path])
```

**Inline mock classes** (common pattern):

```python
class _MockModel2Vec:
    dim = 256
    def __init__(self, model_name=None):
        pass
    @classmethod
    def from_pretrained(cls, model_name):
        return cls()
    def encode(self, texts):
        return np.random.rand(len(texts), self.dim)
```

**Module-level monkeypatching** (for harder cases):

```python
import novelentitymatcher.backends.static_embedding as se
original = se.get_cached_sentence_transformer
se.get_cached_sentence_transformer = lambda name, trust_remote_code=False: FakeSentenceTransformer()
try:
    # test code
finally:
    se.get_cached_sentence_transformer = original
```

**Availability guard pattern** (for optional dependencies):

```python
def _hdbscan_available() -> bool:
    try:
        import hdbscan  # noqa: F401
        return True
    except ImportError:
        return False
```

### Parametrization

Not heavily used, but markers serve as a form of test categorization. Tests rely on descriptive names and separate test files for different scenarios.

### Async Testing

- `asyncio_mode = "auto"` — no need for `@pytest.mark.asyncio` on most async tests
- `conftest.py:16-18` adds `anyio` marker to async tests for compatibility
- Pattern:
  ```python
  def test_matcher_sync_async_parity():
      sync_result = matcher.match(queries)
      async_result = asyncio.run(matcher.match_async(queries))
      assert len(sync_result) == len(async_result)
  ```

## Coverage

### Coverage Tool

- `.coverage` file exists at project root
- No explicit coverage threshold enforced in config
- No `coverage` configuration in `pyproject.toml`

### CI Coverage

- Pre-commit hooks run mypy but not coverage
- No coverage gate in CI pipeline (no `.github/workflows/` CI config observed)

## Test Data

### Fixtures Directory

- `tests/fixtures/` — contains JSON files (`sample_countries.json`, `sample_texts.json`), currently empty
- Most test data is **inline** within test files or fixtures

### Inline Test Data Pattern

Test data is typically defined as fixtures within test classes:

```python
@pytest.fixture
def sample_entities(self):
    return [
        {"id": "physics", "name": "Quantum Physics"},
        {"id": "cs", "name": "Computer Science"},
    ]
```

### tmp_path Usage

Tests use pytest's `tmp_path` fixture for file-based tests:

```python
def test_config_loads_default_and_nested_access(tmp_path, monkeypatch):
    default_path = tmp_path / "default.yaml"
    default_path.write_text("default_model: base-model\n")
```

## Test Types

### Unit Tests (`tests/unit/`)

- Fast, isolated, no external dependencies
- Heavy use of `monkeypatch` to mock models and external services
- Test pure logic: config loading, validation, normalization, error formatting
- Examples: `test_config.py`, `test_normalizer.py`, `test_security_logging.py`

### Integration Tests (`tests/integration/`)

- May load real models (e.g., `model="minilm"`)
- Test cross-component interactions (matcher + novelty detector + pipeline)
- Auto-marked as `slow`
- Examples: `test_integration.py`, `test_async_sync_parity.py`, `test_novel_entity_matcher.py`

### Smoke Tests

- Marked with `@pytest.mark.smoke`
- Critical path verification
- Example: `test_smoke_static_embedding.py`, `test_smoke_paths.py`

### E2E Tests

- Marked with `@pytest.mark.e2e`
- Exercise full pipelines end-to-end
- Example: `test_discovery_pipeline.py`, `test_discovery_pipeline_extended.py`

## Common Patterns

### Error Testing

```python
def test_redacts_openai_key(self):
    result = _redact_api_keys("api key is sk-abc123def456ghi789jkl012")
    assert "sk-abc123def456ghi789jkl012" not in result
    assert "...REDACTED..." in result
```

### Isolated Config Testing

```python
def test_config_instances_do_not_share_state(tmp_path, monkeypatch):
    monkeypatch.setattr(Config, "_default_config_candidates", lambda self: [first_default])
    first = Config()
    monkeypatch.setattr(Config, "_default_config_candidates", lambda self: [second_default])
    second = Config()
    assert first.default_model == "first"
    assert second.default_model == "second"
```

### Sync/Async Parity Testing

```python
def test_matcher_sync_async_parity():
    sync_result = matcher.match(queries)
    async_result = asyncio.run(matcher.match_async(queries))
    for sync_match, async_match in zip(sync_result, async_result, strict=False):
        assert sync_match["id"] == async_match["id"]
        assert abs(sync_match["score"] - async_match["score"]) < 1e-9
```

---

*Testing analysis: 2026-04-30*

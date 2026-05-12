# Testing

## Framework & Setup

- **Framework:** pytest 9.x with pytest-asyncio
- **Configuration:** `pyproject.toml [tool.pytest.ini_options]` (lines 152-167)
- **Import mode:** `importlib` (avoids `sys.path` manipulation)
- **Async mode:** `auto` — all `async def test_*` functions run as coroutines without explicit markers
- **Strict markers:** enabled — all markers must be registered
- **Max failures:** 10 (`--maxfail=10`)
- **Durations:** shows top 10 slowest tests (`--durations=10`)

## Test Directory Structure

```
tests/
├── conftest.py              # Global fixtures + auto-marker assignment
├── fixtures/                # Static test data
│   ├── sample_countries.json
│   └── sample_texts.json
├── unit/                    # Fast, isolated, no external deps
│   ├── backends/            # Backend contract tests
│   ├── benchmarks/          # Benchmark utility tests
│   ├── core/                # Core matcher/classifier/embedding tests
│   ├── ingestion/           # Data ingestion tests
│   ├── monitoring/          # Metrics/performance tests
│   ├── novelty/             # Novelty detection unit tests
│   │   ├── drift/           # Drift detection tests
│   │   └── ...              # Strategy-specific tests
│   ├── pipeline/            # Pipeline orchestrator/stage tests
│   ├── utils/               # Utility function tests
│   └── test_*.py            # Top-level unit tests
├── integration/             # External deps, models, network
│   ├── backends/            # HuggingFace, static embedding tests
│   ├── core/                # Matcher/classifier/hierarchy integration
│   ├── utils/               # Embedding utils integration
│   └── test_*.py            # Pipeline & matcher integration tests
└── e2e/                     # End-to-end tests (currently empty)
```

## Test Markers

Defined in `pyproject.toml [tool.pytest.ini_options]`:

| Marker | Purpose | CI Selection |
|--------|---------|-------------|
| `unit` | Fast isolated tests, no external deps | `not integration and not slow` |
| `integration` | Tests depending on external services/network | `integration or slow` |
| `slow` | Expensive tests (model loading, training) | `integration or slow` |
| `e2e` | End-to-end feature tests | (not currently run in CI) |
| `hf` | HuggingFace model-backed tests | (subset of integration) |
| `llm` | Real LLM API calls (API key required) | (not run in CI) |
| `llm_mocked` | LLM logic with mocks instead of real calls | (unit-level) |
| `serial` | Cannot run in parallel | (future use) |
| `network` | Requires internet access | (subset of integration) |
| `smoke` | Critical path tests | `-m smoke` |

### Auto-Marker Assignment (`tests/conftest.py:15-29`)

The `pytest_collection_modifyitems` hook automatically assigns markers based on file path:
- Files under `tests/unit/` → `@pytest.mark.unit`
- Files under `tests/integration/` → `@pytest.mark.integration` + `@pytest.mark.slow`
- Async tests also get `@pytest.mark.anyio` automatically

## Fixtures

### Global Fixtures (`tests/conftest.py`)

- **`clear_model_cache` (autouse)** — Clears the global embedding model cache before and after every test to prevent cross-test contamination

### Per-Test Fixtures

Fixtures are defined locally in test files, typically as:
- **Data fixtures:** `sample_entities`, `training_data`, `test_queries`, `sample_embeddings`, `reference_labels`
- **Instance fixtures:** `matcher` (trained `EmbeddingMatcher`), `detector` (configured `NoveltyDetector`), `trained_matcher`
- **Private helpers:** `_make_entities()` factory pattern for entity creation

Example fixture pattern:
```python
@pytest.fixture
def matcher():
    entities = _make_entities(["apple", "banana", "cherry"])
    m = EmbeddingMatcher(entities=entities, model_name="potion-32m")
    m.build_index()
    return m
```

## Mocking Patterns

### Primary Approaches

1. **`monkeypatch`** (most common, 89+ uses) — preferred for:
   - Replacing methods: `monkeypatch.setattr(Config, "_default_config_candidates", lambda self: [path])`
   - Replacing module attributes: `monkeypatch.setattr("model2vec.StaticModel", MockClass)`
   - Environment variables: `monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-123")`
   - Module injection: `monkeypatch.setitem(sys.modules, "litellm", mock_litellm)`

2. **`unittest.mock`** — used for:
   - `MagicMock` for complex mock objects
   - `patch` as context manager for scoped mocking
   - `Mock` for simple stub objects

### Skip Patterns

Tests use `@pytest.mark.skipif` for optional dependencies:
```python
def _hdbscan_available() -> bool:
    try:
        import hdbscan
        return True
    except ImportError:
        return False

@pytest.mark.skipif(not _hdbscan_available(), reason="hdbscan not installed")
```

## Test Organization Patterns

### Class-Based Grouping

Related tests are grouped in classes:
```python
class TestAddEntities:
    def test_add_entities_new_ids_in_index(self, matcher): ...
    def test_add_entities_increases_entity_count(self, matcher): ...

class TestNovelClassDetectionIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline(self, trained_matcher): ...
```

### Test Naming

- Descriptive names: `test_config_loads_default_and_nested_access`, `test_add_entities_raises_before_build_index`
- Assert-assertive patterns: `test_<thing>_<expected_behavior>`
- Negative tests: `test_*_raises_*`, `test_*_exits_non_zero_on_failure`

### Helper Functions

- Private helpers prefixed with `_`: `_make_entities()`, `_hdbscan_available()`, `_leiden_available()`
- Factory functions for creating test data

## Coverage

### Configuration (`pyproject.toml [tool.coverage]`)

- **Source:** `src/`
- **Omit:** `*/tests/*`, `*/__pycache__/*`
- **Excluded lines:**
  - `pragma: no cover`
  - `def __repr__`
  - `raise AssertionError`
  - `raise NotImplementedError`
  - `if __name__ == .__main__.:`
- **Coverage file:** `.coverage` at project root

### CI Coverage

Coverage is not enforced as a CI gate. The configuration exists for local development use.

## CI Test Pipeline (`.github/workflows/ci.yml`)

| Job | Trigger | What it runs |
|-----|---------|-------------|
| `pre-commit` | All pushes + PRs | Full pre-commit hook suite (ruff, mypy, pip-audit, uv-lock, etc.) |
| `typecheck` | All pushes + PRs | `mypy src/novelentitymatcher` |
| `test-fast` | All pushes + PRs | Smoke tests, then `not integration and not slow`, coverage floor |
| `test-heavy` | Main push or manual | `integration or slow` |
| `test-matrix` | Main push or manual | Python 3.10, 3.11, 3.12 — `not integration and not slow`, pip-audit |
| `security` | All pushes + PRs | `pip-audit` with CVE ignores |
| `build` | All pushes + PRs | `uv sync --extra all`, build sdist/wheel, twine check |

All jobs use `uv sync --frozen --group dev` for dependency installation.

## Running Tests

```bash
# All tests
uv run pytest

# Unit tests only (fast)
uv run pytest -m "not integration and not slow"

# Integration tests only
uv run pytest -m "integration or slow"

# Smoke tests
uv run pytest -m smoke

# Specific marker
uv run pytest -m "not hf and not llm"

# Single test file
uv run pytest tests/unit/test_config.py

# With coverage
uv run pytest --cov=src --cov-report=term-missing
```

## Test Data

### Fixture Files (`tests/fixtures/`)

- `sample_countries.json` — country entity test data
- `sample_texts.json` — sample text inputs for matching tests

### In-Code Test Data

Most test data is constructed inline via fixtures and helper functions rather than external files. This keeps tests self-contained and readable.

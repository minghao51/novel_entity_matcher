# Testing

**Analysis Date:** 2026-04-23

## Framework

**Test Framework:**
- pytest 8.4.2+ - Primary test framework
- pytest-asyncio 1.2.0+ - Async test support
- Python asyncio_mode: `auto` (configured in pyproject.toml)

**Test Structure:**
- `tests/unit/` - Fast isolated tests with no external dependencies
- `tests/integration/` - Tests that depend on external services or network access
- `tests/fixtures/` - Shared test data and fixtures
- `tests/conftest.py` - Pytest configuration and shared fixtures

## Test Organization

**Directory Layout:**
```
tests/
├── unit/
│   ├── core/          # Core matcher tests
│   ├── novelty/       # Novelty detection tests
│   ├── pipeline/      # Pipeline orchestration tests
│   ├── backends/      # Backend integration tests
│   ├── utils/         # Utility function tests
│   └── ingestion/     # Ingestion script tests
├── integration/
│   ├── core/          # Integration tests for core
│   ├── backends/      # Backend integration tests
│   └── utils/         # Integration tests for utils
├── fixtures/          # Test data
└── conftest.py        # Shared fixtures
```

**Test Files:**
- Naming: `test_*.py`
- Examples: `test_matcher.py`, `test_novelty_detector.py`, `test_discovery_pipeline.py`

## Test Markers

**Unit Tests:**
- `@pytest.mark.unit` - Fast isolated tests with no external dependencies
- Excluded from default test runs on PRs (run with `pytest -m unit`)

**Integration Tests:**
- `@pytest.mark.integration` - Tests with external dependencies
- Require network access, model downloads, or external services

**Performance Tests:**
- `@pytest.mark.slow` - Tests that are expensive to run
- Excluded from default CI runs

**Feature Tests:**
- `@pytest.mark.e2e` - End-to-end / feature tests
- `@pytest.mark.smoke` - Critical path tests

**Model Tests:**
- `@pytest.mark.hf` - Hugging Face model-backed tests
- `@pytest.mark.llm` - LLM API tests (require API key, slow)
- `@pytest.mark.llm_mocked` - LLM logic tests with mocks

**Utility Tests:**
- `@pytest.mark.serial` - Tests that cannot be run in parallel
- `@pytest.mark.network` - Tests requiring internet access

## Test Execution

**Fast Tests (default):**
```bash
uv run pytest -q -m "not integration and not slow"
```

**All Tests:**
```bash
uv run pytest
```

**Specific Marker:**
```bash
uv run pytest -m llm
uv run pytest -m integration
```

**Async Tests:**
- Automatically handled by pytest-asyncio
- Marked with `@pytest.mark.asyncio`
- Supported in both unit and integration tests

## Mocking

**External Dependencies:**
- Hugging Face models: Use mocks in unit tests
- LLM APIs: Use `@pytest.mark.llm_mocked` with mocks
- Network requests: Use `pytest.mark.network` to skip or mock

**Mocking Strategies:**
- unittest.mock for function mocking
- pytest fixtures for shared mock objects
- Conditional imports with `TYPE_CHECKING`

## Fixtures

**Shared Fixtures:**
- `tests/conftest.py` contains shared fixtures
- Examples: `trained_matcher`, `sample_entities`, `test_corpus`

**Fixture Usage:**
```python
@pytest.fixture
def sample_entities():
    return [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "US", "name": "United States", "aliases": ["USA"]},
    ]

def test_match(sample_entities):
    matcher = Matcher(entities=sample_entities)
    # ...
```

## Coverage

**Coverage Goals:**
- Unit tests: Core functionality
- Integration tests: External integrations
- Currently: No explicit coverage tracking (not in CI)

**Coverage Areas:**
- Core matching: `tests/unit/core/`
- Novelty detection: `tests/unit/novelty/`
- Pipeline orchestration: `tests/unit/pipeline/`
- Backends: `tests/unit/backends/` and `tests/integration/backends/`
- Utilities: `tests/unit/utils/` and `tests/integration/utils/`

## CI/CD Testing

**PR Testing:**
- Fast tests only (exclude integration and slow)
- Python 3.11 target
- Command: `pytest -q -m "not integration and not slow"`

**Branch Push Testing:**
- Fast tests only (exclude integration and slow)
- Python 3.11 target
- Same as PR testing

**Main Branch Testing:**
- Fast tests matrix: Python 3.9, 3.10, 3.11, 3.12
- Heavy tests: Integration and slow tests (Python 3.11)
- Triggered on main branch push and workflow_dispatch

**Test Workflows:**
- `.github/workflows/test.yml` - Test automation
- Fast tests: PRs and non-main pushes
- Heavy tests: Main branch pushes and manual dispatch

## Test Data

**Fixtures Location:**
- `tests/fixtures/` - Shared test data
- Examples: sample entity lists, test corpora

**Synthetic Data:**
- Generated in fixtures or test functions
- Minimal datasets for fast tests

**Real Data:**
- External data sources for integration tests
- Ingestion scripts fetch from real APIs

## Async Testing

**Async Test Support:**
- pytest-asyncio with `auto` mode
- Async fixtures supported
- Coroutine functions in tests

**Example:**
```python
@pytest.mark.asyncio
async def test_async_match(trained_matcher):
    result = await trained_matcher.match_async("Germany")
    assert result["id"] == "DE"
```

## Test Configuration

**Pytest Configuration:**
- Located in `pyproject.toml` under `[tool.pytest.ini_options]`
- testpaths: `["tests"]`
- addopts: `["--strict-markers", "-ra", "--durations=10", "--import-mode=importlib"]`
- asyncio_mode: `"auto"`
- Strict markers enabled

**Marker Definitions:**
- unit: Fast isolated tests
- integration: External service tests
- slow: Expensive tests
- e2e: End-to-end tests
- hf: Hugging Face tests
- llm: LLM API tests (requires key, slow)
- llm_mocked: LLM tests with mocks
- serial: Non-parallelizable tests
- network: Tests requiring internet
- smoke: Critical path tests

---

*Testing analysis: 2026-04-23*

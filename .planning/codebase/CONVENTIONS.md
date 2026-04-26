# Code Conventions

**Analysis Date:** 2026-04-23

## Code Style

**Formatting:**
- Tool: black 23.0.0+
- Style: PEP 8 with Black defaults
- Line length: 88 characters (Black default)
- Quotes: Double quotes preferred
- CI check: `uv run black --check .`

**Linting:**
- Tool: ruff 0.1.0+
- CI check: `uv run ruff check .`
- Rules: PEP 8, flake8, isort, pydocstyle integration
- Fast: Ruff is significantly faster than flake8+isort

**Type Checking:**
- Tool: mypy 1.19.1+
- Python version target: 3.13
- Mode: Not strict yet (gradual migration)
- CI check: Manual (not in CI workflow)
- Relaxed rules for:
  - External packages (setfit, transformers, torch, etc.)
  - Legacy implementation files (self_knowledge_impl, prototypical_impl, etc.)
  - Optional types (no_implicit_optional=false for strategy modules)

## Naming Patterns

**Classes:**
- PascalCase
- Descriptive names: `NovelEntityMatcher`, `SignalCombiner`, `ScalableClusterer`
- Suffixes: `Strategy`, `Config`, `Backend`, `Detector`, `Manager`

**Functions/Methods:**
- snake_case
- Verb-first for actions: `match_async`, `fit_async`, `predict`
- Noun phrases for getters: `get_embedding`, `get_reference_corpus`
- Private methods: underscore prefix `_ensure_async_executor`

**Variables:**
- snake_case
- Descriptive: `training_data`, `reference_embeddings`, `confidence_threshold`
- Booleans: `is_novel`, `is_trained`, `use_novelty_detector`

**Constants:**
- UPPER_CASE
- Prefix with underscore for module-private: `_VALID_OOD_STRATEGIES`

## Code Patterns

**Async Support:**
- Async-first design for new features
- Sync fallbacks for compatibility
- Use `asyncio` for concurrent operations
- `@pytest.mark.asyncio` for async tests

**Type Hints:**
- Use type hints on all public APIs
- `from __future__ import annotations` for forward references
- `TYPE_CHECKING` for import-only type hints
- Optional types: `Optional[str]`, `List[str]`, `Dict[str, Any]`

**Pydantic Models:**
- Use Pydantic v2 for configuration
- Field validation with `Field()`
- `field_validator` for custom validation
- Config dict: `ConfigDict`

**Error Handling:**
- Custom exception hierarchy
- Specific exception types: `ValidationError`, `TrainingError`, `MatchingError`, `ModeError`
- Descriptive error messages
- Logging via custom logger

**Logging:**
- Import: `from ..utils.logging_config import get_logger`
- Usage: `logger = get_logger(__name__)`
- Levels: DEBUG, INFO, WARNING, ERROR
- Verbosity controlled by NOVEL_ENTITY_MATCHER_VERBOSE env var

## File Organization

**Import Order:**
1. Standard library
2. Third-party imports
3. Local imports (relative or absolute)
4. Type checking imports (under `if TYPE_CHECKING:`)

**Module Structure:**
1. Module docstring
2. Imports (with __future__ annotations first)
3. Constants
4. Type definitions
5. Classes/functions
6. `__all__` exports

**Test Structure:**
- Test class per module (if needed)
- `test_` prefix for test functions
- Fixtures in `conftest.py` for shared setup
- Markers: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.

## Error Handling Patterns

**Validation:**
- Input validation before processing
- Custom validation functions in `utils/validation.py`
- Pydantic models for config validation
- Raise `ValidationError` for invalid inputs

**Strategy Pattern:**
- Abstract base class: `NoveltyStrategy`
- Concrete implementations: `ConfidenceStrategy`, `ClusteringStrategy`, etc.
- Registry pattern: `StrategyRegistry`

**Adapter Pattern:**
- Base interface: `EmbeddingBackend`
- Implementations: `SentenceTransformerBackend`, `StaticEmbeddingBackend`
- Dynamic resolution via `backends/__init__.py`

## Documentation Patterns

**Docstrings:**
- Google-style docstrings preferred
- Module docstrings explain purpose
- Class docstrings explain role and usage
- Function docstrings explain args/returns

**Examples:**
- Usage examples in `examples/`
- Script examples in `scripts/`
- README with quick start guide

## Configuration Patterns

**Pydantic Config:**
- All config classes inherit from `BaseModel`
- Use `Field()` for constraints
- `field_validator` for custom validation
- Type-safe access to config values

**Model Registry:**
- Centralized registry in `config_registry.py`
- Dynamic model loading
- Aliases and version resolution

**Environment Variables:**
- Prefix: `NOVEL_ENTITY_MATCHER_`
- Logging: `NOVEL_ENTITY_MATCHER_VERBOSE`
- LLM keys: Provider-specific (e.g., `OPENAI_API_KEY`)

## Testing Conventions

**Test Organization:**
- Unit tests: `tests/unit/` (fast, no external deps)
- Integration tests: `tests/integration/` (requires external services)
- Test fixtures: `tests/fixtures/`
- Shared fixtures in `tests/conftest.py`

**Test Markers:**
- `@pytest.mark.unit` - Fast isolated tests
- `@pytest.mark.integration` - Tests with external deps
- `@pytest.mark.slow` - Expensive tests
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.hf` - Hugging Face model tests
- `@pytest.mark.llm` - LLM API tests (requires API key)
- `@pytest.mark.llm_mocked` - LLM tests with mocks
- `@pytest.mark.serial` - Non-parallelizable tests
- `@pytest.mark.network` - Tests requiring internet
- `@pytest.mark.smoke` - Critical path tests

**Test Commands:**
- Fast tests: `pytest -m "not integration and not slow"`
- All tests: `pytest`
- Specific marker: `pytest -m llm`

---

*Conventions analysis: 2026-04-23*

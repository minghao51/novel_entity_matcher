# Coding Conventions

**Analysis Date:** 2026-04-06

## Naming Patterns

**Files:**
- `snake_case.py` for all Python modules
- `test_*.py` for test files, co-located in `tests/` mirroring `src/` structure
- Subdirectories group related tests: `tests/test_core/`, `tests/test_utils/`, `tests/test_backends/`, `tests/test_ingestion/`

**Functions:**
- `snake_case` for all functions and methods
- Private helpers prefixed with `_` (e.g., `_ensure_async_executor`, `_format_message`)
- Descriptive verb-led names: `configure_logging`, `validate_entities`, `build_index`

**Variables:**
- `snake_case` for all variables
- Private attributes prefixed with `_` (e.g., `_async_executor`, `_reference_embeddings`)
- Numpy arrays use descriptive names: `sample_embeddings`, `reference_labels`

**Types:**
- `PascalCase` for classes: `EmbeddingMatcher`, `NoveltyDetector`, `DetectionConfig`
- Private classes prefixed with `_`: `_EntityMatcher`
- Type hints used throughout, with `Optional`, `Union`, `Dict`, `List` from `typing`
- Dataclasses/pydantic models for config: `DetectionConfig`, `WeightConfig`, `ClusteringConfig`

## Code Style

**Formatting:**
- Black >= 23.0.0 (in dev dependencies)
- Ruff >= 0.1.0 (in dev dependencies)
- Python >= 3.9, tested through 3.12

**Linting:**
- Ruff for linting (fast, modern linter)
- MyPy for type checking (python_version = 3.13)
- MyPy configured with gradual typing: `disallow_untyped_defs = false`, `check_untyped_defs = true`
- Per-module overrides for external packages (setfit, transformers, torch, etc.) with `ignore_missing_imports = true`
- Some novelty strategy modules have `ignore_errors = true` due to complex typing

## Import Organization

**Order:**
1. Standard library imports (asyncio, os, json, logging, typing)
2. Third-party imports (numpy, sentence_transformers, pytest)
3. Local imports using relative paths (`..exceptions`, `.embedding_matcher`, `..utils.validation`)

**Path Aliases:**
- No path aliases; uses relative imports within package
- Source lives in `src/novelentitymatcher/`, installed as `novelentitymatcher` package

## Error Handling

**Patterns:**
- Custom exception hierarchy rooted at `SemanticMatcherError(Exception)`
- Domain-specific exceptions with rich context:
  - `ValidationError(ValueError, SemanticMatcherError)` — includes entity, field, suggestion attributes
  - `TrainingError(RuntimeError, SemanticMatcherError)` — includes training_mode, details attributes
  - `MatchingError(RuntimeError, SemanticMatcherError)` — for matching operation failures
  - `ModeError(ValueError, SemanticMatcherError)` — includes invalid_mode, valid_modes attributes
- Exceptions use `_format_message()` to build contextual error strings with field/entity/suggestion info
- Validation happens early: `validate_entities()`, `validate_model_name()`, `validate_threshold()` called in constructors
- `pytest.raises` used for testing error conditions with message matching

## Logging

**Framework:** Standard Python `logging` module with centralized configuration

**Patterns:**
- Centralized config in `novelentitymatcher.utils.logging_config`
- `configure_logging(verbose=False)` called at package import time
- `NOVEL_ENTITY_MATCHER_VERBOSE` env var controls default verbosity
- `get_logger(name)` helper ensures consistent `novelentitymatcher.` namespace
- Third-party loggers suppressed by default (sentence_transformers, transformers, setfit, torch, datasets, etc.)
- Two format modes: simple `%(message)s` (quiet) vs `[%(levelname)s] %(name)s: %(message)s` (verbose)
- Optional file handler support via `log_file` parameter
- Runtime log level changes supported via `set_log_level()`

## Comments

**When to Comment:**
- Docstrings on all public functions/classes with Args/Returns/Example sections
- Inline comments for non-obvious behavior (e.g., CPU fallback for MPS)
- Module-level docstrings describe purpose

**JSDoc/TSDoc:**
- Google-style docstrings with Args, Returns, Example sections
- Triple-quoted module docstrings at file top

## Function Design

**Size:**
- Mix of small focused functions and larger orchestration methods
- Complex logic split into `_`-prefixed private helpers

**Parameters:**
- Keyword-only arguments after `*` for optional config
- Dataclasses/config objects for complex parameter groups (e.g., `DetectionConfig`)
- Sensible defaults throughout

**Return Values:**
- Typed return values with `Optional[T]` when nullable
- Dict returns for complex results (e.g., match results with id, score, metadata)
- Dataclass returns for structured results (e.g., `PipelineRunResult`, `StageResult`)

## Module Design

**Exports:**
- Lazy exports via `__getattr__` in `__init__.py` for fast import
- `__all__` lists all public symbols
- `_EXPORTS` dict maps name to (module, attribute) for lazy loading
- Public API at package root: `Matcher`, `NovelEntityMatcher`, `DiscoveryPipeline`, `NoveltyDetector`

**Barrel Files:**
- `__init__.py` files in subpackages re-export key symbols
- Subpackages: `core/`, `novelty/`, `backends/`, `benchmarks/`, `utils/`, `pipeline/`, `data/`
- Novelty subpackage has deep nesting: `novelty/strategies/`, `novelty/storage/`, `novelty/clustering/`, `novelty/proposal/`, `novelty/evaluation/`, `novelty/config/`, `novelty/schemas/`, `novelty/core/`, `novelty/utils/`

---

*Convention analysis: 2026-04-06*

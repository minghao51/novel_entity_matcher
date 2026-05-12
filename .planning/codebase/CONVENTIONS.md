# Conventions

## Linting & Formatting

**Toolchain:** ruff (lint + format), mypy (type checking), pre-commit hooks

### Ruff Configuration (`pyproject.toml:109-137`)

- **Line length:** 88 characters (E501 ignored — ruff format handles line breaks)
- **Target:** Python 3.10+
- **Quote style:** double quotes
- **Enabled rule sets:**
  - `E`, `F` — pyflakes, pycodestyle errors
  - `I` — isort (import ordering)
  - `UP` — pyupgrade
  - `B` — flake8-bugbear
  - `C4` — flake8-comprehensions
  - `DTZ` — flake8-datetimez (timezone-aware datetimes)
  - `T10` — flake8-debugger (no debugger statements)
  - `ISC` — flake8-implicit-str-concat
  - `PIE` — flake8-pie
  - `PT` — flake8-pytest-style
  - `RUF` — ruff-specific rules
- **Fixable:** all rules auto-fixable
- **Per-file ignores:** `tests/**` ignores `DTZ`; `notebooks/**` ignores all rules

### Pre-commit Hooks (`.pre-commit-config.yaml`)

| Hook | Purpose |
|------|---------|
| trailing-whitespace | Remove trailing spaces |
| end-of-file-fixer | Ensure newline at EOF |
| check-yaml | Validate YAML syntax |
| check-merge-conflict | Detect unresolved conflicts |
| debug-statements | No `pdb`/`breakpoint` left in code |
| check-added-large-files | Max 1000KB for general files, 5000KB for docs artifacts |
| uv-lock | Keep `uv.lock` in sync |
| ruff (lint + format) | Auto-fix lint issues, format code |
| mypy (local) | Type check `src/novelentitymatcher` |
| conventional-pre-commit | Enforce Conventional Commits format on commit messages |
| quarto-render (local) | Render changed `.qmd` notebooks on commit |

### Mypy Configuration (`pyproject.toml:169-247`)

- **Python version:** 3.11
- **Strictness:** gradual adoption — `disallow_untyped_defs=false`, `disallow_any_generics=false`
- **Check untyped defs:** enabled
- **Strict optional:** enabled
- **Search path:** `src`
- **External libraries:** ignore missing imports for ML ecosystem (torch, transformers, sklearn, etc.)
- **Known-complex modules:** `ignore_errors=true` for clustering, storage, strategies, and hybrid modules

## Code Style

### Imports

- **`from __future__ import annotations`** used in most files (72+ source files) for PEP 604 union syntax and forward reference support
- Import order enforced by ruff's isort rules: stdlib → third-party → local
- Conditional imports use `TYPE_CHECKING` guard for type-only imports:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from .matching_strategy import MatchingStrategy
  ```
- Package imports use relative imports within the package (`.`, `..`, `...`)

### Type Hints

- PEP 604 union syntax: `str | None` instead of `Optional[str]`, `dict[str, Any]` instead of `Dict[str, Any]`
- `Any` used for flexible inputs (entity dicts, config values)
- Type aliases defined at module level: `PathLike = Union[str, Path]`, `EmbeddingModel = SentenceTransformer`
- `__all__` exports defined in most public modules for explicit API surface
- Pydantic `BaseModel` used for structured data schemas (`src/novelentitymatcher/novelty/schemas/models.py`)

### Naming

- **Classes:** PascalCase (`EmbeddingMatcher`, `NoveltyDetector`, `DetectionConfig`)
- **Functions/methods:** snake_case (`validate_entities`, `build_index`, `get_logger`)
- **Constants:** UPPER_SNAKE_CASE (`MODEL_SPECS`, `BERT_DEFAULT_MODEL`, `METRIC_MATCH_LATENCY`)
- **Private helpers:** prefixed with underscore (`_coerce_texts`, `_deep_update`, `_BatchEngine`)
- **Internal classes:** prefixed with underscore (`_EntityMatcher`, `_BatchEngine`, `_HybridEngine`)
- **Test helper functions:** prefixed with underscore (`_make_entities`, `_hdbscan_available`)
- **Module-level `__all__`** used consistently for public API declaration

### Docstrings

- Module-level docstrings describe purpose and usage
- Class docstrings describe the class role (not all classes have docstrings)
- Method/function docstrings use Google-style:
  ```
  Args:
      name: Description
  Returns:
      Description
  ```
- Example usage in docstrings using `>>>` prompt style
- `api.py` has a usage example in the module docstring

## Error Handling

### Custom Exception Hierarchy (`src/novelentitymatcher/exceptions.py`)

```
SemanticMatcherError (base)
├── ValidationError (ValueError) — input validation failures
├── TrainingError (RuntimeError) — training failures
├── MatchingError (RuntimeError) — matching operation failures
├── ModeError (ValueError) — invalid mode configuration
└── LLMError — LLM API failures after retries
```

### Exception Design Pattern

All exceptions follow a consistent pattern:
1. Accept a raw `message` plus keyword-only context attributes
2. Store context (`entity`, `field`, `suggestion`, `training_mode`, `invalid_mode`, etc.)
3. Format a rich message via `_format_message()` that appends structured context
4. Call `super().__init__(formatted_message)`

Example:
```python
raise ValidationError(
    "Entity must have 'id' field",
    entity=entity,
    field="id",
    suggestion="Add 'id' field: {'id': 'unique_id', 'name': 'Entity Name'}",
)
```

### API Key Redaction

`_redact_api_keys()` in `exceptions.py` uses regex patterns to redact API keys from error messages before display. Covers OpenRouter, Anthropic, OpenAI, HuggingFace, Google, and Ya29 patterns.

### Validation Pattern (`src/novelentitymatcher/utils/validation.py`)

- Dedicated validation functions: `validate_entities()`, `validate_entity()`, `validate_model_name()`, `validate_threshold()`
- Raise `ValidationError` with context (entity, field, suggestion)
- Called early in constructors (fail-fast)

## Logging

### Centralized Logging (`src/novelentitymatcher/utils/logging_config.py`)

All modules use a unified logging system:

```python
from ..utils.logging_config import get_logger
logger = get_logger(__name__)
```

Key functions:
- **`configure_logging(verbose, log_level, log_file)`** — initialize the `novelentitymatcher` logger namespace
- **`get_logger(name)`** — returns a namespaced logger (auto-prefixes `novelentitymatcher.` if needed)
- **`suppress_third_party_loggers()`** — silences ML library noise (sentence_transformers, transformers, torch, etc.)
- **`set_log_level(level)`** — runtime log level changes

### Log Levels

| Mode | Level | Format |
|------|-------|--------|
| Quiet (default) | WARNING | `%(message)s` |
| Verbose | DEBUG | `[%(levelname)s] %(name)s: %(message)s` |

### Environment Variable

`NOVEL_ENTITY_MATCHER_VERBOSE=true` enables verbose logging at import time (checked in `__init__.py`).

## Configuration

### Config System (`src/novelentitymatcher/config.py`)

- `Config` class loads YAML/JSON config with cascading sources:
  1. Repo root `config.yaml`
  2. Package-bundled `data/default_config.json`
  3. CWD `config.yaml`
  4. Custom path override (deep-merged)
- Attribute access for nested keys: `cfg.training.num_epochs`
- Dot-notation fallback: `cfg.get("training.batch_size")`

### Registry Pattern (`src/novelentitymatcher/config_registry.py`)

- `MODEL_SPECS` dict maps aliases → model metadata (name, backend, supports_training, language)
- `MODEL_REGISTRY`, `STATIC_MODEL_REGISTRY`, `DYNAMIC_MODEL_REGISTRY` for model classification
- `MATCHER_MODE_REGISTRY` for mode resolution
- `RERANKER_REGISTRY` for reranker backends
- Helper functions: `resolve_model_alias()`, `is_bert_model()`, `recommend_model()`

### Strategy Registry (`src/novelentitymatcher/novelty/core/strategies.py`)

- Decorator-based registration: `@StrategyRegistry.register`
- `StrategyRegistry.get(strategy_id)` for lookup
- Prevents duplicate registration

## Structural Patterns

### Pydantic Models

Used for data schemas with validation (`src/novelentitymatcher/novelty/schemas/models.py`):
- `model_config = ConfigDict(arbitrary_types_allowed=True)`
- `Field(default_factory=list)` for mutable defaults
- `Field(ge=0, le=1.0)` for range constraints
- Nested model composition (`DiscoveryCluster` contains `ClusterEvidence`)

### Dataclass Usage

- `@dataclass` for lightweight metric events (`monitoring/metrics.py`)
- No Pydantic overhead for simple internal structures

### Performance Monitoring

- `@track_performance` decorator on key methods (`monitoring/performance.py`)
- `PerformanceMonitor` class for timing and metrics collection
- `metrics_callback` parameter on `Matcher.__init__()` for user-supplied metric handlers
- `MetricEvent` dataclass as standard metric structure

### Public API Surface

- `src/novelentitymatcher/__init__.py` — top-level re-exports
- `src/novelentitymatcher/api.py` — comprehensive `from X import *` surface for power users
- Both define `__all__` explicitly

### Async Patterns

- `asyncio_mode = "auto"` in pytest — all `async def test_*` run as coroutines automatically
- `AsyncExecutor` in `core/async_utils.py` for parallel matching
- Async/sync parity tests ensure consistent behavior

### CLI Entry Points (`pyproject.toml:45-48`)

- `novelentitymatcher-ingest` → `ingestion.cli:main`
- `novelentitymatcher-bench` → `benchmarks.cli:main`
- `novelentitymatcher-review` → `novelty.cli:main`

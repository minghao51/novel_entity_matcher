# Coding Conventions

**Analysis Date:** 2026-04-30

## Code Style

### Formatting

- **Tool:** Ruff (formatter + linter)
- **Config:** `pyproject.toml` `[tool.ruff]` / `[tool.ruff.format]`
- **Line length:** 88 (Black-compatible default)
- **Target version:** Python 3.10+
- **Quote style:** Double quotes (`quote-style = "double"`)
- **E501 ignored:** Long lines allowed; ruff lint ignores `E501`

### Linting

- **Tool:** Ruff (`[tool.ruff.lint]` in `pyproject.toml:178-194`)
- **Selected rule sets:**
  - `E`, `F` — pyflakes + pycodestyle errors
  - `I` — isort (import sorting)
  - `UP` — pyupgrade
  - `B` — flake8-bugbear
  - `C4` — flake8-comprehensions
  - `DTZ` — flake8-datetimez (timezone-aware datetimes)
  - `T10` — flake8-debugger
  - `ISC` — flake8-implicit-str-concat
  - `PIE` — flake8-pie
  - `PT` — flake8-pytest-style
  - `RUF` — Ruff-specific rules
- **Fixable:** All (`fixable = ["ALL"]`)
- **Per-file ignores:**
  - `tests/**` — `DTZ` disabled
  - `notebooks/**` — `ALL` rules disabled

### Type Checking

- **Tool:** mypy (`pyproject.toml:220-296`)
- **Target:** Python 3.11
- **Mode:** Gradual typing — `disallow_untyped_defs = false`, `disallow_any_generics = false`
- **Strictness:** `strict_optional = true`, `check_untyped_defs = true`
- **`mypy_path`:** `src`
- **External stubs:** Many third-party packages use `ignore_missing_imports = true` (setfit, transformers, torch, etc.)
- **Legacy overrides:** Some modules use `ignore_errors = true` (novelty strategies impl files, clustering, storage)

### Pre-commit Hooks

- **Config:** `.pre-commit-config.yaml`
- **Hooks:**
  - trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements
  - check-added-large-files (max 1000 KB)
  - `uv-lock` — keeps lockfile in sync
  - `ruff` (with `--fix`) + `ruff-format`
  - `mypy` — runs `uv run mypy src/novelentitymatcher`
  - `conventional-pre-commit` — enforces Conventional Commits on commit messages

## Naming Patterns

### Files

- **Modules:** `snake_case.py` (e.g., `embedding_matcher.py`, `matcher_runtime.py`)
- **Private/internal modules:** prefixed with underscore (e.g., `matcher_shared.py`, `matcher_entity.py`)
- **Package data:** JSON files in `src/novelentitymatcher/data/` (e.g., `country_codes.json`, `default_config.json`)

### Functions

- `snake_case` throughout (e.g., `validate_entities`, `resolve_model_alias`, `get_cached_sentence_transformer`)
- **Private helpers:** prefixed with underscore (e.g., `_load_file`, `_deep_update`, `_format_message`)
- **Boolean-returning:** use `is_`/`has_` prefix (e.g., `is_bert_model`, `is_static_embedding_model`)

### Variables

- `snake_case` throughout
- Constants: `UPPER_SNAKE_CASE` at module level (e.g., `MODEL_SPECS`, `RERANKER_REGISTRY`, `METRIC_MATCH_LATENCY`)

### Types / Classes

- `PascalCase` for classes (e.g., `Matcher`, `NoveltyDetector`, `ModelCache`, `MetricEvent`)
- Pydantic models: `PascalCase` with `BaseModel` inheritance (e.g., `DetectionConfig`, `NovelSampleMetadata`)
- Dataclasses: `@dataclass` decorator (e.g., `StageContext`, `StageResult`, `PipelineRunResult`)
- Type aliases: `PascalCase` (e.g., `PathLike = Union[str, Path]`)

## Import Conventions

### Order (enforced by Ruff `I` rules)

1. **Future imports:** `from __future__ import annotations` — used in ~55 source files
2. **Standard library:** `import os`, `import re`, `import threading`, etc.
3. **Third-party:** `import numpy as np`, `from pydantic import BaseModel`, `from sentence_transformers import SentenceTransformer`
4. **Local/package:** `from ..config import ...`, `from ..exceptions import ...`, `from .embedding_matcher import ...`

### Path Aliases

- No path aliases configured — all imports use full package-relative paths
- Relative imports use leading dots: `from ..utils.logging_config import get_logger`
- `TYPE_CHECKING` guard used for type-only imports to avoid circular dependencies:
  ```python
  from typing import TYPE_CHECKING
  if TYPE_CHECKING:
      from .matching_strategy import MatchingStrategy
  ```

### Common Import Patterns

- **Lazy imports in `__init__.py`:** Uses `__getattr__` pattern for deferred module loading (`src/novelentitymatcher/__init__.py:97-104`)
- **Heavy dependencies deferred:** `sentence_transformers.CrossEncoder`, `setfit.SetFitModel` imported inside functions (e.g., `src/novelentitymatcher/utils/embeddings.py:169,197`)
- **Optional dependencies guarded:** `try/except ImportError` for NLTK, model2vec, etc. (`src/novelentitymatcher/utils/preprocessing.py:5-14`)

## Error Handling

### Exception Hierarchy

- **Base:** `SemanticMatcherError(Exception)` — `src/novelentitymatcher/exceptions.py:8`
- **Domain exceptions** (all inherit from base + a stdlib type):
  - `ValidationError(ValueError, SemanticMatcherError)` — input validation failures
  - `TrainingError(RuntimeError, SemanticMatcherError)` — training failures
  - `MatchingError(RuntimeError, SemanticMatcherError)` — matching failures
  - `ModeError(ValueError, SemanticMatcherError)` — invalid mode configuration
  - `LLMError(SemanticMatcherError)` — LLM API failures

### Error Design Patterns

- **Rich context:** Each exception carries domain-specific attributes (e.g., `entity`, `field`, `suggestion` for `ValidationError`)
- **Self-formatting:** Exceptions override `_format_message()` to produce human-readable output with context
- **API key redaction:** `LLMError` uses `_redact_api_keys()` to strip sensitive keys from error messages (`src/novelentitymatcher/exceptions.py:136-137`)
- **Validation module:** Centralized `validate_*` functions in `src/novelentitymatcher/utils/validation.py` that raise `ValidationError` with suggestions

### Error Handling in Config

- `_safe_load_file()` wraps file loading in broad exception catch (`src/novelentitymatcher/config.py:123-127`)
- Package init uses `try/except` for logging setup and version detection (`src/novelentitymatcher/__init__.py:21-33`)

## Logging

### Framework

- **Standard library `logging`** — centralized in `src/novelentitymatcher/utils/logging_config.py`
- **Logger factory:** `get_logger(__name__)` — used in 30+ modules

### Configuration

- **Verbose mode:** Controlled by `NOVEL_ENTITY_MATCHER_VERBOSE` env var or `verbose=True` parameter
- **Default level:** `WARNING` (quiet mode)
- **Verbose level:** `DEBUG` with format `[%(levelname)s] %(name)s: %(message)s`
- **Third-party suppression:** `suppress_third_party_loggers()` sets ML libraries to WARNING
- **Optional file logging:** `configure_logging(log_file="path")`

### Usage Patterns

```python
from ..utils.logging_config import get_logger
logger = get_logger(__name__)
logger.info("Starting operation")
logger.debug("Detailed information")
```

- Module-level logger: `logger = get_logger(__name__)` at top of file
- Instance-level logger: `self.logger = get_logger(__name__)` in `__init__` for classes like `Matcher`, `BERTClassifier`

## Shared Utilities

### `src/novelentitymatcher/utils/`

| File | Purpose |
|------|---------|
| `embeddings.py` | `ModelCache` (thread-safe LRU), `get_cached_sentence_transformer()`, `compute_embeddings()`, `cosine_sim()`, `batch_encode()` |
| `validation.py` | `validate_entities()`, `validate_entity()`, `validate_threshold()`, `validate_model_name()` |
| `preprocessing.py` | `clean_text()`, `tokenize()`, `lemmatize()`, `remove_stopwords()`, `extract_aliases()` |
| `logging_config.py` | `configure_logging()`, `get_logger()`, `set_log_level()`, `suppress_third_party_loggers()` |
| `benchmarks.py` | Benchmarking utilities |
| `benchmark_dataset.py` | Benchmark dataset loading |
| `benchmark_reporting.py` | Benchmark report generation |
| `learning_curves.py` | Learning curve utilities |

### `src/novelentitymatcher/exceptions.py`

All custom exceptions in one file with helpful context, suggestions, and API key redaction.

### `src/novelentitymatcher/config_registry.py`

Centralized model/mode registries: `MODEL_SPECS`, `MODEL_REGISTRY`, `STATIC_MODEL_REGISTRY`, `RERANKER_REGISTRY`, `MATCHER_MODE_REGISTRY`, `NOVEL_DETECTION_CONFIG`, `LLM_PROVIDERS`, plus resolver functions.

### `src/novelentitymatcher/monitoring/metrics.py`

`MetricEvent` dataclass + `create_metric()` helper + standard metric name constants (`METRIC_*`, `LABEL_*`).

## Configuration Patterns

### Config Loading

- **Main config class:** `Config` in `src/novelentitymatcher/config.py` — attribute-access via `__getattr__`
- **Sources searched in order:**
  1. Repo root `config.yaml` (walks up from `__file__`)
  2. Package-bundled `data/default_config.json`
  3. CWD `config.yaml`
- **Custom overrides:** `Config(custom_path="...")` deep-merges on top of defaults
- **Dot-notation access:** `cfg.get("training.batch_size")` returns nested values

### Pydantic Configs

- Novelty detection configs use `pydantic.BaseModel` with `ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)`
- Examples: `DetectionConfig` (`src/novelentitymatcher/novelty/config/base.py`), `PipelineConfig` (`src/novelentitymatcher/pipeline/config.py`)
- Per-strategy configs: `ConfidenceConfig`, `KNNConfig`, `ClusteringConfig`, etc. (`src/novelentitymatcher/novelty/config/strategies.py`)

### Environment Variables

- `NOVEL_ENTITY_MATCHER_VERBOSE` — enables verbose/debug logging
- `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — LLM provider keys (referenced in `config_registry.py:LLM_PROVIDERS`)
- `PYTORCH_ENABLE_MPS_FALLBACK` — set for Apple Silicon compatibility

## Module Design

### Exports

- Every module defines `__all__` explicitly (30+ modules)
- Package `__init__.py` uses lazy loading via `__getattr__` to avoid importing heavy dependencies at package import time

### ABC / Interface Pattern

- Abstract backends: `EmbeddingBackend`, `RerankerBackend` in `src/novelentitymatcher/backends/base.py`
- Pipeline stages: `PipelineStage(ABC)` in `src/novelentitymatcher/pipeline/contracts.py`
- Novelty strategies: `NoveltyStrategy(ABC)` with registry pattern

### Registry Pattern

- `StrategyRegistry` for novelty strategies (`src/novelentitymatcher/novelty/core/strategies.py`)
- `ClusteringBackendRegistry` for clustering backends
- Model registries in `config_registry.py` (alias → full name mapping)

---

*Convention analysis: 2026-04-30*

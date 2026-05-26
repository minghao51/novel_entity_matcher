# novel-entity-matcher — Code Style & Conventions

## File Organization

```
src/novelentitymatcher/
  __init__.py              # Lazy exports, package init
  api.py                   # Single-import public surface (re-exports all)
  config.py                # Config loader (YAML/JSON cascading)
  exceptions.py            # Custom exception hierarchy
  backends/                # Embedding/reranker provider ABCs + impls
    base.py                #   EmbeddingBackend, RerankerBackend (ABCs)
    litellm.py             #   LiteLLM (multi-provider LLM)
    sentencetransformer.py #   SentenceTransformer backend
    static_embedding.py    #   model2vec static embeddings
    reranker_st.py         #   SentenceTransformer reranker
  benchmarks/              # Dataset registry, runners, CLI, viz
    cli.py                 #   novelentitymatcher-bench CLI (subcommand-based)
    runner.py              #   BenchmarkRunner orchestrator
    registry.py            #   DATASET_REGISTRY
    loader.py              #   DatasetLoader (HF datasets)
    novelty_bench.py       #   Novelty-specific benchmarks
    weight_optimizer.py    #   Optuna-based ensemble weight tuning
    visualization.py       #   Plotting & chart generation
    async_bench.py         #   Sync vs async perf comparison
    classification/        #   Classification task benchmarks
    entity_resolution/     #   ER task benchmarks
    novelty/               #   Novelty task benchmarks
  core/                    # Entity matching engine
    matcher.py             #   Matcher facade (public entry point)
    matching_strategy.py   #   Strategy pattern (ZeroShot, Bert, Hybrid, etc.)
    embedding_matcher.py   #   Zero-shot embedding-based matching
    classifier.py          #   SetFit classifier
    bert_classifier.py     #   BERT fine-tuning classifier
    hybrid.py              #   Hybrid blocking + retrieval
    blocking.py            #   Blocking strategies (BM25, TF-IDF, Fuzzy)
    reranker.py            #   Cross-encoder reranking
    normalizer.py          #   Text normalization
    hierarchy.py           #   Hierarchical entity scoring
    vector_store.py        #   In-memory vector store (numpy)
    matcher_shared.py      #   Shared types: TextInput, coerce_texts
    matcher_entity.py      #   _EntityMatcher (trained mode impl)
    async_utils.py         #   AsyncExecutor
  data/                    # Package-bundled static data
    country_codes.json
  ingestion/               # External dataset ingestion (7 domains)
    cli.py                 #   novelentitymatcher-ingest CLI
    base.py                #   Base fetcher, concurrent runner
    currencies.py          #   Currency domain ingester
    industries.py          #   Industry domain ingester
    languages.py           #   Language domain ingester
    occupations.py         #   Occupation domain ingester
    products.py            #   Product domain ingester
    timezones.py           #   Timezone domain ingester
    universities.py        #   University domain ingester
  monitoring/              # Metrics & performance tracking
    metrics.py             #   MetricEvent dataclass, create_metric()
    performance.py         #   @track_performance decorator
  novelty/                 # Novelty detection & discovery subsystem
    cli.py                 #   novelentitymatcher-review CLI (HITL review)
    discovery_base.py      #   DiscoveryResult dataclass
    entity_matcher.py      #   NovelEntityMatcher facade
    active_learning/       #   Annotation collector, uncertainty sampler
    clustering/            #   HDBSCAN, UMAP, Leiden, validation
    config/                #   Pydantic config models (DetectionConfig, per-strategy)
    core/                  #   NoveltyDetector, StrategyRegistry, SignalCombiner
    drift/                 #   Drift detection
    evaluation/            #   OOD evaluation metrics
    extraction/            #   Entity extraction from novel samples
    proposal/              #   LLM-based class naming (LiteLLM, DSPy)
    schemas/               #   Pydantic data models (NovelSampleMetadata, ClassProposal, etc.)
    storage/               #   ANN index, review state management
    strategies/            #   15+ OOD detection algorithms (confidence, knn, lof, svm, mahalanobis, setfit, prototypical, self-knowledge, etc.)
    utils/                 #   Novelty-specific utilities
  pipeline/                # Staged discovery pipeline
    contracts.py           #   PipelineStage ABC, StageContext, StageResult
    orchestrator.py        #   Sequential stage runner
    pipeline_builder.py    #   7-stage pipeline construction
    adapters.py            #   Stage implementations (5 core)
    config.py              #   PipelineConfig Pydantic model
    discovery.py           #   DiscoveryPipeline facade
    match_result.py        #   Match result builders
    discovery_support.py   #   Pipeline support utilities
    stages/                #   Optional stages (drift_hook, stability_filter)
  utils/                   # Cross-cutting utilities
    benchmarks.py          #   Benchmark helper functions
    embeddings.py          #   Embedding cache, model loading
    logging_config.py      #   configure_logging(), get_logger()
    preprocessing.py       #   NLP preprocessing
    validation.py          #   Entity/model/threshold validation

tests/
  conftest.py              # Shared fixtures (model cache clearing, marker auto-assign)
  fixtures/                # Test data JSON files
  unit/                    # Fast isolated tests, mirrors src/ layout
    backends/
    benchmarks/
    core/
    ingestion/
    monitoring/
    novelty/
    pipeline/
    utils/
    test_config.py
    test_*.py              # Top-level unit tests
  integration/             # Multi-module/external-service tests
    backends/
    core/
    utils/
    test_*.py              # Top-level integration tests
```

## Naming Conventions

### Python

| Element | Convention | Examples |
|---------|-----------|----------|
| Files/dirs | `snake_case` | `embedding_matcher.py`, `matcher_shared.py`, `core/`, `backends/` |
| Classes | `PascalCase` | `Matcher`, `EmbeddingMatcher`, `NoveltyDetector`, `ClusterEvidence` |
| Private classes | `_PascalCase` | `_EntityMatcher`, `_BatchEngine`, `_HybridEngine`, `_DiagnosisEngine` |
| ABCs | `PascalCase` | `EmbeddingBackend`, `MatchingStrategy`, `PipelineStage` |
| Config models | `PascalCase` + `Config` | `DetectionConfig`, `KNNConfig`, `PipelineConfig`, `ConfidenceConfig` |
| Functions/methods | `snake_case` | `validate_entities()`, `build_index()`, `resolve_model_alias()` |
| Private methods | `_snake_case` | `_deep_update()`, `_format_message()`, `_coerce_training_mode()` |
| Constants | `UPPER_SNAKE_CASE` | `MODEL_SPECS`, `RERANKER_REGISTRY`, `BERT_DEFAULT_MODEL` |
| Type aliases | `PascalCase` | `PathLike`, `EmbeddingModel`, `ReviewState` |
| Module `__all__` | list of str | Used in `config.py`, `api.py`, `__init__.py` |
| Test files | `test_<module>.py` | `test_config.py`, `test_embedding_matcher.py` |
| Test classes | `TestPascalCase` | rarely used; test files prefer function-style |
| Test functions | `test_<scenario>` | `test_config_loads_default_and_nested_access` |
| Test helper functions | `_snake_case` | `_make_entities()`, `_hdbscan_available()` |
| Package dirs | `flat_snake_case` | `novelentitymatcher` (no underscores), `backends/`, `benchmarks/` |
| Implementation modules | `<name>_impl.py` | `setfit_impl.py`, `pattern_impl.py`, `prototypical_impl.py` |

## Code Patterns

### Pydantic Models
Structured data schemas with validation (in `novelty/schemas/models.py` and `novelty/config/strategies.py`):
```python
class NovelSampleMetadata(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    text: str
    confidence: float
    cluster_id: int | None = None
    signals: dict[str, bool] = Field(default_factory=dict)
```
Key rules:
- `Field(default_factory=...)` for mutable defaults (never `[]` or `{}`)
- `Field(ge=0.0, le=1.0)` for numeric range constraints
- `model_config = ConfigDict(arbitrary_types_allowed=True)` when embedding numpy arrays or torch tensors
- Config models use pure `Field()` with doc comments (see `novelty/config/strategies.py`)

### Dataclasses for Internal Structures
Lightweight data containers without Pydantic overhead (in `pipeline/contracts.py` and `monitoring/metrics.py`):
```python
@dataclass
class StageContext:
    inputs: list[str]
    artifacts: dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricEvent:
    name: str
    value: float
    unit: str
```

### Custom Exception Hierarchy (src/novelentitymatcher/exceptions.py)
```python
class ValidationError(ValueError, SemanticMatcherError):
    def __init__(self, message: str, *, entity=None, field=None, suggestion=None):
        ...
```
Pattern: inherit from both `SemanticMatcherError` and a stdlib type (`ValueError`, `RuntimeError`). Use keyword-only context args. Store raw message, format rich message in `_format_message()`.

### Strategy Pattern (core/matching_strategy.py)
```python
class MatchingStrategy(ABC):
    @abstractmethod
    def match(self, texts, top_k=1, threshold_override=None, **kwargs): ...
    @abstractmethod
    async def match_async(self, texts, top_k=1, threshold_override=None, **kwargs): ...

class ZeroShotStrategy(MatchingStrategy): ...
class HeadOnlyFullStrategy(MatchingStrategy): ...

_STRATEGY_MAP = {"zero-shot": ZeroShotStrategy, "head-only": HeadOnlyFullStrategy, ...}
def get_strategy(mode: str) -> type[MatchingStrategy]: ...
```
A `MatcherFacade` class provides access to all matcher components. The `Matcher` facade delegates to `get_strategy()` via facade.

### CLI with argparse (not click)
All three CLIs use `argparse` with:
1. A `main(argv=None)` entry point
2. Docstring module header
3. Subcommand dispatch via `subparsers` (benchmarks CLI has 10+ subcommands; novelty CLI has `list/show/approve/reject/promote/stats`)
4. Dispatch dict or `if/elif` chain

```python
# ingestion/cli.py
def main(argv=None):
    parser = argparse.ArgumentParser(description="Ingest external datasets")
    parser.add_argument("dataset", nargs="?", default="all", choices=list(INGESTORS.keys()))
    args = parser.parse_args(argv)
    ...

# benchmarks/cli.py — subcommand pattern
def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="novelentitymatcher-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    add_run_parser(subparsers)
    add_load_parser(subparsers)
    ...
    return parser

def main(argv=None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    if args.command == "run": run_benchmarks(...)
    ...
```

### Lazy Imports & TYPE_CHECKING Guards
```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .matching_strategy import MatchingStrategy
```
Type-only imports guarded to avoid circular deps. Heavy ML imports (`torch`, `transformers`, etc.) are local/deferred inside methods.

### Config with Cascading YAML/JSON
`Config` class in `config.py`: searches repo root, CWD, custom path. YAML primarily, with JSON fallback. Deep-merges custom overrides. Attribute/dot-notation access.

### Matcher Facade with Engine Delegation
`Matcher` (`core/matcher.py`) delegates to internal engines (`_HybridEngine`, `_BatchEngine`, `_DiagnosisEngine`) instantiated at `__init__`. Component construction done via `MatcherComponentFactory` with lazy property accessors.

### Async/Sync Parity
Every public method has a sync (`match`) and async (`match_async`) variant. `AsyncExecutor` wraps thread-pool execution. Tests verify parity (`tests/integration/test_async_sync_parity.py`).

### Registry Pattern (dict-based)
```python
MODEL_SPECS = {"potion-8m": {"name": "minishlab/potion-base-8M", "backend": "static", ...}}
MATCHER_MODE_REGISTRY = {"zero-shot": "EmbeddingMatcher", ...}
INGESTORS = {"languages": run_languages, "all": None, ...}
```

## Testing

### Python (pytest)

- **Runner**: `uv run pytest`
- **Coverage**: `uv run pytest --cov=src --cov-report=term-missing`
- **File naming**: `test_<module>.py` mirrors source module name (`test_config.py` → `config.py`, `test_embedding_matcher.py` → `embedding_matcher.py`)
- **Test organization**: Function-style (not class-based), with subdirectory mirror of `src/` layout (`tests/unit/core/`, `tests/unit/novelty/`)
- **Async**: `asyncio_mode = "auto"` — `async def test_*` run as coroutines automatically
- **Fixtures** (`conftest.py`): Auto-use model cache clearing before each test. Auto-assigns `unit`/`integration` markers based on path (`/unit/` → `@pytest.mark.unit`, `/integration/` → `@pytest.mark.integration` + `@pytest.mark.slow`)
- **Markers** (from `pyproject.toml`):

| Marker | Purpose | Used In |
|--------|---------|---------|
| `unit` | Fast isolated, no external deps | Auto-assigned from path |
| `integration` | External services/network | Auto-assigned from `/integration/` path |
| `slow` | Expensive to run | Auto-assigned alongside `integration` |
| `smoke` | Critical path sanity checks | `test_smoke_paths.py`, `test_smoke_static_embedding.py` |
| `hf` | HuggingFace model-backed | `tests/integration/backends/test_static_embedding.py` |
| `llm` | Real LLM API calls (needs keys) | Explicit marker on test functions |
| `llm_mocked` | LLM logic with mocks | Explicit marker |
| `serial` | Cannot run in parallel | Explicit marker |
| `network` | Requires internet | Explicit marker |
| `e2e` | Multi-component end-to-end | Explicit marker (tests/e2e/ is empty) |

- **CI selection**: `--strict-markers -ra --durations=10 --import-mode=importlib --maxfail=10`

## Linting & Formatting

### Python
- **Tool**: ruff (both lint and format)
- **Config** (in `pyproject.toml`):
  - `line-length = 88`, `target-version = "py310"`, `src = ["src"]`
  - **Lint rules**: `E`, `F`, `I`, `UP`, `B`, `C4`, `DTZ`, `T10`, `ISC`, `PIE`, `PT`, `RUF`
  - **Ignored**: `E501` (line length — handled by formatter)
  - **Fixable**: ALL
  - **Quote style**: double quotes
  - **Per-file**: `tests/**` ignores `DTZ`; `notebooks/**` ignores all rules
- **Type checking**: mypy, `python_version = "3.11"`, `strict_optional = true`, `check_untyped_defs = true`, `mypy_path = "src"`
- **Pre-commit**: trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements, check-added-large-files, uv-lock, ruff (lint+format), mypy, conventional-pre-commit, quarto-render

## Build/Dev Commands

```
uv sync                          → Install all dependencies
uv run pytest                    → Run test suite
uv run pytest --cov=src          → Run tests with coverage report
uv run pytest -m "not slow"      → Skip slow/integration tests
uv run pytest -m "unit"          → Run only unit tests
uv run pytest -m "smoke"         → Run smoke tests only
uv run ruff check                → Lint all source files
uv run ruff check --fix          → Auto-fix lint issues
uv run ruff format               → Format all files
uv run mypy src                  → Type-check the source package
uv run pre-commit run --all-files → Run all pre-commit hooks
uv run python -m build           → Build sdist + wheel
make notebooks                   → Render Quarto notebooks
make docs                        → Render notebooks + build MkDocs site
```

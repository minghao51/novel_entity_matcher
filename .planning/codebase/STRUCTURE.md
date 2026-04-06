# Codebase Structure

**Analysis Date:** 2026-04-06

## Directory Layout

```
novel_entity_matcher/
├── src/novelentitymatcher/       # Main package source
│   ├── __init__.py               # Public API (lazy exports)
│   ├── api.py                    # Single import surface (from . import *)
│   ├── config.py                 # Config loader with YAML/JSON support
│   ├── config_registry.py        # Model registries and resolution
│   ├── exceptions.py             # Custom exception hierarchy
│   ├── core/                     # Core matching engine
│   ├── novelty/                  # Novelty detection & discovery
│   ├── pipeline/                 # Pipeline orchestration
│   ├── backends/                 # Embedding/model provider backends
│   ├── ingestion/                # Reference data ingestion CLIs
│   ├── benchmarks/               # Benchmark framework
│   ├── utils/                    # Shared utilities
│   └── data/                     # Static data files
├── tests/                        # Test suite
├── scripts/                      # Utility scripts
├── examples/                     # Usage examples
├── experiments/                  # Experimental code
├── proposals/                    # Novelty proposal storage
├── checkpoints/                  # Model checkpoints
├── artifacts/                    # Build/output artifacts
├── dist/                         # Distribution packages
├── docs/                         # Documentation
├── config.yaml                   # Default configuration
├── pyproject.toml                # Project metadata, deps, tool config
└── uv.lock                       # Locked dependencies (uv)
```

## Directory Purposes

**src/novelentitymatcher/core/:**
- Purpose: Core entity matching algorithms and components
- Contains: Matcher, classifiers (SetFit, BERT), embedding matcher, reranker, normalizer, blocking strategies, hierarchical matching, async utilities, monitoring
- Key files: `matcher.py` (main matcher), `classifier.py` (SetFit classifier), `embedding_matcher.py`, `reranker.py`, `blocking.py`, `hierarchy.py`

**src/novelentitymatcher/novelty/:**
- Purpose: Novelty detection, class proposal, and discovery workflows
- Contains: Detector, strategies, clustering, LLM proposer, schemas, configs, storage, evaluation, CLI
- Key files: `entity_matcher.py` (NovelEntityMatcher), `core/detector.py` (NoveltyDetector), `cli.py`
- Subdirectories:
  - `core/`: Detector, strategy registry, signal combiner, metadata builder, adaptive weights
  - `strategies/`: 10+ novelty detection strategy implementations
  - `clustering/`: Clustering backends (HDBSCAN, SOPTICS, UMAP+HDBSCAN), scalable clustering, validation
  - `config/`: Detection config, strategy configs, weight configs
  - `schemas/`: Pydantic data models, reports, results
  - `storage/`: Persistence, review management, ANN index
  - `proposal/`: LLM class proposer, retrieval-augmented proposer
  - `evaluation/`: Data splitters (OOD, gradual novelty), evaluator

**src/novelentitymatcher/pipeline/:**
- Purpose: Pipeline-first discovery orchestration
- Contains: DiscoveryPipeline, orchestrator, stage adapters, contracts, config, match result types
- Key files: `discovery.py` (DiscoveryPipeline), `orchestrator.py`, `adapters.py`, `contracts.py`

**src/novelentitymatcher/backends/:**
- Purpose: Abstract embedding and model provider backends
- Contains: Base backend, SentenceTransformer, LiteLLM, reranker, static embedding backends
- Key files: `base.py`, `sentencetransformer.py`, `litellm.py`

**src/novelentitymatcher/ingestion/:**
- Purpose: CLI tools for ingesting reference entity datasets
- Contains: Base ingestion class, domain-specific ingestors (currencies, industries, languages, occupations, products, timezones, universities)
- Key files: `base.py`, `cli.py`

**src/novelentitymatcher/benchmarks/:**
- Purpose: Benchmarking framework for matching and novelty detection
- Contains: Base benchmark, runner, registry, loader, domain-specific benchmarks (classification, entity_resolution, novelty)
- Key files: `runner.py`, `registry.py`, `cli.py`

**src/novelentitymatcher/utils/:**
- Purpose: Shared utilities across the package
- Contains: Logging config, validation, preprocessing, embedding helpers, benchmark dataset/reporting helpers, learning curve analysis
- Key files: `logging_config.py`, `validation.py`, `preprocessing.py`

**src/novelentitymatcher/data/:**
- Purpose: Static data files shipped with the package
- Contains: Country codes JSON, default config JSON

## Key File Locations

**Entry Points:**
- `src/novelentitymatcher/__init__.py`: Package initialization, lazy API exports, logging setup
- `src/novelentitymatcher/api.py`: Single import surface (`from novelentitymatcher.api import *`)
- `src/novelentitymatcher/ingestion/cli.py`: CLI entry point (`novelentitymatcher-ingest`)
- `src/novelentitymatcher/benchmarks/cli.py`: CLI entry point (`novelentitymatcher-bench`)
- `src/novelentitymatcher/novelty/cli.py`: CLI entry point (`novelentitymatcher-review`)

**Configuration:**
- `config.yaml`: Default project configuration (model, training, embedding thresholds)
- `src/novelentitymatcher/config.py`: Config loader with YAML/JSON support and deep merge
- `src/novelentitymatcher/config_registry.py`: Model registries, alias resolution, mode resolution
- `src/novelentitymatcher/novelty/config/base.py`: DetectionConfig base class
- `src/novelentitymatcher/novelty/config/strategies.py`: Per-strategy config dataclasses
- `src/novelentitymatcher/novelty/config/weights.py`: WeightConfig for signal combination
- `src/novelentitymatcher/data/default_config.json`: Package-shipped default config

**Core Logic:**
- `src/novelentitymatcher/core/matcher.py`: Main Matcher class (1200+ lines)
- `src/novelentitymatcher/novelty/entity_matcher.py`: NovelEntityMatcher orchestration
- `src/novelentitymatcher/novelty/core/detector.py`: NoveltyDetector with strategy orchestration
- `src/novelentitymatcher/pipeline/discovery.py`: DiscoveryPipeline public API
- `src/novelentitymatcher/pipeline/orchestrator.py`: PipelineOrchestrator (sequential stage execution)

**Testing:**
- `tests/`: All test files, organized by component
- `tests/conftest.py`: Pytest fixtures and configuration
- `tests/fixtures/`: Test fixture data
- `tests/core/`: Core matcher tests
- `tests/test_novelty/`: Novelty detection tests
- `tests/test_ingestion/`: Ingestion tests
- `tests/test_backends/`: Backend tests
- `tests/test_utils/`: Utility tests

## Naming Conventions

**Files:**
- Snake case for all Python files: `matcher.py`, `signal_combiner.py`, `detection_config.py`
- Test files prefixed with `test_`: `test_novel_entity_matcher.py`, `test_pipeline_orchestrator.py`
- Implementation files suffixed with `_impl`: `setfit_impl.py`, `prototypical_impl.py` (legacy complex implementations)
- Base/abstract classes in `base.py` files

**Directories:**
- Lowercase, no separators: `core/`, `novelty/`, `pipeline/`, `backends/`, `utils/`
- Plural for collections of similar items: `strategies/`, `schemas/`, `benchmarks/`

**Classes:**
- PascalCase: `Matcher`, `NovelEntityMatcher`, `DiscoveryPipeline`, `NoveltyDetector`
- Abstract bases prefixed or suffixed with context: `BlockingStrategy`, `NoveltyStrategy`, `ClusteringBackend`

**Modules:**
- Public API classes exported via `__all__` in `__init__.py`
- Lazy loading via `__getattr__` for import performance

## Where to Add New Code

**New Feature:**
- Primary code: `src/novelentitymatcher/core/` (matching features) or `src/novelentitymatcher/novelty/` (novelty features)
- Tests: `tests/` with matching structure (e.g., `tests/core/`, `tests/test_novelty/`)
- Export: Add to `_EXPORTS` dict in `src/novelentitymatcher/__init__.py` and `__all__` in `api.py`

**New Matching Mode/Classifier:**
- Implementation: `src/novelentitymatcher/core/` (e.g., new `*.py` file)
- Register model: `src/novelentitymatcher/config_registry.py`
- Tests: `tests/core/`

**New Novelty Strategy:**
- Implementation: `src/novelentitymatcher/novelty/strategies/` (new `*.py` file)
- Config: `src/novelentitymatcher/novelty/config/strategies.py` (add config dataclass)
- Register: `src/novelentitymatcher/novelty/core/strategies.py` (StrategyRegistry)
- Tests: `tests/test_novelty/` or `tests/test_*_strategy.py`

**New Clustering Backend:**
- Implementation: `src/novelentitymatcher/novelty/clustering/backends.py`
- Tests: `tests/test_novelty/`

**New Benchmark:**
- Implementation: `src/novelentitymatcher/benchmarks/` (new module under domain directory)
- Register: `src/novelentitymatcher/benchmarks/registry.py`
- Tests: `tests/`

**Utilities:**
- Shared helpers: `src/novelentitymatcher/utils/`
- Logging config: `src/novelentitymatcher/utils/logging_config.py`
- Validation: `src/novelentitymatcher/utils/validation.py`

## Special Directories

**proposals/:**
- Purpose: Storage for novelty class proposal review records
- Generated: Yes (by NovelEntityMatcher/DiscoveryPipeline)
- Committed: No (user data)

**checkpoints/:**
- Purpose: Saved model checkpoints (SetFit, BERT, etc.)
- Generated: Yes (by Matcher training)
- Committed: No (binary artifacts)

**artifacts/:**
- Purpose: Build and output artifacts
- Generated: Yes
- Committed: No

**dist/:**
- Purpose: Distribution packages (wheels, source distributions)
- Generated: Yes (by `python -m build`)
- Committed: No

**.tmp/:**
- Purpose: Temporary files during processing
- Generated: Yes
- Committed: No

---

*Structure analysis: 2026-04-06*

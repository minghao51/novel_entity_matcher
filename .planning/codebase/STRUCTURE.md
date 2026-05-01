# Structure

## Directory Layout

```
novel_entity_matcher/
├── .claude/                          # Claude Code config
├── .github/                          # GitHub Actions / CI
├── .opencode/                        # OpenCode skills/config
├── .planning/                        # Planning docs
│   └── codebase/                     # Codemap output (this file)
├── .superset/                        # Superset config
├── .tmp/                             # Temporary training artifacts
├── artifacts/                        # Benchmark artifacts
│   ├── bench-smoke/                  # Smoke test artifacts
│   └── benchmarks/                   # Full benchmark artifacts
├── benchmark_results/                # Benchmark result files
├── checkpoints/                      # Model checkpoints (SetFit fine-tuned)
├── data/                             # Data directories
│   ├── hf_benchmarks/                # HuggingFace benchmark datasets (cached)
│   │   ├── ag_news/
│   │   ├── ag_news_novelty/
│   │   ├── amazon_google/
│   │   ├── beer/
│   │   ├── dblp_acm/
│   │   ├── dblp_googlescholar/
│   │   ├── fodors_zagats/
│   │   ├── goemotions/
│   │   ├── goemotions_novelty/
│   │   ├── itunes_amazon/
│   │   ├── walmart_amazon/
│   │   └── yahoo_answers/
│   ├── processed/                    # Ingested & processed entity data
│   │   ├── currencies/
│   │   ├── industries/
│   │   ├── languages/
│   │   ├── occupations/
│   │   ├── products/
│   │   ├── timezones/
│   │   └── universities/
│   └── raw/                          # Raw downloaded data
│       ├── currencies/
│       ├── industries/
│       ├── languages/
│       ├── occupations/
│       ├── products/
│       ├── timezones/
│       └── universities/
├── dist/                             # Build output (sdist/wheel)
├── docs/                             # Documentation
│   ├── api/                          # API docs
│   ├── architecture/                 # Architecture docs
│   ├── archive/                      # Archived docs
│   │   └── implementation/
│   ├── assets/                       # Doc assets
│   ├── experiments/                  # Experiment docs
│   ├── images/                       # Images (including benchmarks)
│   ├── methodology/                  # Methodology docs
│   │   ├── classification/
│   │   └── novelty-detection/
│   ├── notebooks/                    # Notebook docs
│   └── superpowers/                  # Planning docs
│       └── plans/
├── examples/                         # Usage examples
│   ├── current/                      # Current examples
│   ├── legacy/                       # Legacy examples
│   └── raw/                          # Raw example data
├── experiments/                      # Experiment scripts/output
├── notebooks/                        # Jupyter/marimo notebooks
├── proposals/                        # LLM-generated class proposals (output)
├── scripts/                          # Shell scripts
│   └── setup_llm.sh                  # LLM API key setup
├── src/                              # Source code root
│   └── novelentitymatcher/           # Main package
│       ├── __init__.py               # Lazy exports, version, logging setup
│       ├── api.py                    # Complete public API surface (re-exports all)
│       ├── config.py                 # Config loader (YAML/JSON with merge)
│       ├── config_registry.py        # Model aliases, registries, resolver functions
│       ├── exceptions.py             # Custom exception hierarchy
│       ├── backends/                 # Model backend implementations
│       │   ├── base.py               # Backend base class
│       │   ├── static_embedding.py   # Static embeddings (model2vec)
│       │   ├── sentencetransformer.py # SentenceTransformer backend
│       │   ├── litellm.py            # LLM backend (litellm)
│       │   └── reranker_st.py        # Reranker backend
│       ├── benchmarks/               # Benchmarking infrastructure
│       │   ├── cli.py                # CLI: novelentitymatcher-bench
│       │   ├── runner.py             # BenchmarkRunner
│       │   ├── registry.py           # Dataset registry
│       │   ├── loader.py             # HuggingFace dataset loader
│       │   ├── base.py               # Benchmark base classes
│       │   ├── shared.py             # Shared utilities
│       │   ├── classifier_bench.py   # Classifier benchmarks
│       │   ├── novelty_bench.py      # Novelty benchmarks
│       │   ├── async_bench.py        # Async performance benchmarks
│       │   ├── infra_bench.py        # ANN & reranker benchmarks
│       │   ├── weight_optimizer.py   # Bayesian weight optimization (Optuna)
│       │   ├── visualization.py      # Result visualization
│       │   ├── classification/       # Classification benchmark impl
│       │   ├── entity_resolution/    # Entity resolution benchmark impl
│       │   └── novelty/              # Novelty benchmark impl
│       ├── core/                     # Core matching layer
│       │   ├── matcher.py            # Matcher (unified entry, 701 lines)
│       │   ├── embedding_matcher.py  # EmbeddingMatcher (zero-shot)
│       │   ├── classifier.py         # SetFitClassifier
│       │   ├── bert_classifier.py    # BERTClassifier
│       │   ├── hybrid.py             # HybridMatcher
│       │   ├── blocking.py           # Blocking strategies (BM25, TF-IDF, Fuzzy)
│       │   ├── reranker.py           # CrossEncoderReranker
│       │   ├── hierarchy.py          # HierarchicalMatcher
│       │   ├── normalizer.py         # TextNormalizer
│       │   ├── matching_strategy.py  # MatcherFacade strategy interface
│       │   ├── matcher_components.py # Component factory
│       │   ├── matcher_engines.py    # Batch, diagnosis, hybrid engines
│       │   ├── matcher_entity.py     # EntityMatcher internal
│       │   ├── matcher_runtime.py    # Runtime state management
│       │   ├── matcher_shared.py     # Shared helpers
│       │   └── async_utils.py        # Async execution utilities
│       ├── data/                     # Bundled package data
│       │   ├── country_codes.json
│       │   └── default_config.json
│       ├── ingestion/                # Data ingestion pipelines
│       │   ├── cli.py                # CLI: novelentitymatcher-ingest
│       │   ├── base.py               # Base fetcher class
│       │   ├── currencies.py
│       │   ├── industries.py
│       │   ├── languages.py
│       │   ├── occupations.py
│       │   ├── products.py
│       │   ├── timezones.py
│       │   └── universities.py
│       ├── monitoring/               # Monitoring & metrics
│       │   ├── metrics.py
│       │   └── performance.py
│       ├── novelty/                  # Novelty detection subsystem
│       │   ├── cli.py                # CLI: novelentitymatcher-review
│       │   ├── entity_matcher.py     # NovelEntityMatcher orchestration
│       │   ├── clustering/           # Clustering backends
│       │   │   ├── base.py           # ClusteringBackend ABC
│       │   │   ├── backends.py       # HDBSCAN, SOPTICS, UMAP backends
│       │   │   ├── scalable.py       # ScalableClusterer
│       │   │   ├── validation.py     # ClusterValidator
│       │   │   └── params.py         # Parameter selection
│       │   ├── config/               # Detection configuration
│       │   │   ├── base.py           # DetectionConfig
│       │   │   ├── strategies.py     # Per-strategy configs
│       │   │   └── weights.py        # WeightConfig
│       │   ├── core/                 # Detector core
│       │   │   ├── detector.py       # NoveltyDetector
│       │   │   ├── strategies.py     # StrategyRegistry
│       │   │   ├── signal_combiner.py # SignalCombiner
│       │   │   └── metadata.py       # MetadataBuilder
│       │   ├── evaluation/           # Evaluation utilities
│       │   │   ├── evaluator.py      # NoveltyEvaluator
│       │   │   └── splitters.py      # OOD data splitters
│       │   ├── extraction/           # Evidence extraction
│       │   ├── proposal/             # Class proposal generation
│       │   │   ├── llm.py            # LLMClassProposer
│       │   │   └── retrieval.py      # RetrievalAugmentedProposer
│       │   ├── schemas/              # Pydantic data models
│       │   │   ├── models.py         # Domain models
│       │   │   ├── results.py        # Result models
│       │   │   └── reports.py        # Report models
│       │   ├── storage/              # Persistence layer
│       │   │   ├── index.py          # ANNIndex, ANNBackend
│       │   │   ├── review.py         # ProposalReviewManager
│       │   │   └── persistence.py    # File export
│       │   ├── strategies/           # Detection strategies (12+)
│       │   │   ├── base.py           # NoveltyStrategy ABC
│       │   │   ├── confidence.py
│       │   │   ├── knn_distance.py
│       │   │   ├── uncertainty.py
│       │   │   ├── clustering.py
│       │   │   ├── self_knowledge.py / self_knowledge_impl.py
│       │   │   ├── prototypical.py / prototypical_impl.py
│       │   │   ├── oneclass.py / oneclass_impl.py
│       │   │   ├── pattern.py / pattern_impl.py
│       │   │   ├── setfit.py / setfit_impl.py / setfit_centroid.py
│       │   │   ├── mahalanobis.py
│       │   │   ├── lof.py
│       │   │   └── conformal.py
│       │   └── utils/                # Shared novelty utilities
│       ├── pipeline/                 # Discovery pipeline
│       │   ├── discovery.py          # DiscoveryPipeline (top-level API)
│       │   ├── config.py             # PipelineConfig
│       │   ├── contracts.py          # PipelineStage ABC, StageContext
│       │   ├── orchestrator.py       # PipelineOrchestrator
│       │   ├── pipeline_builder.py   # PipelineBuilder
│       │   ├── adapters.py           # Stage implementations (5 stages)
│       │   ├── discovery_support.py  # Helper functions
│       │   └── match_result.py       # MatchResultWithMetadata
│       └── utils/                    # Shared utilities
│           ├── logging_config.py
│           ├── validation.py
│           ├── preprocessing.py
│           ├── embeddings.py
│           ├── benchmark_dataset.py
│           ├── benchmark_reporting.py
│           ├── benchmarks.py
│           └── learning_curves.py
├── tests/                            # Test suite
│   ├── conftest.py                   # Shared fixtures
│   ├── fixtures/                     # Test data fixtures
│   ├── unit/                         # Unit tests (mirrors src/ structure)
│   │   ├── backends/
│   │   ├── benchmarks/
│   │   ├── core/
│   │   ├── ingestion/
│   │   ├── monitoring/
│   │   ├── novelty/
│   │   ├── pipeline/
│   │   └── utils/
│   └── integration/                  # Integration tests
│       ├── backends/
│       ├── core/
│       └── utils/
├── config.yaml                       # Default config (model, training, embedding)
├── pyproject.toml                    # Project metadata, deps, tool configs
├── uv.lock                           # Lock file
├── AGENTS.md                         # AI agent instructions
├── CLAUDE.md                         # Claude Code instructions
├── README.md                         # Project readme
├── CHANGELOG.md                      # Version history
├── CONTRIBUTING.md                   # Contribution guidelines
├── LICENSE                           # MIT license
├── mkdocs.yml                        # Documentation config
├── .pre-commit-config.yaml           # Pre-commit hooks
└── .python-version                   # Python version pin
```

## Key Locations

| Purpose | Path |
|---|---|
| Main package | `src/novelentitymatcher/` |
| Package init (lazy exports) | `src/novelentitymatcher/__init__.py` |
| Full public API | `src/novelentitymatcher/api.py` |
| Core Matcher | `src/novelentitymatcher/core/matcher.py` |
| Discovery pipeline | `src/novelentitymatcher/pipeline/discovery.py` |
| Novelty orchestrator | `src/novelentitymatcher/novelty/entity_matcher.py` |
| Novelty detector | `src/novelentitymatcher/novelty/core/detector.py` |
| Pipeline config | `src/novelentitymatcher/pipeline/config.py` |
| Model registry | `src/novelentitymatcher/config_registry.py` |
| Config loader | `src/novelentitymatcher/config.py` |
| Exceptions | `src/novelentitymatcher/exceptions.py` |
| CLI: ingestion | `src/novelentitymatcher/ingestion/cli.py` |
| CLI: benchmarks | `src/novelentitymatcher/benchmarks/cli.py` |
| CLI: review | `src/novelentitymatcher/novelty/cli.py` |
| Default config | `config.yaml` |
| Pydantic schemas | `src/novelentitymatcher/novelty/schemas/models.py` |
| Strategy base class | `src/novelentitymatcher/novelty/strategies/base.py` |
| Strategy registry | `src/novelentitymatcher/novelty/core/strategies.py` |
| Pipeline stages | `src/novelentitymatcher/pipeline/adapters.py` |
| Pipeline contracts | `src/novelentitymatcher/pipeline/contracts.py` |
| Pipeline builder | `src/novelentitymatcher/pipeline/pipeline_builder.py` |
| Test config | `tests/conftest.py` |
| Tool config (ruff, pytest, mypy) | `pyproject.toml` |

## Naming Conventions

### Files
- **Package modules**: lowercase, no separators (e.g., `entity_matcher.py`, `pipeline_builder.py`)
- **Private/internal modules**: prefixed with underscore (e.g., `_impl.py` suffix for strategy implementations like `self_knowledge_impl.py`, `prototypical_impl.py`)
- **Test files**: `test_<module>.py` in corresponding `tests/unit/` or `tests/integration/` subdirectory
- **Config files**: lowercase with underscores (e.g., `config_registry.py`, `logging_config.py`)
- **CLI entry points**: `cli.py` in each subsystem (`ingestion/cli.py`, `benchmarks/cli.py`, `novelty/cli.py`)

### Classes
- **Public API classes**: PascalCase (e.g., `Matcher`, `DiscoveryPipeline`, `NoveltyDetector`, `NovelEntityMatcher`)
- **Internal classes**: PascalCase with underscore prefix (e.g., `_EntityMatcher`, `_BatchEngine`, `_HybridEngine`, `_NovelEntityMatcherCompat`)
- **Pydantic models**: PascalCase with `Config` suffix for configs (e.g., `PipelineConfig`, `DetectionConfig`, `KNNConfig`)
- **Result models**: PascalCase with `Result` or `Report` suffix (e.g., `NovelEntityMatchResult`, `NovelClassDiscoveryReport`, `StageResult`)
- **ABC base classes**: PascalCase with descriptive suffix (e.g., `ClusteringBackend`, `PipelineStage`, `NoveltyStrategy`)
- **Registry classes**: PascalCase with `Registry` suffix (e.g., `StrategyRegistry`, `ClusteringBackendRegistry`)
- **Fetcher classes**: PascalCase with `Fetcher` suffix (e.g., `LanguagesFetcher`, `CurrenciesFetcher`)
- **Exception classes**: PascalCase with `Error` suffix (e.g., `SemanticMatcherError`, `ValidationError`, `MatchingError`)

### Functions/Methods
- Public methods: snake_case (e.g., `discover()`, `match_batch()`, `fit_async()`)
- Private methods: snake_case with underscore prefix (e.g., `_build_orchestrator()`, `_collect_match_result_sync()`)
- Resolver functions: `resolve_` prefix (e.g., `resolve_model_alias()`, `resolve_matcher_mode()`)
- Factory methods: `from_` prefix (e.g., `from_config()`, `from_dict()`)
- CLI commands: `cmd_` prefix (e.g., `cmd_list()`, `cmd_approve()`, `cmd_show()`)

### Constants
- Module-level constants: UPPER_SNAKE_CASE (e.g., `MODEL_SPECS`, `MODEL_REGISTRY`, `NOVEL_DETECTION_CONFIG`, `LLM_PROVIDERS`)

## File Organization

### Source Layout
- Uses **src layout**: package lives under `src/novelentitymatcher/`
- Build configured via `pyproject.toml`: `packages = ["src/novelentitymatcher"]`

### Module Decomposition
Large modules are decomposed into internal sub-modules:
- `core/matcher.py` (701 lines) is the facade; internals split into `matcher_components.py`, `matcher_engines.py`, `matcher_entity.py`, `matcher_runtime.py`, `matcher_shared.py`
- `novelty/strategies/` has public strategy files (`confidence.py`, `knn_distance.py`) with separate `_impl.py` files for complex implementations
- `pipeline/adapters.py` contains all 5 stage implementations as separate classes

### Test Mirroring
- `tests/unit/` mirrors `src/novelentitymatcher/` structure exactly
- `tests/integration/` mirrors key subsystems
- `tests/conftest.py` provides shared fixtures
- `tests/fixtures/` holds test data

### Data Organization
- `data/raw/` — Raw downloaded data per domain
- `data/processed/` — Cleaned CSV output per domain
- `data/hf_benchmarks/` — Cached HuggingFace datasets
- `src/novelentitymatcher/data/` — Bundled package data (shipped with wheel)
- `checkpoints/` — Fine-tuned model checkpoints
- `proposals/` — Discovery output (auto-generated)
- `benchmark_results/` — Benchmark CSV output

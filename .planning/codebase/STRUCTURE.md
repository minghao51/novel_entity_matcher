# Codebase Structure

**Analysis Date:** 2026-05-11

## Directory Layout

```
novel_entity_matcher/
├── .claude/                    # Claude agent config
├── .github/                    # GitHub workflows / CI
├── .opencode/                  # OpenCode agent config
├── .planning/                  # Codebase analysis docs (this file)
├── .superset/                  # Superset hooks config
├── .tmp/                       # Temporary scratch files
├── artifacts/                  # Benchmark artifacts and outputs
├── benchmark_results/          # Benchmark result JSON/CSV files
├── checkpoints/                # Model training checkpoints
├── data/                       # Raw and processed datasets
│   ├── hf_benchmarks/          # HuggingFace benchmark datasets
│   ├── processed/              # Normalized/processed datasets (7 domains)
│   └── raw/                    # Raw scraped data (7 domains)
├── dist/                       # Built distribution packages (.whl, .tar.gz)
├── docs/                       # MkDocs documentation source
│   ├── api/                    # API reference pages
│   ├── architecture/           # Architecture diagrams/pages
│   ├── methodology/            # Methodology documentation
│   ├── notebooks/              # Rendered notebook HTML
│   └── ...                     # Other doc pages
├── examples/                   # Usage examples
│   ├── current/                # Current/active examples
│   ├── legacy/                 # Legacy examples
│   └── raw/                    # Raw example data
├── experiments/                # Experimental scripts (mostly empty)
├── notebooks/                  # Marimo/Quarto notebooks (4 explorations)
├── proposals/                  # Discovery run outputs (YAML + JSON)
├── scripts/                    # Build and utility scripts
├── site/                       # Built MkDocs static site
├── src/                        # Source code root
│   └── novelentitymatcher/     # Main package
│       ├── backends/           # Embedding/reranker backend abstractions
│       ├── benchmarks/         # Benchmark runners, dataset registry, CLI
│       ├── core/               # Core matching engine
│       ├── data/               # Package-bundled data (country codes, defaults)
│       ├── ingestion/          # External dataset ingestion
│       ├── monitoring/         # Metrics and performance tracking
│       ├── novelty/            # Novelty detection subsystem
│       ├── pipeline/           # Discovery pipeline orchestration
│       └── utils/              # Shared utilities
└── tests/                      # Test suite
    ├── e2e/                    # End-to-end tests (empty)
    ├── fixtures/               # Test fixture files (JSON samples)
    ├── integration/            # Integration tests
    └── unit/                   # Unit tests (mirrors src/ structure)
```

## Directory Purposes

**`src/novelentitymatcher/core/`:**
- Purpose: Entity matching engine with multiple training modes
- Contains: `Matcher` (main facade), `EmbeddingMatcher` (zero-shot), `_EntityMatcher` (trained), `BERTClassifier`, `SetFitClassifier`, strategy pattern, blocking, reranking, hierarchy, vector store, normalization
- Key files: `matcher.py` (708 lines, main entry), `matching_strategy.py` (strategy hierarchy), `embedding_matcher.py`, `classifier.py`, `bert_classifier.py`

**`src/novelentitymatcher/novelty/`:**
- Purpose: Out-of-distribution detection and novel class discovery
- Contains: Detector core, 15+ strategy implementations, clustering backends, LLM proposal, evaluation, active learning, drift detection, extraction, schemas, storage, config, review management
- Key files: `core/detector.py` (strategy orchestrator), `entity_matcher.py` (`NovelEntityMatcher` facade), `strategies/` (15+ OOD strategies), `proposal/llm.py` (LLM-based naming), `storage/review.py` (HITL)

**`src/novelentitymatcher/pipeline/`:**
- Purpose: Staged discovery pipeline with builder and orchestrator
- Contains: `PipelineOrchestrator` (sequential stage runner), `PipelineBuilder` (constructs 7-stage pipeline), stage contracts, adapters (5 stages), config, match result types, optional drift/stability stages
- Key files: `discovery.py` (`DiscoveryPipeline` facade), `orchestrator.py`, `pipeline_builder.py`, `adapters.py`, `contracts.py`

**`src/novelentitymatcher/backends/`:**
- Purpose: Abstract and concrete embedding/reranker providers
- Contains: `EmbeddingBackend` (ABC), `RerankerBackend` (ABC), `StaticEmbeddingBackend`, `SentenceTransformerBackend`, `LiteLLMEmbedding`, `LiteLLMReranker`, `STReranker`
- Key files: `base.py` (ABCs), `static_embedding.py` (model2vec/static), `sentencetransformer.py`, `litellm.py`

**`src/novelentitymatcher/benchmarks/`:**
- Purpose: Benchmark execution against HuggingFace datasets
- Contains: Dataset registry, benchmark runners (entity resolution, classification, novelty), CLI, visualization, weight optimizer
- Key files: `cli.py` (840 lines), `runner.py`, `registry.py`

**`src/novelentitymatcher/ingestion/`:**
- Purpose: External dataset ingestion for 7 entity domains
- Contains: Per-domain ingesters (currencies, industries, languages, occupations, products, timezones, universities), base utilities, CLI
- Key files: `cli.py`, `base.py`, domain-specific modules

**`src/novelentitymatcher/utils/`:**
- Purpose: Shared utilities used across all layers
- Contains: Validation, logging config, embedding cache, embedding helpers, preprocessing, API key management, benchmark helpers, learning curves
- Key files: `validation.py`, `logging_config.py`, `embedding_cache.py`, `embeddings.py`

**`src/novelentitymatcher/monitoring/`:**
- Purpose: Metrics and performance tracking
- Contains: Metric event creation, performance utilities
- Key files: `metrics.py`, `performance.py`

**`src/novelentitymatcher/data/`:**
- Purpose: Package-bundled static data
- Contains: `country_codes.json`, `default_config.json`
- Key files: `default_config.json` (fallback configuration)

**`tests/unit/`:**
- Purpose: Fast isolated unit tests
- Structure: Mirrors `src/` layout (`backends/`, `benchmarks/`, `core/`, `ingestion/`, `monitoring/`, `novelty/`, `pipeline/`)
- Contains: Per-module test files, `utils/` for test helpers

**`tests/integration/`:**
- Purpose: Tests that span multiple modules or require external services
- Structure: `backends/`, `core/`, `utils/` subdirs + top-level integration test files
- Key files: `test_integration.py`, `test_novel_entity_matcher.py`, `test_async_sync_parity.py`

**`data/`:**
- Purpose: External datasets (raw and processed)
- Contains: 7 domains (currencies, industries, languages, occupations, products, timezones, universities), HuggingFace benchmark cache
- Generated: Yes (by `novelentitymatcher-ingest` CLI)
- Committed: Yes

**`proposals/`:**
- Purpose: Discovery run outputs — YAML proposals and JSON review records
- Contains: Timestamped discovery summaries + review records
- Generated: Yes (by discovery pipeline)
- Committed: Yes

**`notebooks/`:**
- Purpose: Interactive explorations via Marimo/Quarto
- Contains: 4 numbered explorations (entity matching, novelty detection, discovery pipeline, benchmarks)
- Format: `.py` (marimo) + `.qmd` (Quarto) pairs

**`docs/`:**
- Purpose: MkDocs documentation source
- Contains: Architecture, API reference, guides, methodology, rendered notebooks
- Built via: `make docs` -> `mkdocs build`

**`site/`:**
- Purpose: Built static documentation site
- Generated: Yes (by `mkdocs build`)
- Committed: Yes

**`dist/`:**
- Purpose: Built Python distribution packages
- Generated: Yes (by `uv build` / `python -m build`)
- Committed: Yes (with `.gitignore`)

**`artifacts/`:**
- Purpose: Benchmark artifacts and temporary outputs
- Contains: `bench-smoke/`, `benchmarks/`, intermediate JSON files
- Generated: Yes

**`benchmark_results/`:**
- Purpose: Persisted benchmark results
- Contains: JSON and CSV result files from benchmark runs
- Generated: Yes

**`checkpoints/`:**
- Purpose: Model training checkpoints
- Contains: `checkpoint-135/` (fine-tuned model weights)
- Generated: Yes

## Key File Locations

**Entry Points:**
- `src/novelentitymatcher/__init__.py`: Package init with lazy exports and public API
- `src/novelentitymatcher/api.py`: Full re-export surface for `from novelentitymatcher.api import *`
- `src/novelentitymatcher/core/matcher.py`: `Matcher` class — primary entity matching facade
- `src/novelentitymatcher/novelty/entity_matcher.py`: `NovelEntityMatcher` — novelty-aware matching facade
- `src/novelentitymatcher/pipeline/discovery.py`: `DiscoveryPipeline` — pipeline-first discovery facade
- `src/novelentitymatcher/ingestion/cli.py`: `novelentitymatcher-ingest` CLI entry
- `src/novelentitymatcher/benchmarks/cli.py`: `novelentitymatcher-bench` CLI entry
- `src/novelentitymatcher/novelty/cli.py`: `novelentitymatcher-review` CLI entry

**Configuration:**
- `config.yaml`: Root project configuration (model defaults, training params, embedding threshold)
- `src/novelentitymatcher/config.py`: `Config` loader class
- `src/novelentitymatcher/config_registry.py`: Model registries, aliases, specs, mode resolution
- `src/novelentitymatcher/pipeline/config.py`: `PipelineConfig` Pydantic model
- `src/novelentitymatcher/novelty/config/base.py`: `DetectionConfig` Pydantic model
- `src/novelentitymatcher/novelty/config/strategies.py`: Per-strategy config models
- `src/novelentitymatcher/data/default_config.json`: Package-bundled fallback config

**Core Logic:**
- `src/novelentitymatcher/core/matcher.py`: Unified `Matcher` with mode auto-detection
- `src/novelentitymatcher/core/embedding_matcher.py`: Zero-shot embedding-based matching
- `src/novelentitymatcher/core/classifier.py`: SetFit few-shot classifier
- `src/novelentitymatcher/core/bert_classifier.py`: BERT fine-tuning classifier
- `src/novelentitymatcher/core/matching_strategy.py`: Strategy pattern (zero-shot, head-only, full, bert, hybrid)
- `src/novelentitymatcher/core/hybrid.py`: Hybrid blocking + retrieval matcher
- `src/novelentitymatcher/core/blocking.py`: Blocking strategies (BM25, TF-IDF, Fuzzy, NoOp)
- `src/novelentitymatcher/core/reranker.py`: Cross-encoder reranking
- `src/novelentitymatcher/core/normalizer.py`: Text normalization
- `src/novelentitymatcher/core/hierarchy.py`: Hierarchical entity matching
- `src/novelentitymatcher/core/vector_store.py`: In-memory vector store

**Pipeline:**
- `src/novelentitymatcher/pipeline/orchestrator.py`: Sequential stage execution
- `src/novelentitymatcher/pipeline/pipeline_builder.py`: 7-stage pipeline construction
- `src/novelentitymatcher/pipeline/adapters.py`: Stage implementations (5 core stages)
- `src/novelentitymatcher/pipeline/contracts.py`: `StageContext`, `StageResult`, `PipelineRunResult`, `PipelineStage` ABC
- `src/novelentitymatcher/pipeline/stages/drift_hook.py`: Optional drift detection stage
- `src/novelentitymatcher/pipeline/stages/stability_filter.py`: Optional cluster stability stage

**Novelty Detection:**
- `src/novelentitymatcher/novelty/core/detector.py`: `NoveltyDetector` — strategy orchestrator
- `src/novelentitymatcher/novelty/core/strategies.py`: `StrategyRegistry` for strategy lookup
- `src/novelentitymatcher/novelty/core/signal_combiner.py`: Multi-signal fusion
- `src/novelentitymatcher/novelty/core/score_calibrator.py`: OOD score normalization
- `src/novelentitymatcher/novelty/strategies/`: 15+ OOD strategy implementations
- `src/novelentitymatcher/novelty/clustering/`: Clustering backends and validation
- `src/novelentitymatcher/novelty/proposal/llm.py`: LLM-based class naming via LiteLLM/DSPy
- `src/novelentitymatcher/novelty/storage/review.py`: HITL review state management
- `src/novelentitymatcher/novelty/storage/index.py`: ANN index abstraction

**Testing:**
- `tests/conftest.py`: Shared pytest fixtures
- `tests/unit/`: Unit tests mirroring `src/` structure
- `tests/integration/`: Integration tests with external service dependencies
- `tests/e2e/`: End-to-end tests (currently empty)
- `tests/fixtures/`: Test data files (`sample_countries.json`, `sample_texts.json`)

**Build/Tooling:**
- `pyproject.toml`: Project metadata, dependencies, tool config (ruff, mypy, pytest, coverage)
- `Makefile`: Docs build targets (`notebooks`, `docs`)
- `mkdocs.yml`: Documentation site configuration
- `.pre-commit-config.yaml`: Pre-commit hooks
- `scripts/generate_notebook_docs.py`: Notebook doc generation script
- `scripts/setup_llm.sh`: LLM provider setup script

## Naming Conventions

**Files:**
- Python modules: `snake_case.py` (e.g., `embedding_matcher.py`, `score_calibrator.py`)
- Private/internal modules: Leading underscore prefix (e.g., `_entity_matcher` as class name, but file is `matcher_entity.py`)
- Test files: `test_<module>.py` (e.g., `test_config.py`, `test_discovery_pipeline.py`)
- Config files: `snake_case.yaml` / `snake_case.json` (e.g., `config.yaml`, `default_config.json`)
- Notebooks: `NN_descriptive_name.py` / `.qmd` (e.g., `01_entity_matching_explorer.py`)
- Proposal outputs: `discovery_YYYYMMDD-HHMMSS_<hash>_<type>.<ext>`

**Directories:**
- Package directories: `snake_case` (e.g., `novelentitymatcher`, `pipeline`, `backends`)
- Test directories: Mirror source structure (`unit/core/`, `unit/novelty/`, etc.)
- Data directories: Domain names in `snake_case` (`currencies/`, `industries/`, `languages/`)

**Classes:**
- Public facades: PascalCase noun (e.g., `Matcher`, `NoveltyDetector`, `DiscoveryPipeline`)
- ABC bases: PascalCase with descriptive suffix (e.g., `EmbeddingBackend`, `MatchingStrategy`, `PipelineStage`)
- Private implementations: Leading underscore (e.g., `_EntityMatcher`, `_BatchEngine`, `_HybridEngine`)
- Config models: PascalCase + `Config` suffix (e.g., `DetectionConfig`, `PipelineConfig`, `KNNConfig`)

**Module Organization:**
- `__init__.py` files are used for package marker and lazy exports
- Implementation details split into `<name>_impl.py` files (e.g., `setfit_impl.py`, `pattern_impl.py`)
- Base/ABC classes in `base.py` within each subpackage
- Schemas/models in dedicated `schemas/` subdirectories

## Where to Add New Code

**New Matching Strategy:**
- Implementation: `src/novelentitymatcher/core/matching_strategy.py` (add class extending `MatchingStrategy`)
- Registration: Add to `_STRATEGY_MAP` dict in same file
- Tests: `tests/unit/core/`

**New Novelty Detection Strategy:**
- Implementation: `src/novelentitymatcher/novelty/strategies/<name>.py` (extend `NoveltyStrategy`)
- Registration: Add to `StrategyRegistry` in `novelty/core/strategies.py`
- Config: Add config model to `novelty/config/strategies.py`
- Tests: `tests/unit/novelty/`

**New Pipeline Stage:**
- Implementation: Extend `PipelineStage` ABC from `pipeline/contracts.py`
- Registration: Add to `PipelineBuilder.build()` in `pipeline/pipeline_builder.py`
- Config: Add fields to `PipelineConfig` in `pipeline/config.py`
- Tests: `tests/unit/pipeline/`

**New Embedding Backend:**
- Implementation: `src/novelentitymatcher/backends/<name>.py` (extend `EmbeddingBackend`)
- Registration: Wire into model resolution in `config_registry.py`
- Tests: `tests/unit/backends/`

**New Ingestion Domain:**
- Implementation: `src/novelentitymatcher/ingestion/<domain>.py`
- Registration: Add to `INGESTORS` dict in `ingestion/cli.py`
- Tests: `tests/unit/ingestion/`

**Utilities:**
- Shared helpers: `src/novelentitymatcher/utils/`
- Test utilities: `tests/unit/utils/`, `tests/integration/utils/`

## Special Directories

**`dist/`:**
- Purpose: Built distribution packages
- Generated: Yes (by `uv build` or `python -m build`)
- Committed: Yes

**`site/`:**
- Purpose: Built MkDocs static site for documentation
- Generated: Yes (by `mkdocs build` or `make docs`)
- Committed: Yes

**`proposals/`:**
- Purpose: Discovery run outputs with timestamp-based naming
- Generated: Yes (by `DiscoveryPipeline.discover()` / `NovelEntityMatcher.discover_novel_classes()`)
- Committed: Yes

**`checkpoints/`:**
- Purpose: Model fine-tuning checkpoints
- Generated: Yes (by training runs)
- Committed: Yes

**`benchmark_results/`:**
- Purpose: Persisted benchmark outputs
- Generated: Yes (by `novelentitymatcher-bench` CLI)
- Committed: Yes

**`artifacts/`:**
- Purpose: Intermediate benchmark and processing artifacts
- Generated: Yes
- Committed: Yes

**`notebooks/_freeze/`:**
- Purpose: Quarto freeze cache for notebook rendering
- Generated: Yes (by `quarto render`)
- Committed: Yes

**`.venv/`:**
- Purpose: Python virtual environment
- Generated: Yes (by `uv sync`)
- Committed: No (gitignored)

---

*Structure analysis: 2026-05-11*

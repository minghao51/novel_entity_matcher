# Architecture

**Analysis Date:** 2026-05-11

## Pattern Overview

**Overall:** Layered pipeline architecture with strategy pattern for matching and novelty detection.

**Key Characteristics:**
- Pipeline-first discovery: staged processing via `PipelineOrchestrator` running ordered `PipelineStage` instances
- Strategy pattern for matching modes (zero-shot, head-only, full, bert, hybrid) and novelty detection algorithms
- Facade pattern for public API surface (`Matcher`, `NovelEntityMatcher`, `DiscoveryPipeline`)
- Backend abstraction for embedding providers (static, SentenceTransformer, LiteLLM)
- Lazy imports via `__getattr__` to defer heavy ML dependencies

## Layers

### Presentation / CLI Layer
- Purpose: CLI entry points and user-facing argument parsing
- Location: `src/novelentitymatcher/ingestion/cli.py`, `src/novelentitymatcher/benchmarks/cli.py`, `src/novelentitymatcher/novelty/cli.py`
- Contains: `argparse`-based CLIs registered as `[project.scripts]` in `pyproject.toml`
- Depends on: Business logic layer
- Used by: End users via console scripts (`novelentitymatcher-ingest`, `novelentitymatcher-bench`, `novelentitymatcher-review`)

### Public API Layer
- Purpose: Three primary facades for different usage patterns
- Location: `src/novelentitymatcher/__init__.py` (lazy exports), `src/novelentitymatcher/api.py` (full re-exports)
- Contains: `Matcher`, `NovelEntityMatcher`, `DiscoveryPipeline`
- Depends on: Core layer, Pipeline layer, Novelty layer
- Used by: External consumers, examples, tests

### Core Matching Layer
- Purpose: Entity matching with multiple training strategies
- Location: `src/novelentitymatcher/core/`
- Contains: `Matcher` (unified entry point), `_EntityMatcher` (SetFit/BERT), `EmbeddingMatcher` (zero-shot), matching strategy hierarchy, blocking, reranking, normalization, vector store
- Depends on: `backends/` for embedding generation, `config.py` for model aliases/specs, `utils/` for validation
- Used by: API layer, Novelty layer (for reference corpus), Pipeline layer

### Novelty Detection Layer
- Purpose: Out-of-distribution detection, clustering, and novel class discovery
- Location: `src/novelentitymatcher/novelty/`
- Contains: `NoveltyDetector` (strategy orchestrator), 15+ novelty strategies, clustering backends (HDBSCAN, OPTICS, Leiden), LLM-based class proposal, evaluation, drift detection, active learning, storage/indexing
- Depends on: Core matching layer (for embeddings and reference corpus), `backends/`, external ML libs
- Used by: `NovelEntityMatcher`, `DiscoveryPipeline`

### Pipeline Layer
- Purpose: Staged discovery pipeline with orchestrator and builder
- Location: `src/novelentitymatcher/pipeline/`
- Contains: `PipelineOrchestrator`, `PipelineBuilder`, `PipelineStage` contracts, stage adapters (`MatcherMetadataStage`, `OODDetectionStage`, `CommunityDetectionStage`, `ClusterEvidenceStage`, `ProposalStage`), optional `DriftCheckStage` and `StabilityFilterStage`
- Depends on: Core matching layer, Novelty detection layer
- Used by: `DiscoveryPipeline`, `NovelEntityMatcher`

### Backend Abstraction Layer
- Purpose: Pluggable embedding and reranker providers
- Location: `src/novelentitymatcher/backends/`
- Contains: `EmbeddingBackend` (ABC), `RerankerBackend` (ABC), concrete implementations: `StaticEmbeddingBackend`, `SentenceTransformerBackend`, `LiteLLMEmbedding`, `LiteLLMReranker`, `STReranker`
- Depends on: External ML libraries (sentence-transformers, model2vec, litellm)
- Used by: Core matching layer

### Data Access / Ingestion Layer
- Purpose: External dataset ingestion and processing
- Location: `src/novelentitymatcher/ingestion/`
- Contains: Domain-specific ingesters (currencies, industries, languages, occupations, products, timezones, universities), base utilities
- Depends on: External data APIs, `data/raw/` and `data/processed/` directories
- Used by: CLI (`novelentitymatcher-ingest`)

### Infrastructure Layer
- Purpose: Configuration, logging, monitoring, validation, benchmarks
- Location: `src/novelentitymatcher/utils/`, `src/novelentitymatcher/monitoring/`, `src/novelentitymatcher/benchmarks/`, `src/novelentitymatcher/config.py`
- Contains: `Config` loader (YAML/JSON), model registries, embedding cache, logging setup, metrics, benchmark runners, dataset registry
- Depends on: Core libraries (yaml, pydantic)
- Used by: All layers

## Data Flow

### Entity Matching Flow (zero-shot):
1. User creates `Matcher(entities=..., model="potion-32m", mode="zero-shot")`
2. `Matcher.fit()` -> `EmbeddingMatcher.build_index()` encodes all entity names into embeddings
3. `Matcher.match(texts)` -> `ZeroShotStrategy.match()` -> `EmbeddingMatcher.match()` computes cosine similarity against index
4. Returns `{"id": ..., "score": ..., "text": ...}` per query

### Entity Matching Flow (trained):
1. User creates `Matcher(entities=..., mode="auto")` and calls `fit(training_data)`
2. `Matcher._detect_training_mode()` selects head-only/full/bert based on example counts
3. `_EntityMatcher.train()` fits SetFit or BERT classifier on labeled data
4. `Matcher.match()` routes through `HeadOnlyFullStrategy` or `BertStrategy`

### Novelty-Aware Matching Flow:
1. User creates `NovelEntityMatcher(entities=..., matcher=matcher)`
2. `NovelEntityMatcher.match(text)` -> collects matcher metadata + embeddings
3. `NoveltyDetector.detect()` runs configured strategies (confidence, knn_distance, clustering, etc.)
4. `SignalCombiner` merges signals into composite novelty score
5. Returns `NovelEntityMatchResult` with `is_novel`, `novel_score`, `signals`

### Discovery Pipeline Flow (7 stages):
1. `MatcherMetadataStage` — collects top-k candidates, embeddings, reference corpus from fitted `Matcher`
2. `DriftCheckStage` (optional) — compares embedding distribution against baseline
3. `OODDetectionStage` — runs `NoveltyDetector` to flag novel inputs
4. `CommunityDetectionStage` — clusters novel samples via `ScalableClusterer` (HDBSCAN/OPTICS/Leiden)
5. `StabilityFilterStage` (optional) — bootstraps cluster stability assessment
6. `ClusterEvidenceStage` — extracts keywords, representative examples per cluster
7. `ProposalStage` — calls `LLMClassProposer` to name novel clusters

### Pipeline Orchestration:
- `PipelineBuilder` constructs `PipelineOrchestrator` from `PipelineStageConfig` or `PipelineConfig`
- `PipelineOrchestrator.run()` / `.run_async()` executes stages sequentially, threading `StageContext` with mutable artifacts
- Each stage returns `StageResult` with artifacts that merge into shared context

### Discovery Report Generation:
1. Pipeline completes -> `PipelineRunResult` with all stage artifacts
2. `DiscoveryBase._build_discovery_report()` assembles `NovelClassDiscoveryReport`
3. `ProposalReviewManager` creates `ProposalReviewRecord` for human-in-the-loop
4. Report serialized to `proposals/` directory (YAML summary + JSON records)

### Data Ingestion Flow:
1. CLI triggers `run_<domain>()` functions (e.g., `run_languages()`)
2. Each ingester fetches from external source, normalizes, writes to `data/raw/` and `data/processed/`

**State Management:**
- `MatcherRuntimeState` holds mutable matcher state (mode, training status)
- `MatcherComponentFactory` lazily instantiates sub-matchers (embedding, entity, bert, hybrid)
- `StageContext.artifacts` is a mutable dict passed between pipeline stages
- `LRUEmbeddingCache` caches computed embeddings with bounded size
- `ProposalReviewManager` persists review state to JSON file

## Key Abstractions

**`MatchingStrategy` (ABC):**
- Purpose: Polymorphic matching across modes
- Examples: `src/novelentitymatcher/core/matching_strategy.py` (`ZeroShotStrategy`, `HeadOnlyFullStrategy`, `BertStrategy`, `HybridStrategy`)
- Pattern: Strategy pattern with `MatcherFacade` as context

**`PipelineStage` (ABC):**
- Purpose: Composable pipeline stages with sync/async execution
- Examples: `src/novelentitymatcher/pipeline/adapters.py` (`MatcherMetadataStage`, `OODDetectionStage`, etc.)
- Pattern: Stage pattern with `PipelineOrchestrator` as compositor

**`NoveltyStrategy` (ABC):**
- Purpose: Pluggable out-of-distribution detection algorithms
- Examples: `src/novelentitymatcher/novelty/strategies/` (15+ implementations: confidence, knn_distance, clustering, mahalanobis, energy, pattern, self_knowledge, etc.)
- Pattern: Strategy pattern with `NoveltyDetector` as context, `StrategyRegistry` for lookup

**`EmbeddingBackend` / `RerankerBackend` (ABC):**
- Purpose: Abstract embedding generation and reranking
- Examples: `src/novelentitymatcher/backends/` (`StaticEmbeddingBackend`, `SentenceTransformerBackend`, `LiteLLMEmbedding`)
- Pattern: Provider/adapter pattern

**`ClusteringBackend` (ABC):**
- Purpose: Pluggable clustering algorithms for novel class discovery
- Examples: `src/novelentitymatcher/novelty/clustering/backends.py` (`HDBSCANBackend`, `SOPTICSBackend`, `UMAPHDBSCANBackend`)
- Pattern: Strategy pattern with `ScalableClusterer` as context

**`DiscoveryBase`:**
- Purpose: Shared base class for discovery report generation and persistence
- Examples: `src/novelentitymatcher/novelty/discovery_base.py`
- Pattern: Template method pattern for report building

**`Config`:**
- Purpose: Layered configuration loading (repo root -> package defaults -> cwd -> custom override)
- Examples: `src/novelentitymatcher/config.py`
- Pattern: Chain of responsibility for config resolution

**`PipelineConfig` (Pydantic BaseModel):**
- Purpose: Validated, typed configuration for the full discovery pipeline
- Examples: `src/novelentitymatcher/pipeline/config.py`
- Pattern: Configuration object pattern

## Entry Points

**`Matcher` class:**
- Location: `src/novelentitymatcher/core/matcher.py`
- Triggers: Direct instantiation, `NovelEntityMatcher`, `DiscoveryPipeline`
- Responsibilities: Unified entity matching with auto mode detection, training, prediction, diagnostics

**`NovelEntityMatcher` class:**
- Location: `src/novelentitymatcher/novelty/entity_matcher.py`
- Triggers: Direct instantiation for matcher-first novelty workflows
- Responsibilities: Matcher + novelty detection + LLM proposal in a single API

**`DiscoveryPipeline` class:**
- Location: `src/novelentitymatcher/pipeline/discovery.py`
- Triggers: Direct instantiation for pipeline-first discovery workflows (recommended for new projects)
- Responsibilities: Full discovery pipeline with HITL review, promotion, and configuration

**CLI: `novelentitymatcher-ingest`:**
- Location: `src/novelentitymatcher/ingestion/cli.py:main`
- Triggers: Console script entry point
- Responsibilities: Ingest external datasets (languages, currencies, industries, etc.)

**CLI: `novelentitymatcher-bench`:**
- Location: `src/novelentitymatcher/benchmarks/cli.py:main`
- Triggers: Console script entry point
- Responsibilities: Run benchmarks against HuggingFace datasets with configurable models/modes

**CLI: `novelentitymatcher-review`:**
- Location: `src/novelentitymatcher/novelty/cli.py:main`
- Triggers: Console script entry point
- Responsibilities: Human-in-the-loop review of proposed novel classes (list, show, approve, reject, promote)

## Error Handling

**Strategy:** Hierarchical custom exception tree with context-rich error messages.

**Exceptions:**
- `SemanticMatcherError` — base for all library errors (`src/novelentitymatcher/exceptions.py`)
- `ValidationError` — input validation failures with field/entity/suggestion context
- `TrainingError` — training failures with mode and diagnostic details
- `MatchingError` — runtime matching failures
- `ModeError` — invalid mode configuration with valid-mode listing
- `LLMError` — LLM call failures with API key redaction and attempted-model listing

**Patterns:**
- All exceptions carry structured context (entity, field, suggestion, training_mode, details)
- API keys are automatically redacted via `_redact_api_keys()` regex
- Errors surface through the public API with actionable suggestions

## Cross-Cutting Concerns

**Logging:** Configured via `utils/logging_config.py` with `NOVEL_ENTITY_MATCHER_VERBOSE` env var toggle. Lazy logger initialization.

**Validation:** `utils/validation.py` provides entity list validation, threshold bounds checking, model name validation. Input validation happens at construction time.

**Configuration:** Layered config: `Config` loads repo-root `config.yaml`, then cwd `config.yaml`, then any explicit override path. `config.py` also manages model aliases and specs. `PipelineConfig` uses Pydantic for validated pipeline settings.

**Monitoring:** `monitoring/metrics.py` provides metric event creation. `Matcher` accepts optional `metrics_callback` for emitting timing/count metrics.

**Async Support:** All major classes provide `*_async` counterparts. `AsyncExecutor` wraps thread-based execution for CPU-bound ML operations. `asyncio.Lock` prevents concurrent fit operations.

**Caching:** `LRUEmbeddingCache` and `ModelCache` in `utils/embeddings.py` avoid redundant embedding computation and model reloads.

---

*Architecture analysis: 2026-05-11*

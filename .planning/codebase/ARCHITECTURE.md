# Architecture

## High-Level Pattern

**Layered pipeline architecture** with a strategy-pattern novelty detection subsystem. The system follows a "match → detect → cluster → propose" pipeline pattern for entity matching with novel class discovery.

Three public entry points form a tiered API:
1. **`Matcher`** (`src/novelentitymatcher/core/matcher.py`) — Core entity matching (zero-shot, SetFit, BERT, hybrid)
2. **`NovelEntityMatcher`** (`src/novelentitymatcher/novelty/entity_matcher.py`) — Matcher-first orchestration wrapping Matcher + NoveltyDetector
3. **`DiscoveryPipeline`** (`src/novelentitymatcher/pipeline/discovery.py`) — Pipeline-first discovery with staged processing (recommended for new projects)

All three share a common internal pipeline orchestrated by `PipelineOrchestrator` (`src/novelentitymatcher/pipeline/orchestrator.py`).

## Layers & Modules

### Layer 1: Core Matching (`src/novelentitymatcher/core/`)
The foundation layer providing entity matching primitives:
- `matcher.py` — `Matcher` class: unified entry point with auto-strategy selection
- `embedding_matcher.py` — `EmbeddingMatcher`: zero-shot embedding similarity
- `classifier.py` — `SetFitClassifier`: few-shot SetFit training
- `bert_classifier.py` — `BERTClassifier`: BERT-based fine-tuning
- `hybrid.py` — `HybridMatcher`: combines embedding + trained approaches
- `blocking.py` — Blocking strategies (BM25, TF-IDF, Fuzzy) for candidate pre-filtering
- `reranker.py` — `CrossEncoderReranker`: cross-encoder re-scoring
- `hierarchy.py` — `HierarchicalMatcher`: hierarchical entity matching
- `normalizer.py` — `TextNormalizer`: text preprocessing
- `matcher_components.py` / `matcher_engines.py` / `matcher_entity.py` / `matcher_runtime.py` / `matcher_shared.py` — Internal decomposition of Matcher internals
- `matching_strategy.py` — `MatcherFacade`: strategy interface
- `async_utils.py` — Async execution support

### Layer 2: Backends (`src/novelentitymatcher/backends/`)
Pluggable model backends:
- `base.py` — Base backend interface
- `static_embedding.py` — Static embedding models (model2vec, potion)
- `sentencetransformer.py` — SentenceTransformer backend
- `litellm.py` — LLM backend (via litellm for class proposals)
- `reranker_st.py` — SentenceTransformer-based reranker

### Layer 3: Novelty Detection (`src/novelentitymatcher/novelty/`)
The largest subsystem, implementing multi-signal out-of-distribution detection:

**Core** (`novelty/core/`):
- `detector.py` — `NoveltyDetector`: strategy orchestrator, initializes and runs registered strategies
- `strategies.py` — `StrategyRegistry`: registers and resolves detection strategies by name
- `signal_combiner.py` — `SignalCombiner`: combines multi-strategy signals (weighted, voting, any, all)
- `metadata.py` — `MetadataBuilder`: builds `NovelSampleMetadata` from detection results

**Strategies** (`novelty/strategies/`) — Each strategy implements `NoveltyStrategy` base class from `base.py`:
- `confidence.py` — Low confidence detection
- `knn_distance.py` — KNN distance-based detection
- `uncertainty.py` — Prediction uncertainty
- `clustering.py` — Clustering-based detection
- `self_knowledge.py` / `self_knowledge_impl.py` — Self-supervised knowledge detection
- `prototypical.py` / `prototypical_impl.py` — Prototypical network approach
- `oneclass.py` / `oneclass_impl.py` — One-class classification
- `pattern.py` / `pattern_impl.py` — Pattern-based detection
- `setfit.py` / `setfit_impl.py` / `setfit_centroid.py` — SetFit-based detection
- `mahalanobis.py` — Mahalanobis distance with conformal calibration
- `lof.py` — Local Outlier Factor
- `conformal.py` — Conformal prediction helpers

**Clustering** (`novelty/clustering/`):
- `base.py` — `ClusteringBackend` ABC
- `backends.py` — `HDBSCANBackend`, `SOPTICSBackend`, `UMAPHDBSCANBackend`, `ClusteringBackendRegistry`
- `scalable.py` — `ScalableClusterer`: auto-selects backend based on data size
- `validation.py` — `ClusterValidator`: validates cluster quality
- `params.py` — Parameter selection utilities

**Config** (`novelty/config/`):
- `base.py` — `DetectionConfig`: Pydantic model for detection configuration
- `strategies.py` — Per-strategy config models (KNNConfig, ConfidenceConfig, ClusteringConfig, etc.)
- `weights.py` — `WeightConfig`: strategy weight configuration

**Schemas** (`novelty/schemas/`):
- `models.py` — Pydantic models: `NovelSampleMetadata`, `DiscoveryCluster`, `ClassProposal`, `ProposalReviewRecord`, etc.
- `results.py` — `DetectionReport`, `EvaluationReport`, `SampleMetrics`, `StrategyMetrics`
- `reports.py` — Report models

**Storage** (`novelty/storage/`):
- `index.py` — `ANNIndex`, `ANNBackend`: ANN index abstraction (hnswlib, faiss)
- `review.py` — `ProposalReviewManager`: HITL review record management
- `persistence.py` — Export/save proposals and summaries

**Proposal** (`novelty/proposal/`):
- `llm.py` — `LLMClassProposer`: LLM-based class name proposal via litellm
- `retrieval.py` — `RetrievalAugmentedProposer`, `BGERetriever`: RAG-based proposals

**Evaluation** (`novelty/evaluation/`):
- `evaluator.py` — `NoveltyEvaluator`: strategy evaluation
- `splitters.py` — `OODSplitter`, `GradualNoveltySplitter`: train/test splitting for OOD evaluation

**Extraction** (`novelty/extraction/`) — Evidence extraction from clusters

**Utils** (`novelty/utils/`) — Shared novelty utilities

### Layer 4: Pipeline (`src/novelentitymatcher/pipeline/`)
Staged discovery pipeline:
- `discovery.py` — `DiscoveryPipeline`: top-level public API
- `config.py` — `PipelineConfig`: Pydantic model for all pipeline settings
- `contracts.py` — `PipelineStage` ABC, `StageContext`, `StageResult`, `PipelineRunResult`
- `orchestrator.py` — `PipelineOrchestrator`: runs ordered stages against shared context
- `pipeline_builder.py` — `PipelineBuilder` + `PipelineStageConfig`: constructs 5-stage pipelines
- `adapters.py` — Stage implementations: `MatcherMetadataStage`, `OODDetectionStage`, `ClusterEvidenceStage`, `CommunityDetectionStage`, `ProposalStage`
- `discovery_support.py` — Helper functions for match result collection and novel match result building
- `match_result.py` — `MatchResultWithMetadata`: rich match result dataclass

### Layer 5: Ingestion (`src/novelentitymatcher/ingestion/`)
Data ingestion pipelines for building entity lists:
- `base.py` — Base fetcher class with shared utilities
- `cli.py` — CLI entry point (`novelentitymatcher-ingest`)
- Per-domain fetchers: `currencies.py`, `industries.py`, `languages.py`, `occupations.py`, `products.py`, `timezones.py`, `universities.py`

### Layer 6: Benchmarks (`src/novelentitymatcher/benchmarks/`)
Comprehensive benchmarking infrastructure:
- `cli.py` — CLI entry point (`novelentitymatcher-bench`) with subcommands: run, load, list, clear, sweep, bench-classifier, bench-novelty, bench-async, bench-weights, bench-ann, bench-reranker, render, plot
- `runner.py` — `BenchmarkRunner`: orchestrates benchmark execution
- `registry.py` — `DATASET_REGISTRY`: HuggingFace dataset configurations
- `loader.py` — Dataset loading from HuggingFace
- `classification/` — Classification benchmark implementations
- `entity_resolution/` — Entity resolution benchmark implementations
- `novelty/` — Novelty detection benchmark implementations
- `weight_optimizer.py` — Bayesian optimization of ensemble weights (Optuna)
- `classifier_bench.py`, `novelty_bench.py`, `async_bench.py`, `infra_bench.py` — Specific benchmark types
- `visualization.py` — Result visualization and plotting
- `shared.py` — Shared benchmark utilities

### Support Modules
- **Config** (`config.py`, `config_registry.py`) — `Config` class for YAML/JSON config loading + model registry
- **Exceptions** (`exceptions.py`) — `SemanticMatcherError`, `ValidationError`, `TrainingError`, `MatchingError`, `ModeError`, `LLMError`
- **Monitoring** (`monitoring/`) — `metrics.py`, `performance.py`
- **Utils** (`utils/`) — `logging_config.py`, `validation.py`, `preprocessing.py`, `embeddings.py`, `benchmark_dataset.py`, `benchmark_reporting.py`, `benchmarks.py`, `learning_curves.py`
- **Data** (`data/`) — Bundled static data: `country_codes.json`, `default_config.json`

## Data Flow

### Primary Discovery Flow
```
User queries (list[str])
    ↓
DiscoveryPipeline.discover()
    ↓
PipelineOrchestrator runs 5 stages sequentially:
    ↓
[1. MatcherMetadataStage]
    Matcher.match_batch() → MatchResultWithMetadata (predictions, confidences, embeddings, candidate_results)
    ↓
[2. OODDetectionStage]
    NoveltyDetector.detect() → NovelSampleReport (novel samples identified via multi-strategy signals)
    ↓
[3. ClusterEvidenceStage]
    ScalableClusterer.fit_predict() → cluster assignments on novel embeddings
    ↓
[4. CommunityDetectionStage]
    Refines clusters, extracts keywords/examples per cluster → DiscoveryCluster list
    ↓
[5. ProposalStage]
    LLMClassProposer.propose() (or RetrievalAugmentedProposer) → ClassProposal list
    ↓
NovelClassDiscoveryReport (assembled by DiscoveryPipeline)
    ↓
Auto-save (YAML/JSON) + HITL review records
```

### Single Match Flow
```
text → Matcher.predict() → (prediction, confidence, embedding)
                                 ↓
                    NovelEntityMatcher.match() → NovelEntityMatchResult
                      (is_novel decision based on threshold + novelty signals)
```

### Strategy Signal Combining
```
NoveltyDetector.detect()
    → for each strategy in config.strategies:
        strategy.detect() → per-sample signal dict
    → SignalCombiner.combine() → merged novelty decisions
    → MetadataBuilder.build() → NovelSampleMetadata list
```

## Entry Points

### CLI Entry Points (from `pyproject.toml`)
- `novelentitymatcher-ingest` → `src/novelentitymatcher/ingestion/cli.py:main` — Data ingestion
- `novelentitymatcher-bench` → `src/novelentitymatcher/benchmarks/cli.py:main` — Benchmark runner
- `novelentitymatcher-review` → `src/novelentitymatcher/novelty/cli.py:main` — HITL review CLI

### Programmatic Entry Points
- `from novelentitymatcher import Matcher` — Core matching
- `from novelentitymatcher import NovelEntityMatcher` — Novelty-aware matching (matcher-first)
- `from novelentitymatcher import DiscoveryPipeline` — Full discovery pipeline (pipeline-first, recommended)
- `from novelentitymatcher.api import *` — Complete public API surface

### Lazy Import System
`src/novelentitymatcher/__init__.py` uses `__getattr__` with a `_EXPORTS` dict for lazy module loading — imports only resolve when an attribute is accessed.

## Key Abstractions

| Abstraction | Location | Purpose |
|---|---|---|
| `PipelineStage` | `pipeline/contracts.py:51` | ABC for pipeline stages with `run()` / `run_async()` |
| `StageContext` | `pipeline/contracts.py:13` | Mutable context passed between stages (inputs, artifacts, metadata) |
| `NoveltyStrategy` | `novelty/strategies/base.py` | ABC for detection strategies with `initialize()`, `detect()` |
| `ClusteringBackend` | `novelty/clustering/base.py` | ABC for clustering implementations |
| `ANNIndex` / `ANNBackend` | `novelty/storage/index.py` | ANN index abstraction (hnswlib, faiss) |
| `DetectionConfig` | `novelty/config/base.py` | Pydantic config for detection |
| `PipelineConfig` | `pipeline/config.py:27` | Pydantic config for full pipeline |
| `Config` | `config.py:68` | YAML/JSON config loader with merge |
| `Matcher` | `core/matcher.py:54` | Unified entity matcher with auto-strategy selection |

## Configuration

### Config Loading Chain
1. `Config` class (`src/novelentitymatcher/config.py`) searches for config in order:
   - Repo root `config.yaml` (walks up from package to find `.git`/`pyproject.toml`)
   - Package-bundled `data/default_config.json`
   - `config.yaml` in current working directory
2. Custom config path can override via `Config(custom_path=...)`
3. Deep-merge of defaults with custom config

### Model Registry
`src/novelentitymatcher/config_registry.py` maintains:
- `MODEL_SPECS` — All known model aliases with backend type, language, and training support
- `STATIC_MODEL_REGISTRY` — Static embedding models (potion, mrl)
- `DYNAMIC_MODEL_REGISTRY` — SentenceTransformer models (bge, mpnet, minilm)
- `RERANKER_REGISTRY` — Reranker model aliases
- `MATCHER_MODE_REGISTRY` — Mode name → implementation class mapping
- `LLM_PROVIDERS` — LLM provider configs (openrouter, anthropic, openai)

### Pipeline Configuration
`PipelineConfig` (`src/novelentitymatcher/pipeline/config.py`) is a Pydantic `BaseModel` that drives:
- Stage selection and ordering via `stages()` method
- OOD strategy validation
- Clustering backend validation
- All tunable parameters with sensible defaults

### Environment Variables
- `NOVEL_ENTITY_MATCHER_VERBOSE=true` — Enable verbose logging
- `OPENROUTER_API_KEY` / `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` — LLM API keys
- `PYTORCH_ENABLE_MPS_FALLBACK=1` — Auto-set on macOS ARM64

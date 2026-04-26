# Architecture

**Analysis Date:** 2026-04-23

## Pattern Overview

**Overall:** Layered architecture with pipeline orchestration

**Key Characteristics:**
- Multi-layer separation: core matching → novelty detection → pipeline orchestration
- Async-first design with sync fallbacks
- Strategy pattern for pluggable detection algorithms
- Modular configuration via Pydantic models
- Lazy imports for optional features

## Layers

**Core Layer (Matching):**
- Purpose: Text normalization, embedding generation, classification/matching
- Location: `src/novelentitymatcher/core/`
- Contains: Matcher, BERTClassifier, SetFitClassifier, EmbeddingMatcher, TextNormalizer, BlockingStrategy, HierarchicalMatcher, Reranker
- Depends on: sentence-transformers, setfit, transformers, torch, numpy
- Used by: novelty layer, pipeline layer, public API

**Novelty Layer (Detection):**
- Purpose: Novel entity detection, clustering, class proposal
- Location: `src/novelentitymatcher/novelty/`
- Contains: NoveltyDetector, NovelEntityMatcher, novelty strategies (confidence, knn, clustering, etc.), clustering backends, LLM class proposer
- Depends on: core layer, sklearn, hnswlib, faiss, hdbscan, umap, litellm
- Used by: pipeline layer, public API

**Pipeline Layer (Orchestration):**
- Purpose: Staged discovery pipeline, data flow orchestration
- Location: `src/novelentitymatcher/pipeline/`
- Contains: DiscoveryPipeline, PipelineOrchestrator, PipelineStage, stage adapters (OODDetectionStage, ClusterEvidenceStage, etc.)
- Depends on: core layer, novelty layer
- Used by: public API

**Backends Layer (Infrastructure):**
- Purpose: Embedding backends, LLM backends, rerankers
- Location: `src/novelentitymatcher/backends/`
- Contains: SentenceTransformerBackend, StaticEmbeddingBackend, LiteLLMBackend, CrossEncoderReranker
- Depends on: sentence-transformers, model2vec, litellm
- Used by: core layer, novelty layer

**Ingestion Layer (Data Loading):**
- Purpose: Data ingestion for entity corpora
- Location: `src/novelentitymatcher/ingestion/`
- Contains: Fetchers for languages, currencies, industries, timezones, occupations, products, universities
- Depends on: requests, csv, json
- Used by: CLI tools, setup scripts

**Benchmarks Layer (Evaluation):**
- Purpose: Benchmarking entity matching, classification, novelty detection, and async performance
- Location: `src/novelentitymatcher/benchmarks/`
- Contains:
  - `cli.py` — CLI entry point with 10 subcommands (run, load, list, clear, sweep, bench-classifier, bench-novelty, bench-async, render, plot)
  - `runner.py` — BenchmarkRunner orchestrating dataset loading and evaluation
  - `loader.py` — Async HuggingFace dataset loader with parquet caching
  - `registry.py` — Dataset registry, configs, cache configuration
  - `base.py` — BaseEvaluator, EvaluationResult abstractions
  - `shared.py` — Shared utilities (SplitData, timer, compute_ood_metrics, generate_synthetic_data, benchmark_training, benchmark_inference, prepare_binary_labels)
  - `classifier_bench.py` — BERT vs SetFit classifier benchmarking (merged from scripts)
  - `novelty_bench.py` — Novelty strategy benchmarking with quick/standard/full depth levels (merged from scripts)
  - `async_bench.py` — Sync vs async matcher API benchmarking (moved from scripts)
  - `visualization.py` — Markdown rendering and chart generation from benchmark JSON (merged from scripts)
  - `entity_resolution/` — Entity resolution evaluation subpackage
  - `classification/` — Classification evaluation subpackage
  - `novelty/` — Novelty evaluation subpackage
- Depends on: core layer, novelty layer, datasets, huggingface_hub, matplotlib
- Used by: CLI (`novelentitymatcher-bench` entry point)

**Utils Layer (Shared):**
- Purpose: Logging, validation, preprocessing
- Location: `src/novelentitymatcher/utils/`
- Contains: logging_config, validation, preprocessing, embeddings
- Depends on: Standard library, numpy, sklearn
- Used by: All layers

## Data Flow

**Matching Flow (Matcher):**
1. Input text → TextNormalizer (optional)
2. Normalized text → Embedding generation (SentenceTransformers/Static)
3. Embeddings → Classification/SetFit/BERT
4. Output: predicted_id, confidence, candidates

**Novelty-Aware Flow (NovelEntityMatcher):**
1. Input text(s) → Matcher (top-k candidates + embeddings)
2. Novelty detection (multi-signal via ANN index)
3. Optional: Clustering of novel samples
4. Optional: LLM class proposal
5. Output: NovelEntityMatchResult with is_novel flag

**Pipeline Flow (DiscoveryPipeline):**
1. Stage 1: Matcher metadata collection
2. Stage 2: OOD detection (NoveltyDetector)
3. Stage 3: Clustering (ScalableClusterer)
4. Stage 4: Cluster evidence extraction
5. Stage 5: LLM class proposal
6. Output: NovelClassDiscoveryReport

**State Management:**
- Stateless matching (per-request)
- Stateful novelty detection (maintains reference embeddings in ANN index)
- Pipeline stages share immutable StageContext
- Optional persistence via ProposalReviewManager

## Key Abstractions

**Matcher (Facade Pattern):**
- Purpose: Unified API for entity matching with auto mode selection
- Examples: `src/novelentitymatcher/core/matcher.py`, `src/novelentitymatcher/api.py`
- Pattern: Facade with strategy pattern (zero-shot, head-only, full training)

**NoveltyStrategy (Strategy Pattern):**
- Purpose: Pluggable novelty detection algorithms
- Examples: `src/novelentitymatcher/novelty/strategies/`, `src/novelentitymatcher/novelty/strategies/base.py`
- Pattern: Abstract base class with multiple implementations

**PipelineStage (Pipeline Pattern):**
- Purpose: Composable stages in discovery pipeline
- Examples: `src/novelentitymatcher/pipeline/adapters.py`, `src/novelentitymatcher/pipeline/contracts.py`
- Pattern: Ordered list of stages with shared context

**EmbeddingBackend (Adapter Pattern):**
- Purpose: Abstraction over different embedding sources
- Examples: `src/novelentitymatcher/backends/sentencetransformer.py`, `src/novelentitymatcher/backends/static_embedding.py`
- Pattern: Base class with implementations for different providers

## Entry Points

**Matcher Class:**
- Location: `src/novelentitymatcher/core/matcher.py`
- Triggers: Instantiation by user code
- Responsibilities: Entity matching with auto mode selection, async/sync APIs

**NovelEntityMatcher Class:**
- Location: `src/novelentitymatcher/novelty/entity_matcher.py`
- Triggers: Instantiation by user code
- Responsibilities: Novelty-aware matching orchestration, class proposal

**DiscoveryPipeline Class:**
- Location: `src/novelentitymatcher/pipeline/discovery.py`
- Triggers: Instantiation by user code
- Responsibilities: Multi-stage discovery pipeline for novel entities

**CLI Commands:**
- Location: `src/novelentitymatcher/ingestion/cli.py`, `src/novelentitymatcher/benchmarks/cli.py`
- Triggers: Command line invocation via `novelentitymatcher-bench` entry point
- Responsibilities: Data ingestion, benchmark running (10 subcommands: run, load, list, clear, sweep, bench-classifier, bench-novelty, bench-async, render, plot)

## Error Handling

**Strategy:** Custom exception hierarchy with descriptive types

**Patterns:**
- SemanticMatcherError - Base exception
- ValidationError - Input validation failures
- TrainingError - Training process failures
- MatchingError - Matching process failures
- ModeError - Mode/configuration errors
- Logging via custom logger with configurable verbosity

## Cross-Cutting Concerns

**Logging:** Centralized via `novelentitymatcher.utils.logging_config` with DEBUG/INFO/WARNING/ERROR levels, optional file handler, environment-controlled verbosity

**Validation:** Pydantic models for configuration, custom validation functions in `utils/validation.py`

**Async Support:** Async-first design with AsyncExecutor in `core/async_utils.py`, pytest-asyncio for testing

**Configuration:** Pydantic-based config in `novelty/config/`, model registry in `config_registry.py`

---

*Architecture analysis: 2026-04-23*

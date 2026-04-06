# Architecture

**Analysis Date:** 2026-04-06

## Pattern Overview

**Overall:** Layered Architecture with Strategy Pattern and Pipeline Orchestration

**Key Characteristics:**
- Multi-layered design separating core matching, novelty detection, and pipeline orchestration
- Strategy pattern for pluggable novelty detection algorithms (10+ strategies)
- Pipeline pattern with staged processing via `PipelineOrchestrator` and `PipelineStage` contracts
- Backend abstraction for interchangeable embedding/model providers
- Lazy-loaded public API via `__getattr__` in `__init__.py`

## Layers

**Public API Layer:**
- Purpose: Single import surface for all public classes and functions
- Location: `src/novelentitymatcher/__init__.py` and `src/novelentitymatcher/api.py`
- Contains: Re-exported classes, lazy-loaded modules, version info
- Depends on: All internal layers
- Used by: External consumers, CLI entry points

**Orchestration Layer:**
- Purpose: High-level workflow coordination (matching → novelty detection → class proposal)
- Location: `src/novelentitymatcher/novelty/entity_matcher.py`, `src/novelentitymatcher/pipeline/`
- Contains: `NovelEntityMatcher`, `DiscoveryPipeline`, `PipelineOrchestrator`
- Depends on: Core matching, novelty detection, clustering, proposal
- Used by: Public API, CLI tools

**Core Matching Layer:**
- Purpose: Entity matching algorithms and supporting components
- Location: `src/novelentitymatcher/core/`
- Contains: `Matcher`, `SetFitClassifier`, `BERTClassifier`, `EmbeddingMatcher`, `TextNormalizer`, `CrossEncoderReranker`, `BlockingStrategy` variants, `HierarchicalMatcher`
- Depends on: Backends, config, utils
- Used by: Orchestration layer, novelty layer

**Novelty Detection Layer:**
- Purpose: Detect novel/unknown entities not in the known entity set
- Location: `src/novelentitymatcher/novelty/`
- Contains: `NoveltyDetector`, 10+ strategy implementations, clustering backends, LLM class proposer, storage, schemas
- Depends on: Core matching (for embeddings), config, backends
- Used by: Orchestration layer

**Backend Layer:**
- Purpose: Abstract embedding/model providers for interchangeability
- Location: `src/novelentitymatcher/backends/`
- Contains: `SentenceTransformerBackend`, `LiteLLMBackend`, `RerankerSTBackend`, `StaticEmbeddingBackend`, base backend class
- Depends on: External ML libraries (sentence-transformers, litellm, etc.)
- Used by: Core matching layer

**Configuration Layer:**
- Purpose: Centralized config loading, model registry, and resolution
- Location: `src/novelentitymatcher/config.py`, `src/novelentitymatcher/config_registry.py`, `src/novelentitymatcher/novelty/config/`
- Contains: `Config` class, model registries, detection configs, strategy configs
- Depends on: PyYAML, JSON data files
- Used by: All layers

**Utility Layer:**
- Purpose: Shared helpers for logging, validation, benchmarking, preprocessing
- Location: `src/novelentitymatcher/utils/`
- Contains: Logging config, validation, preprocessing, benchmark helpers, embedding utils
- Depends on: Standard library, external packages
- Used by: All layers

## Data Flow

**Known-Entity Matching Flow:**
1. User creates `Matcher` with known entities and optional model/threshold config
2. Input text is normalized via `TextNormalizer` (optional)
3. Text is embedded using configured backend (SentenceTransformer, LiteLLM, etc.)
4. Embeddings are compared against reference entity embeddings via cosine similarity
5. Top-k candidates are ranked and filtered by threshold
6. Optional: SetFit/BERT classifier refines predictions if trained
7. Optional: CrossEncoder reranker re-scores top candidates
8. `MatchResultWithMetadata` is returned with predictions, scores, and alternatives

**Novelty-Aware Matching Flow:**
1. `NovelEntityMatcher` wraps a fitted `Matcher` + `NoveltyDetector` + optional `LLMClassProposer`
2. Matcher produces rich metadata (embeddings, top-k candidates)
3. `NoveltyDetector` scores novelty using multiple registered strategies (confidence, KNN, clustering, etc.)
4. `SignalCombiner` aggregates strategy signals into a novelty decision
5. If novel, `LLMClassProposer` optionally suggests new class names
6. `NovelEntityMatchResult` is returned with match status, novelty status, and signals

**Pipeline Discovery Flow:**
1. `DiscoveryPipeline` is configured with entities and `PipelineConfig`
2. `PipelineOrchestrator` runs stages sequentially against shared `StageContext`:
   - `MatcherMetadataStage`: Run matcher, collect embeddings and candidates
   - `OODDetectionStage`: Score out-of-distribution novelty
   - `CommunityDetectionStage`: Cluster novel samples
   - `ClusterEvidenceStage`: Build evidence for each cluster
   - `ProposalStage`: Propose class names for novel clusters
3. `PipelineRunResult` aggregates all stage results
4. Results can be reviewed, promoted, and persisted via `ProposalReviewManager`

**State Management:**
- `Matcher`: Holds entities, classifier, reference embeddings in memory
- `NoveltyDetector`: Holds initialized strategies and reference embedding signature
- `DiscoveryPipeline`: Owns all components, manages lifecycle
- Persistence: Proposals and review records saved to JSON files via `storage/persistence.py` and `storage/review.py`

## Key Abstractions

**Matcher (`src/novelentitymatcher/core/matcher.py`):**
- Purpose: Primary entity matching engine with multiple modes (zero-shot, head-only, full, hybrid, auto)
- Examples: `Matcher`, `_EntityMatcher`
- Pattern: Facade over classifier, embedding, reranker, blocking components

**NoveltyStrategy (`src/novelentitymatcher/novelty/strategies/base.py`):**
- Purpose: Abstract interface for novelty detection algorithms
- Examples: `ConfidenceStrategy`, `KNNDistanceStrategy`, `ClusteringStrategy`, `OneClassStrategy`, `PrototypicalStrategy`, `SetFitStrategy`, `SelfKnowledgeStrategy`, `PatternStrategy`, `LOFStrategy`, `MahalanobisDistanceStrategy`, `UncertaintyStrategy`
- Pattern: Strategy pattern with registry-based instantiation

**PipelineStage (`src/novelentitymatcher/pipeline/contracts.py`):**
- Purpose: Abstract interface for pipeline processing stages
- Examples: `MatcherMetadataStage`, `OODDetectionStage`, `CommunityDetectionStage`, `ClusterEvidenceStage`, `ProposalStage`
- Pattern: Chain of Responsibility / Pipeline with shared context

**ClusteringBackend (`src/novelentitymatcher/novelty/clustering/base.py`):**
- Purpose: Abstract interface for clustering algorithms
- Examples: `HDBSCANBackend`, `SOPTICSBackend`, `UMAPHDBSCANBackend`
- Pattern: Strategy pattern with registry

**Backend (`src/novelentitymatcher/backends/base.py`):**
- Purpose: Abstract interface for embedding/model providers
- Examples: `SentenceTransformerBackend`, `LiteLLMBackend`, `RerankerSTBackend`, `StaticEmbeddingBackend`
- Pattern: Strategy pattern

## Entry Points

**Package Import (`src/novelentitymatcher/__init__.py`):**
- Location: `src/novelentitymatcher/__init__.py`
- Triggers: `import novelentitymatcher` or `from novelentitymatcher import ...`
- Responsibilities: Configure logging, expose public API via lazy loading

**CLI: Ingestion (`src/novelentitymatcher/ingestion/cli.py`):**
- Location: `src/novelentitymatcher/ingestion/cli.py`
- Triggers: `novelentitymatcher-ingest` command
- Responsibilities: Ingest reference data (currencies, industries, languages, occupations, products, timezones, universities)

**CLI: Benchmarking (`src/novelentitymatcher/benchmarks/cli.py`):**
- Location: `src/novelentitymatcher/benchmarks/cli.py`
- Triggers: `novelentitymatcher-bench` command
- Responsibilities: Run benchmark suites for classification, entity resolution, novelty

**CLI: Novelty Review (`src/novelentitymatcher/novelty/cli.py`):**
- Location: `src/novelentitymatcher/novelty/cli.py`
- Triggers: `novelentitymatcher-review` command
- Responsibilities: Review and manage novelty class proposals

## Error Handling

**Strategy:** Custom exception hierarchy with contextual information and suggestions

**Patterns:**
- `SemanticMatcherError`: Base exception for all package errors
- `ValidationError`: Input validation failures with entity context, field info, and fix suggestions
- `TrainingError`: Training failures with mode and diagnostic details
- `MatchingError`: Matching operation failures
- `ModeError`: Invalid mode configuration with valid mode suggestions

## Cross-Cutting Concerns

**Logging:** Centralized via `utils/logging_config.py`, configured at package import time, respects `NOVEL_ENTITY_MATCHER_VERBOSE` env var, uses structured logger per module

**Validation:** Dedicated `utils/validation.py` for entities, models, thresholds; Pydantic-style dataclasses in `novelty/schemas/`; config validation in `novelty/config/base.py`

**Authentication:** Not applicable (library package, no auth required); LLM provider API keys managed via LiteLLM env vars

---

*Architecture analysis: 2026-04-06*

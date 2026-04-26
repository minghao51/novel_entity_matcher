# Architecture

Related docs: [`index.md`](./index.md) | [`quickstart.md`](./quickstart.md)

## Overview

Novel Entity Matcher is a text-to-entity matching library built on SetFit few-shot learning with Sentence Transformers, with automatic novel entity detection and classification.
Use this page for internals and module boundaries rather than first-run usage.

## Module Structure

```text
src/novelentitymatcher/
├── __init__.py              # Public exports / lazy import surface
├── config.py                # Config loading and defaults
├── core/                    # Matching pipelines and domain logic
│   ├── matcher.py           # EntityMatcher / EmbeddingMatcher / Matcher
│   ├── classifier.py        # SetFitClassifier wrapper
│   ├── bert_classifier.py   # BERTClassifier wrapper
│   ├── normalizer.py        # Text normalization
│   ├── blocking.py          # Candidate blocking strategies
│   ├── reranker.py          # Cross-encoder reranking
│   ├── hybrid.py            # Multi-stage matching pipeline
│   └── monitoring.py        # Metrics/monitoring helpers
├── backends/                # Provider integrations (embeddings/reranking)
│   ├── base.py              # Backend interfaces / shared abstractions
│   ├── static_embedding.py  # Static embedding backend (model2vec, StaticEmbedding)
│   ├── sentencetransformer.py
│   ├── reranker_st.py
│   ├── litellm.py           # Planned/in-progress cloud backend support
│   └── ...
├── pipeline/                # Internal stage contracts and orchestrator
│   ├── contracts.py         # StageContext / StageResult / PipelineStage
│   ├── orchestrator.py      # Ordered stage execution
│   └── adapters.py          # Wrappers around matcher, OOD, and proposal flows
├── novelty/                 # Novelty detection, discovery, and persistence
│   ├── entity_matcher.py    # Public novelty-aware API
│   ├── match_result.py      # Stable metadata/result contract for discovery work
│   ├── core/
│   ├── proposal/
│   ├── schemas/
│   └── storage/
├── ingestion/               # Dataset ingestion and normalization CLI/pipelines
│   ├── cli.py               # `novelentitymatcher-ingest` entrypoint target
│   └── *.py                 # Source-specific ingestors (countries/products/etc.)
├── utils/                   # Cross-cutting helpers (non-domain specific)
└── data/                    # Packaged static data files / defaults
```

## Package Boundaries

- `core/`: orchestration and domain logic for matching, retrieval/reranking pipelines, and normalization.
- `pipeline/`: internal stage-oriented contracts used to compose discovery flows without exposing unstable stage APIs yet.
- `novelty/`: novelty detection, discovery reports, proposal generation, and persistence.
- `backends/`: provider-specific integrations for embeddings and rerankers (Hugging Face, LiteLLM, etc.).
- `ingestion/`: data acquisition/transformation utilities and the ingestion CLI.
- `utils/`: shared helpers used across modules that are not themselves product/domain entrypoints.
- `data/`: packaged JSON/static assets required at runtime.

## Module Placement Rules

- Add a new matcher or pipeline stage to `core/` unless it is provider-specific.
- Add internal discovery-stage contracts or orchestrators to `pipeline/`.
- Add a new model/provider integration to `backends/`.
- Add novelty scoring, proposal generation, and review/persistence logic to `novelty/`.
- Add dataset import/transformation logic or CLI wiring to `ingestion/`.
- Put generic helpers in `utils/`; avoid moving domain logic there just to “reuse” it.
- Keep the public import surface curated through `src/novelentitymatcher/__init__.py` (avoid exposing internal modules unintentionally).

## Core Components

### Matcher (Unified API)

The recommended `Matcher` class with smart auto-selection of the optimal matching strategy.

**Modes:**
- `zero-shot`: Embedding similarity without training
- `head-only`: Lightweight SetFit training (~30s)
- `full`: Full SetFit training (~3min)
- `bert`: BERT-based classifier (~5min, high accuracy)
- `hybrid`: Multi-stage pipeline (blocking → retrieval → reranking)
- `auto`: Auto-detects based on training data volume

**Workflow:**
```python
matcher = Matcher(entities=[...], mode="auto")
matcher.fit(training_data=None)  # Auto-selects mode
result = matcher.match("query")   # Routes to appropriate strategy
```

**Auto-selection Rules:**
- No training data → zero-shot mode
- < 3 examples per entity → head-only mode
- ≥ 3 examples per entity, < 100 total → full training mode
- ≥ 100 total, ≥ 8 examples per entity → bert mode

### EntityMatcher (Deprecated)

SetFit-based entity matching with optional text normalization.

**Workflow:**
1. Initialize with entities list (id, name, aliases)
2. Train with labeled examples
3. Predict entity ID for new inputs

```python
matcher = EntityMatcher(entities=[
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]}
])
matcher.train(training_data)
result = matcher.predict("Deutschland")  # → "DE"
```

### EmbeddingMatcher

Similarity-based matching without training. Uses cosine similarity between embeddings.

**Workflow:**
1. Initialize with entities
2. Build index (encodes all names/aliases)
3. Match queries against index

```python
matcher = EmbeddingMatcher(entities=[...])
matcher.build_index()
result = matcher.match("Deutschland")  # → {"id": "DE", "score": 0.92}
```

### SetFitClassifier

Low-level wrapper around SetFit for training and prediction.

### BERTClassifier

Low-level wrapper around transformers library for BERT-based classification.

### TextNormalizer

Text preprocessing with options for:
- Lowercase conversion
- Accent removal
- Punctuation removal

## Data Flow

```
Input Text
    ↓
TextNormalizer (optional)
    ↓
Matcher metadata collection
    ↓
Known-entity routing / top-k candidates
    ↓
OOD / novelty detection
    ↓
Optional class proposal generation
    ↓
Result (entity match, novelty report, and optional proposals)
```

## Discovery Pipeline

`DiscoveryPipeline` is the pipeline-first public discovery entry point. It owns and wires together:

- a fitted `Matcher` for top-k match metadata and embeddings
- a `NoveltyDetector` built from `PipelineConfig`
- a `ScalableClusterer` configured from pipeline clustering settings
- an optional `LLMClassProposer` for class proposal generation

The internal stage sequence is:

1. `MatcherMetadataStage`
2. `OODDetectionStage`
3. `CommunityDetectionStage`
4. `ClusterEvidenceStage`
5. `ProposalStage`

Key stage-level behaviors:

- OOD strategy and calibration settings are resolved before stage execution and applied through the owned `DetectionConfig`
- clustering metric / density parameters are applied both to the owned `ScalableClusterer` and the cluster stage runtime call
- proposal generation can run in cluster mode or sample mode
- schema discovery augments proposals with discovered attributes and a normalized `attribute_schema`

This keeps `DiscoveryPipeline` as the richest configuration surface for discovery workflows, while `NovelEntityMatcher` remains the simpler novelty-aware classification entry point.

## Backends

### Static Embeddings

Fast retrieval-oriented embeddings using pre-computed lookups.

**Supports two approaches:**
- **model2vec** (`StaticModel`): minishlab potion models (potion-8m, potion-32m)
- **StaticEmbedding** (sentence-transformers): RikkaBotan MRL models

**Benefits:**
- 10-100x faster than dynamic embeddings
- Lower memory usage
- Sufficient accuracy for retrieval scenarios

**Usage:**
```python
matcher = Matcher(entities=[...], model="potion-8m")  # Static by default
```

### HuggingFace (SentenceTransformers)

- `HFEmbedding` - Generate embeddings
- `HFReranker` - Cross-encoder reranking

### LiteLLM (future)

- Cloud LLM embedding support (planned/in progress; confirm implementation status before relying on it)

### Ollama (future)

- Local LLM embeddings (planned; may not be fully wired in current release)

## Model Registries

### MODEL_SPECS

Central registry of model specifications with metadata:
- **Static models**: potion-8m, potion-32m, mrl-en, mrl-multi
- **Dynamic models**: bge-base, bge-m3, nomic, mpnet, minilm
- **Training support**: Marks which models can be used for SetFit training

### Resolution Logic

- `resolve_model_alias()`: Maps short aliases to full model names
- `is_static_embedding_model()`: Detects static embedding models
- `resolve_training_model_alias()`: Falls back to training-safe models

### Default Models

- **Retrieval default**: `potion-8m` (fast static embeddings)
- **Training default**: `mpnet` (SetFit-compatible)

## Matcher Mode System

### MATCHER_MODE_REGISTRY

Maps mode names to implementation classes:
- `zero-shot` → `EmbeddingMatcher`
- `head-only` → `EntityMatcher` (lightweight training)
- `full` → `EntityMatcher` (full training)
- `hybrid` → `HybridMatcher` (multi-stage pipeline)
- `auto` → `SmartSelection` (runtime detection)

### Mode Selection Process

1. User specifies mode (or uses `auto`)
2. Matcher routes to appropriate implementation
3. Training requests with static models auto-fallback to training-safe backbone
4. Hybrid mode uses blocking → retrieval → reranking pipeline

## Design Decisions

1. **Optional normalization** - Users can disable if input is already clean
2. **Lazy model loading** - SentenceTransformer loaded on first use
3. **Flexible input** - Single string or list of strings for batch prediction
4. **Threshold-based matching** - Configurable confidence threshold for EmbeddingMatcher

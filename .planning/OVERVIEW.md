# novel-entity-matcher — Overview

## Architecture

**Pattern description:** Text-to-entity matching with SetFit few-shot learning — a library (not a service) that matches input text to a known entity catalog, detects novel/unseen classes via multi-strategy OOD detection, and proposes new entity classes via LLM.

```
┌─────────────┐     ┌──────────────────────────────────┐     ┌───────────────────────┐
│  User Text   │ ──▶ │         Matcher (Core)           │ ──▶ │    MatchResult         │
│             │     │  ┌───────────┐  ┌──────────────┐  │     │  (predictions, scores) │
│             │     │  │ Embedding  │  │ Blocking +    │  │     └───────────┬───────────┘
│             │     │  │ Matcher   │  │ Reranker      │  │                 │
│             │     │  └───────────┘  └──────────────┘  │                 ▼
│             │     │  ┌───────────┐  ┌──────────────┐  │     ┌───────────────────────┐
│             │     │  │ SetFit    │  │ CrossEncoder  │  │ ──▶ │  NoveltyDetector       │
│             │     │  │Classifier │  │ Reranker      │  │     │  (15 OOD strategies)   │
│             │     │  └───────────┘  └──────────────┘  │     └───────────┬───────────┘
│             │     └──────────────────────────────────┘                 │
│             │                                                          ▼
│             │     ┌──────────────────────────────────┐     ┌───────────────────────┐
│             │     │     DiscoveryPipeline            │     │  Cluster + LLM         │
│             │     │  Match → OOD → Cluster → Propose │ ──▶ │  (new class proposal)  │
│             │     └──────────────────────────────────┘     └───────────────────────┘
└─────────────┘
```

### `novelentitymatcher` — Python 3.10+ (Library)

Layered: **Matcher (known-entity matching) → NoveltyDetector (OOD detection) → DiscoveryPipeline (staged discovery)**

| Layer | Location | Pattern |
|-------|----------|---------|
| Public API | `src/novelentitymatcher/api.py:1` | Re-exports from `__init__.__all__` |
| Entry facade | `src/novelentitymatcher/__init__.py:1` | Lazy `__getattr__` for 30+ public symbols |
| **Core (Matcher)** | `src/novelentitymatcher/core/matcher.py:54` | `Matcher` facade → `EmbeddingMatcher` / `_EntityMatcher` / `HybridMatcher` via `MatcherFacade` |
| Core (Blocking) | `src/novelentitymatcher/core/blocking.py` | Strategy: BM25, TF-IDF, Fuzzy, NoOp |
| Core (Classifier) | `src/novelentitymatcher/core/classifier.py` | `SetFitClassifier` wrapper |
| Core (Reranker) | `src/novelentitymatcher/core/reranker.py` | `CrossEncoderReranker` |
| Backends | `src/novelentitymatcher/backends/` | Adapter: sentence-transformers, static-embedding, BERT, LiteLLM |
| **Novelty Detection** | `src/novelentitymatcher/novelty/core/detector.py:44` | `NoveltyDetector` orchestrates 15+ OOD strategies via `StrategyRegistry` |
| Novelty (Strategies) | `src/novelentitymatcher/novelty/strategies/base.py:14` | `NoveltyStrategy` ABC — confidence, uncertainty, kNN, clustering, LOF, OCSVM, mahalanobis, energy, setfit, prototypical, self-knowledge, pattern, conformal, react_energy, mixture_gaussian |
| Novelty (Signal) | `src/novelentitymatcher/novelty/core/signal_combiner.py` | Weighted fusion of raw OOD scores |
| Novelty (Calibration) | `src/novelentitymatcher/novelty/core/score_calibrator.py` | Per-strategy score normalization to [0,1] |
| Novelty (Clustering) | `src/novelentitymatcher/novelty/clustering/` | Backends: HDBSCAN, UMAP+HDBSCAN, SOPtics; scalable + incremental + graph-based |
| Novelty (LLM Proposal) | `src/novelentitymatcher/novelty/proposal/llm.py` | `LLMClassProposer` via LiteLLM (OpenRouter, Anthropic, OpenAI) |
| **Pipeline** | `src/novelentitymatcher/pipeline/discovery.py:57` | `DiscoveryPipeline` — owned `Matcher` + `NoveltyDetector` + `ScalableClusterer` + `LLMClassProposer` |
| Pipeline (Stages) | `src/novelentitymatcher/pipeline/orchestrator.py` | `PipelineOrchestrator` runs staged pipeline (match → OOD → cluster → evidence → propose) |
| Ingestion | `src/novelentitymatcher/ingestion/` | Entity catalog generators (languages, currencies, occupations, industries, products, timezones, universities) |
| Benchmarks | `src/novelentitymatcher/benchmarks/` | Entity-resolution, classification, novelty benchmarks via HuggingFace datasets |
| Config | `src/novelentitymatcher/config.py:346` | `Config` — YAML/JSON loader with fallback to `config.yaml` |
| Monitoring | `src/novelentitymatcher/monitoring/` | Performance tracking + metrics collection |
| Exceptions | `src/novelentitymatcher/exceptions.py` | `SemanticMatcherError`, `ValidationError`, `TrainingError`, `MatchingError`, `ModeError` |

**Entry point**: `novelentitymatcher-ingest` → `src/novelentitymatcher/ingestion/cli.py:34` (argparse CLI for entity ingestion)
**Entry point**: `novelentitymatcher-bench` → `src/novelentitymatcher/benchmarks/cli.py:1` (argparse CLI for HuggingFace benchmarks)
**Entry point**: `novelentitymatcher-review` → `src/novelentitymatcher/novelty/cli.py:1` (argparse CLI for HITL proposal review)

## Key Data Flows

**Known-entity matching (single text):** `query text` → `Matcher.match()` → Embedding lookup + Blocking + SetFit/BERT/hybrid classification + CrossEncoder reranking → `MatchResult` (predictions, confidences)

**Novelty detection (OOD):** `Matcher` outputs (embeddings, confidences, classes) + `reference_embeddings` → `NoveltyDetector.detect_novel_samples()` → 15 parallel OOD strategies → `SignalCombiner` weighted fusion → `OODScoreCalibrator` → `NovelSampleReport` (per-sample novelty scores + flagged indices)

**Full discovery (pipeline):** `queries` → `DiscoveryPipeline.discover()` → Stage:process → Stage:match → Stage:OOD detect → Stage:cluster (novel groups) → Stage:extract evidence → Stage:LLM propose → `NovelClassDiscoveryReport` + `ProposalReviewRecord`

**Human-in-the-loop:** Pipeline emits `ProposalReviewRecord` → `novelentitymatcher-review` CLI for `list` / `approve` / `reject` / `promote` workflows

**Benchmarking:** `novelentitymatcher-bench` → HuggingFace dataset loader → model evaluation (entity resolution, classification, novelty detection) → metrics CSV output

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python ≥3.10 | Runtime |
| Package mgmt | uv + hatchling | Build, dependency, venv |
| Embedding models | sentence-transformers ≥3.0 | Sentence encoding |
| Few-shot learning | setfit ≥1.0 | SetFit classifier fine-tuning |
| Static embeddings | model2vec ≥0.1 | Potion-base, MRL static models |
| Deep learning | torch ≥2.0 + transformers ≥4.45 | Neural backends for BERT/SetFit |
| ML / clustering | scikit-learn ≥1.3, HDBSCAN, UMAP | Classic ML + clustering |
| ANN search | hnswlib ≥0.8, faiss-cpu ≥1.7 | Approximate nearest neighbor |
| Vector store | chromadb ≥0.5 | Optional persistent vector store |
| OOD strategies | numpy, scipy, sklearn native | 15 OOD algorithms (kNN, LOF, OCSVM, Mahalanobis, GMM, energy, conformal, etc.) |
| LLM gateway | litellm ≥1.83 | Unified API for 100+ LLM providers |
| LLM framework | dspy ≥3.2 | Prompt optimization, programmatic LLM |
| Graph clustering | python-igraph, leidenalg, networkx | Graph-based clustering |
| Circuit breaker | aiobreaker ≥1.1 | LLM call resilience |
| Retry | tenacity ≥9.0 | Transient error retry |
| Config | pyyaml ≥6.0, pydantic ≥2.0, pydantic-settings ≥2.14 | YAML + Pydantic config models |
| Data | numpy ≥2.0, pandas ≥2.0 | Array processing, data frames |
| Fuzzy matching | rapidfuzz ≥3.0, rank-bm25 ≥0.2 | String similarity, BM25 blocking |
| HP optimization | optuna ≥4.8 | Strategy weight tuning |
| Viz | matplotlib ≥3.9, seaborn ≥0.13 | Benchmark charts |
| Async | asyncio (stdlib) + pytest-asyncio | Async matcher/pipeline support |
| Testing | pytest ≥9.0, pytest-cov ≥6.2 | Test runner + coverage |
| Linting | ruff ≥0.1 | Lint + format (88 chars, double quotes) |
| Type checking | mypy ≥1.19 | Static type checking |
| Pre-commit | pre-commit ≥3.6 | Git hook automation |
| CI | GitHub Actions (`.github/`) | CI/CD |
| Docs | mkdocs, mkdocs-material, mkdocstrings | API docs |

## Infrastructure

No deployed service infrastructure. Operates as a standalone Python library. All data stored in-process (config via `config.yaml`, review records via JSON files at `./proposals/`, benchmarks output CSVs).

## Integrations

| Service | SDK | Purpose | Status |
|---------|-----|---------|--------|
| HuggingFace Hub | `datasets`, `sentence-transformers`, `transformers` | Model loading, benchmark datasets | Core |
| OpenRouter | `litellm` | LLM class proposal (primary) | Configurable |
| Anthropic | `litellm` | LLM class proposal | Configurable |
| OpenAI | `litellm` | LLM class proposal | Configurable |
| ChromaDB | `chromadb` | Optional vector store for `opinion`/`novelty` extras | Optional |

### Auth Flow

LLM API keys read from environment variables (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`). No other auth.

## Environment Variables

| Variable | Context | Purpose |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | LLM | OpenRouter API key |
| `ANTHROPIC_API_KEY` | LLM | Anthropic API key |
| `OPENAI_API_KEY` | LLM | OpenAI API key |
| `LLM_CLASS_PROPOSER_PROVIDER` | LLM | Provider (openrouter/anthropic/openai) |
| `LLM_CLASS_PROPOSER_MODEL` | LLM | Model name |
| `LLM_TIMEOUT` | LLM | Request timeout in s (default 30) |
| `LLM_MAX_RETRIES` | LLM | Retry count (default 5) |
| `LLM_CIRCUIT_FAIL_MAX` | LLM | Circuit breaker threshold (default 3) |
| `LLM_CIRCUIT_RESET_SECONDS` | LLM | Circuit breaker cooldown (default 60) |
| `HF_TOKEN` | Models | HuggingFace token (optional) |
| `NOVEL_ENTITY_MATCHER_VERBOSE` | Logging | Enable verbose logging (default false) |
| `PYTORCH_ENABLE_MPS_FALLBACK` | Runtime | Set in `matcher.py:17` for Apple Silicon |

# Codebase Structure

**Analysis Date:** 2026-04-23

## Directory Layout

```
novel_entity_matcher/
├── src/novelentitymatcher/     # Main package source
│   ├── core/                   # Core matching functionality
│   ├── novelty/                # Novelty detection and clustering
│   │   ├── core/              # Novelty detector core
│   │   ├── strategies/        # Novelty detection strategies
│   │   ├── clustering/        # Clustering backends
│   │   ├── config/            # Pydantic configuration models
│   │   ├── extraction/        # Cluster evidence extraction
│   │   ├── proposal/          # LLM class proposal
│   │   ├── storage/           # Persistence and review management
│   │   ├── schemas/           # Pydantic schemas for results
│   │   └── evaluation/        # Evaluation utilities
│   ├── pipeline/              # Discovery pipeline orchestration
│   ├── backends/              # Embedding and LLM backends
│   ├── ingestion/             # Data ingestion scripts
│   ├── utils/                 # Shared utilities
│   ├── monitoring/            # Monitoring and metrics
│   ├── benchmarks/            # Benchmark framework
│   │   ├── entity_resolution/ # Entity resolution eval
│   │   ├── classification/    # Classification eval
│   │   └── novelty/           # Novelty detection eval
│   ├── data/                  # Data fixtures and samples
│   ├── config.py              # Main config and model registry
│   ├── config_registry.py     # Dynamic model/configuration registry
│   ├── api.py                 # Public API surface
│   ├── exceptions.py          # Custom exceptions
│   └── __init__.py            # Package exports
├── tests/                     # Test suite
│   ├── unit/                  # Fast isolated tests
│   │   ├── core/
│   │   ├── novelty/
│   │   ├── pipeline/
│   │   ├── backends/
│   │   ├── utils/
│   │   └── ingestion/
│   ├── integration/           # Tests with external dependencies
│   │   ├── core/
│   │   ├── backends/
│   │   └── utils/
│   ├── fixtures/              # Test data fixtures
│   └── conftest.py            # Pytest configuration
├── examples/                  # Usage examples
│   ├── raw/                   # Low-level examples
│   └── *.py                   # Feature-specific examples
├── scripts/                   # Benchmark and utility scripts
├── docs/                      # Documentation
│   ├── methodology/           # Methodological documentation
│   ├── architecture/          # Architecture diagrams
│   ├── experiments/           # Experiment results
│   ├── benchmarks/            # Benchmark results
│   └── archive/               # Archived documentation
├── .github/workflows/         # CI/CD workflows
├── pyproject.toml            # Project configuration
├── AGENTS.md                 # Agent workflow guidelines
└── README.md                 # Project README
```

## Key Locations

**Public API:**
- `src/novelentitymatcher/__init__.py` - Main package exports (lazy loading)
- `src/novelentitymatcher/api.py` - Comprehensive public API surface

**Core Entry Points:**
- `src/novelentitymatcher/core/matcher.py` - Unified Matcher class (1241 lines)
- `src/novelentitymatcher/novelty/entity_matcher.py` - NovelEntityMatcher (566 lines)
- `src/novelentitymatcher/pipeline/discovery.py` - DiscoveryPipeline (597 lines)

**Configuration:**
- `src/novelentitymatcher/config.py` - Main config entry point
- `src/novelentitymatcher/config_registry.py` - Model and mode registry
- `src/novelentitymatcher/novelty/config/` - Novelty detection config

**Backends:**
- `src/novelentitymatcher/backends/sentencetransformer.py` - SentenceTransformer backend
- `src/novelentitymatcher/backends/static_embedding.py` - Static embeddings (model2vec)
- `src/novelentitymatcher/backends/litellm.py` - LLM backend

**Novelty Strategies:**
- `src/novelentitymatcher/novelty/strategies/` - Strategy implementations
- `src/novelentitymatcher/novelty/strategies/base.py` - Base strategy interface

**Clustering:**
- `src/novelentitymatcher/novelty/clustering/backends.py` - HDBSCAN, SOPTICS, UMAP+HDBSCAN
- `src/novelentitymatcher/novelty/clustering/scalable.py` - Scalable clustering

**Testing:**
- `tests/conftest.py` - Shared pytest fixtures
- `tests/unit/` - Unit tests (fast, no external deps)
- `tests/integration/` - Integration tests (requires external services)

## Naming Conventions

**Modules:**
- Lowercase with underscores: `entity_matcher.py`, `signal_combiner.py`
- Impl suffix for implementation details: `self_knowledge_impl.py`, `setfit_impl.py`
- Test modules: `test_*.py` prefix

**Classes:**
- PascalCase: `Matcher`, `NoveltyDetector`, `DiscoveryPipeline`
- Strategy suffix for strategies: `ConfidenceStrategy`, `ClusteringStrategy`
- Config suffix for Pydantic configs: `PipelineConfig`, `DetectionConfig`
- Backend suffix for backends: `SentenceTransformerBackend`, `LiteLLMBackend`

**Functions:**
- snake_case: `match_async`, `fit_async`, `collect_match_result`
- Private prefix: `_ensure_async_executor`, `_get_training_data`

**Constants:**
- UPPER_CASE: `_VALID_OOD_STRATEGIES`, `_VALID_CLUSTERING_BACKENDS`
- Private prefix: `_EXPORTS`, `_coerce_texts`

**Files:**
- Test fixtures: `tests/fixtures/`
- Benchmark data: `artifacts/benchmarks/`
- Checkpoints: `checkpoints/`
- Proposals: `proposals/`

**CLI Commands:**
- `novelentitymatcher-ingest` - Data ingestion CLI
- `novelentitymatcher-bench` - Benchmark CLI
- `novelentitymatcher-review` - Review management CLI

## Package Organization

**Top-level exports (lazy loading):**
- Matcher, SetFitClassifier, TextNormalizer, CrossEncoderReranker
- HierarchicalMatcher, BlockingStrategy variants
- NovelEntityMatcher, NoveltyDetector, LLMClassProposer
- DiscoveryPipeline and pipeline types
- ClusteringBackend and variants
- All exception types
- Config models

**Import patterns:**
- Public API: `from novelentitymatcher import Matcher` or `from novelentitymatcher.api import *`
- Internal: `from ..core.matcher import Matcher` (relative imports)
- External: `from sentence_transformers import SentenceTransformer`

**Lazy imports:**
- Optional features imported on-demand (e.g., LLM, novelty detection)
- TYPE_CHECKING for type hints without runtime dependencies
- `if TYPE_CHECKING:` blocks for circular import avoidance

---

*Structure analysis: 2026-04-23*

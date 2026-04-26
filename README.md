# Novel Entity Matcher

Map messy text to canonical entities with automatic novel entity detection and classification.

**New:** Unified `Matcher` class with smart auto-selection - no need to choose between different matchers!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/novel-entity-matcher)](https://pypi.org/project/novel-entity-matcher/)
[![Python Version](https://img.shields.io/pypi/pyversions/novel-entity-matcher)](https://pypi.org/project/novel-entity-matcher/)

## What It Solves

- Normalize messy entity strings (typos, aliases, alternate names)
- Map text to canonical IDs (for example, country code matching)
- Run locally with Sentence Transformers + SetFit

Example: `"Deutchland"` → `DE`

## Installation

```bash
uv add novel-entity-matcher
```

Optional extras:

```bash
# Novel class detection and ANN-backed discovery
uv add "novel-entity-matcher[novelty]"

# LiteLLM-powered embeddings, reranking, and class proposal features
uv add "novel-entity-matcher[llm]"

# Benchmark visualization scripts
uv add "novel-entity-matcher[viz]"

# Everything
uv add "novel-entity-matcher[all]"
```

If you are not using `uv`, the equivalent `pip` commands still work:

```bash
pip install novel-entity-matcher
pip install "novel-entity-matcher[novelty]"
pip install "novel-entity-matcher[llm]"
pip install "novel-entity-matcher[viz]"
pip install "novel-entity-matcher[all]"
```

## Quick Start

### The New Unified API (Recommended Async Default)

**Single `Matcher` class that auto-detects the best approach, with async shown first:**

```python
import asyncio
from novelentitymatcher import Matcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

async def main():
    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()
        print(await matcher.match_async("America"))  # {"id": "US", "score": 0.95}

        training_data = [
            {"text": "Germany", "label": "DE"},
            {"text": "Deutschland", "label": "DE"},
            {"text": "USA", "label": "US"},
        ]
        await matcher.fit_async(training_data)  # Auto: head-only for <3 examples, full for ≥3
        print(await matcher.match_async("Deutschland"))  # {"id": "DE", "score": 1.0}

asyncio.run(main())
```

Prefer the async API for new integrations, especially in web services, batch jobs, or concurrent workloads. The sync API remains fully supported for scripts and simple one-off usage.

**How it works:**
- No training data → zero-shot (embedding similarity)
- < 3 examples/entity → head-only training (~30s)
- ≥ 3 examples/entity → full training (~3min)

### Sync Alternative

```python
from novelentitymatcher import Matcher

matcher = Matcher(entities=entities)
matcher.fit()
print(matcher.match("America"))
```

### Alternative: Explicit Mode Selection

```python
# Force zero-shot mode
matcher = Matcher(entities=entities, mode="zero-shot")

# Force full training mode
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data)

# Force hybrid mode (blocking + retrieval + reranking)
matcher = Matcher(entities=entities, mode="hybrid")
matcher.fit()
results = matcher.match("America", top_k=3)
```

## Feature Comparison

| Mode | Training | Speed | Best For |
|---|---|---|---|
| `zero-shot` | No | Fast (~50K qps) | Prototyping, simple matching |
| `head-only` | Yes (~30s) | Medium (~30 q/s) | Quick accuracy boost |
| `full` | Yes (~3min) | Medium (~30 q/s) | Production, complex variations |
| `hybrid` | No | Slower, highest precision | Large candidate sets, reranking |
| `bert` | Yes (~5min) | Medium | High-stakes, ≥100 samples/entity |
| `auto` | Varies | Varies | Automatic selection (recommended) |

**The `Matcher` class auto-selects the best mode** based on your training data.

## Novel Entity Detection

Detect when input doesn't match any known entity — critical for production systems that need to handle out-of-distribution inputs.

```python
from novelentitymatcher import Matcher, NovelEntityMatcher
from novelentitymatcher.novelty import DetectionConfig

matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data, num_epochs=4)

novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    detection_config=DetectionConfig(
        strategies=["confidence", "knn_distance", "setfit_centroid"],
    ),
)

result = novel_matcher.match("Unknown entity name")
print(result.is_novel)  # True if not similar to known entities
```

### Detection Strategies (12 available)

| Strategy | AUROC | DR@1% | Best For |
|----------|-------|-------|----------|
| **setfit_centroid** | **0.907** | **14.7%** | Production default, free text |
| **knn_distance** | **0.883** | **10.3%** | Production default, scalable |
| **confidence** | — | — | Baseline, always included |
| oneclass_svm | 0.834 | 14.3% | Boundary detection |
| lof | 0.871 | 6.9% | Varying density |
| pattern | 0.630 | 0.2% | Entity name matching |
| mahalanobis | 0.691 | 2.9% | Gaussian class structure |
| self_knowledge | 0.563 | 1.1% | Embedding reconstruction |
| clustering (HDBSCAN) | — | — | Novel cluster discovery |
| prototypical | 0.507 | 0.6% | Few-shot |
| setfit (contrastive) | 0.452 | 1.7% | Contrastive learning |
| uncertainty | 0.500 | 0.2% | Classifier confusion |

**Default production config:** `["confidence", "knn_distance", "setfit_centroid"]` with weighted combination.

See [Novelty Detection Methods](docs/methodology/novelty-detection/) for full documentation and [Benchmark Results](docs/experiments/20260426-comprehensive-benchmark.md) for detailed numbers.

## Async API

The async API is the recommended default for new code. It provides non-blocking operations with progress tracking and cancellation support:

```python
import asyncio
from novelentitymatcher import Matcher

async def main():
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "US", "name": "United States", "aliases": ["USA"]},
    ]

    # Use async context manager for automatic cleanup
    async with Matcher(entities=entities) as matcher:
        await matcher.fit_async()

        # Batch processing with progress tracking
        async def show_progress(completed, total):
            print(f"Progress: {completed}/{total}")

        results = await matcher.match_batch_async(
            queries=["USA", "Germany"] * 1000,
            batch_size=100,
            on_progress=show_progress
        )

        # Or use concurrent matchers
        async def match_category(category_entities):
            async with Matcher(entities=category_entities) as m:
                await m.fit_async()
                return await m.match_async("query")

        results = await asyncio.gather(
            match_category(entities_1),
            match_category(entities_2),
            match_category(entities_3),
        )

asyncio.run(main())
```

**Key features:**
- `fit_async()`, `match_async()`, `match_batch_async()` - Async versions of core methods
- Progress tracking via `on_progress` callback
- Cancellation support for long-running operations
- Thread-safe concurrent matching
- Sync and async APIs are both supported

See [Async API Guide](docs/async-guide.md) for comprehensive documentation.

## Embedding Models

### Default: Static Embeddings

Novel Entity Matcher uses **static embeddings by default for retrieval**:

```python
# Retrieval default
matcher = Matcher(mode="zero-shot")  # Uses "potion-8m" by default
```

**Benefits:**
- 10-100x faster than dynamic embeddings
- Lower memory usage
- Sufficient accuracy for most use cases

### Available Models

| Alias | Model | Type | Backend | Best For |
|-------|-------|------|---------|----------|
| `potion-8m` | minishlab/potion-base-8M | Static | model2vec | Ultra-fast retrieval |
| `potion-32m` | minishlab/potion-base-32M | Static | model2vec | Fast retrieval (default) |
| `mrl-en` | RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en | Static (MRL) | StaticEmbedding | English with MRL support* |
| `mrl-multi` | sentence-transformers/static-similarity-mrl-multilingual-v1 | Static (MRL) | StaticEmbedding | Multilingual |
| `bge-base` | BAAI/bge-base-en-v1.5 | Dynamic | SentenceTransformer | High accuracy English |
| `bge-m3` | BAAI/bge-m3 | Dynamic | SentenceTransformer | Multilingual |
| `mpnet` | sentence-transformers/all-mpnet-base-v2 | Dynamic | SentenceTransformer | Training default |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | Dynamic | SentenceTransformer | Balanced speed/quality |
| `nomic` | nomic-ai/nomic-embed-text-v1 | Dynamic | SentenceTransformer | Long context |

### BERT Classification Models

| Alias | Model | Speed | Accuracy |
|-------|-------|-------|----------|
| `distilbert` | distilbert-base-uncased | Fast | High (default BERT) |
| `tinybert` | huawei-noah/TinyBERT_General_4L_312D | Very fast | Medium |
| `roberta-base` | roberta-base | Medium | Very high |
| `deberta-v3` | microsoft/deberta-v3-base | Slow | State of the art |
| `bert-multilingual` | bert-base-multilingual-cased | Slow | Multilingual |

### Reranker Models (Hybrid mode)

| Alias | Model | Use Case |
|-------|-------|----------|
| `bge-m3` | BAAI/bge-reranker-v2-m3 | Default, multilingual |
| `bge-large` | BAAI/bge-reranker-large | English, high accuracy |
| `ms-marco` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Fast, lightweight |

*Note: The RikkaBotan MRL model may require MPS fallback on Apple Silicon (set `PYTORCH_ENABLE_MPS_FALLBACK=1`).

### Static Embedding Backends

Novel Entity Matcher supports two static embedding approaches:

1. **StaticEmbedding** (sentence-transformers): For models like RikkaBotan MRL
2. **model2vec**: For minishlab potion models and custom distillations

Both are automatically detected and used based on the model name.

### Training-Safe Default

`Matcher.fit(...)` automatically switches to a training-compatible backbone for
`head-only` and `full` modes. The current training default is `mpnet`, so
`Matcher(model="default")` stays fast for zero-shot retrieval without breaking
SetFit-based training.

### Opting into Dynamic Embeddings

For scenarios requiring contextual understanding:

```python
matcher = Matcher(mode="zero-shot", model="bge-base")
```

### Multilingual Support

```python
matcher = Matcher(mode="zero-shot", model="bge-m3")
```

### Benchmarking

Run benchmarks via the CLI:

```bash
# Novelty strategy benchmark (all 12 strategies)
uv run novelentitymatcher-bench bench-novelty --depth standard

# Multi-model sweep
uv run novelentitymatcher-bench bench-novelty --models minilm mpnet bge-base --depth standard

# BERT vs SetFit comparison
uv run novelentitymatcher-bench bench-classifier --mode compare

# SetFit model sweep
uv run novelentitymatcher-bench bench-classifier --mode sweep-setfit

# Bayesian ensemble weight optimization
uv run novelentitymatcher-bench bench-weights --trials 200 --dataset ag_news

# Parameter sweep (e.g., KNN k values)
uv run novelentitymatcher-bench sweep --task novelty --dataset ag_news --param knn_k

# ANN backend comparison
uv run novelentitymatcher-bench bench-ann --sizes 1000 10000 100000

# Reranker model comparison
uv run novelentitymatcher-bench bench-reranker
```

See [Comprehensive Benchmark Results](docs/experiments/20260426-comprehensive-benchmark.md) for the latest numbers.

## Documentation

- [Documentation Index](docs/index.md) - Organized entry point for guides, experiments, and archive material
- [Quick Start Guide](docs/quickstart.md) - Complete getting started guide
- [Examples Catalog](docs/examples.md) - Maintained runnable examples
- [Methodology](docs/methodology/overview.md) - Pipeline overview, classification modes, and novelty strategies
- [Novelty Detection Methods](docs/methodology/novelty-detection/index.md) - All 12 strategies with math, configs, and benchmarks
- [Classification Methods](docs/methodology/classification/index.md) - All 6 modes with model backends
- [Benchmark Results](docs/experiments/20260426-comprehensive-benchmark.md) - Latest comprehensive benchmarks
- [Troubleshooting](docs/troubleshooting.md) - Common issues and fixes
- [Architecture](docs/architecture.md) - Module layout and design

## Where To Start

1. **New Users**: [Quick Start Guide](docs/quickstart.md)
2. **Working Examples**: [examples/current/basic_matcher.py](examples/current/basic_matcher.py)
3. **Advanced**: [docs/examples.md](docs/examples.md)

## Project Layout

```text
novel_entity_matcher/              # Repository root
├── src/novelentitymatcher/       # Python package
├── examples/                  # Maintained runnable examples
├── experiments/               # Exploratory scripts
├── artifacts/                 # Local generated benchmark outputs
├── tests/                     # Automated tests
├── docs/                      # Documentation
└── pyproject.toml             # Packaging config
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run python -m pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for contributor guidelines.

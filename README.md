# SemanticMatcher

Map messy text to canonical entities using semantic matching.

**New:** Unified `Matcher` class with smart auto-selection - no need to choose between different matchers!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/semantic-matcher)](https://pypi.org/project/semantic-matcher/)
[![Python Version](https://img.shields.io/pypi/pyversions/semantic-matcher)](https://pypi.org/project/semantic-matcher/)

## What It Solves

- Normalize messy entity strings (typos, aliases, alternate names)
- Map text to canonical IDs (for example, country code matching)
- Run locally with Sentence Transformers + SetFit

Example: `"Deutchland"` → `DE`

## Installation

```bash
pip install semantic-matcher
```

## Quick Start

### The New Unified API (Recommended)

**Single `Matcher` class that auto-detects the best approach:**

```python
from semanticmatcher import Matcher

entities = [
    {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
    {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
]

# Zero-shot mode (no training required)
matcher = Matcher(entities=entities)
matcher.fit()
print(matcher.match("America"))  # {"id": "US", "score": 0.95}

# Or with training data (auto-detects training mode)
training_data = [
    {"text": "Germany", "label": "DE"},
    {"text": "Deutschland", "label": "DE"},
    {"text": "USA", "label": "US"},
]
matcher.fit(training_data)  # Auto: head-only for <3 examples, full training for ≥3
print(matcher.match("Deutschland"))  # {"id": "DE", "score": 1.0}
```

**How it works:**
- No training data → zero-shot (embedding similarity)
- < 3 examples/entity → head-only training (~30s)
- ≥ 3 examples/entity → full training (~3min)

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
|---|---|---|---|---|
| `zero-shot` | No | Fast (~50 q/s) | Prototyping, simple matching |
| `head-only` | Yes (~30s) | Medium (~30 q/s) | Quick accuracy boost |
| `full` | Yes (~3min) | Medium (~30 q/s) | Production, complex variations |
| `hybrid` | No | Slower, highest precision | Large candidate sets, reranking |

**The new `Matcher` class auto-selects the best mode** based on your training data.

## Embedding Models

### Default: Static Embeddings

Semantic Matcher uses **static embeddings by default for retrieval**:

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
| `potion-8m` | minishlab/potion-base-8M | Static | model2vec | English general use (retrieval default) |
| `potion-32m` | minishlab/potion-base-32M | Static | model2vec | Fast English with better quality |
| `mrl-en` | RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en | Static (MRL) | StaticEmbedding | English with MRL support* |
| `mrl-multi` | sentence-transformers/static-similarity-mrl-multilingual-v1 | Static (MRL) | StaticEmbedding | Multilingual (NEW!) |
| `bge-m3` | BAAI/bge-m3 | Dynamic | SentenceTransformer | Multilingual (opt-in dynamic) |
| `bge-base` | BAAI/bge-base-en-v1.5 | Dynamic | SentenceTransformer | High accuracy (English) |

*Note: The RikkaBotan MRL model may require MPS fallback on Apple Silicon (set `PYTORCH_ENABLE_MPS_FALLBACK=1`).

### Static Embedding Backends

Semantic Matcher supports two static embedding approaches:

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

Run the comprehensive benchmark suite with:

```bash
uv run python scripts/benchmark_embeddings.py --track all --output artifacts/benchmarks/benchmark-results.json
```

## Documentation

- [Documentation Index](docs/index.md) - Organized entry point for guides, experiments, and archive material
- [Quick Start Guide](docs/quickstart.md) - Complete getting started guide
- [Examples Catalog](docs/examples.md) - Current, legacy, and raw examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues and fixes
- [Architecture](docs/architecture.md) - Module layout and design

## Where To Start

1. **New Users**: [Quick Start Guide](docs/quickstart.md)
2. **Working Examples**: [examples/current/basic_matcher.py](examples/current/basic_matcher.py)
3. **Advanced**: [docs/examples.md](docs/examples.md)

## Project Layout

```text
semantic_matcher/              # Repository root
├── src/semanticmatcher/       # Python package
├── examples/                  # Current, legacy, and raw runnable examples
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

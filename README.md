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
```

## Feature Comparison

| Mode | Training | Speed | Best For |
|---|---|---|---|---|
| `zero-shot` | No | Fast (~50 q/s) | Prototyping, simple matching |
| `head-only` | Yes (~30s) | Medium (~30 q/s) | Quick accuracy boost |
| `full` | Yes (~3min) | Medium (~30 q/s) | Production, complex variations |

**The new `Matcher` class auto-selects the best mode** based on your training data.

## Documentation

- [Migration Guide](docs/migration-guide.md) - Migrate from old API to unified Matcher
- [Quick Start Guide](docs/quickstart.md) - Complete getting started guide
- [Examples Catalog](docs/examples.md) - All examples with difficulty ratings
- [Troubleshooting](docs/troubleshooting.md) - Common issues and fixes
- [Architecture](docs/architecture.md) - Module layout and design

## Where To Start

1. **New Users**: [Quick Start Guide](docs/quickstart.md)
2. **Working Examples**: [examples/embedding_matcher_demo.py](examples/embedding_matcher_demo.py)
3. **Advanced**: [docs/examples.md](docs/examples.md)

## Project Layout

```text
semantic_matcher/              # Repository root
├── src/semanticmatcher/       # Python package
├── examples/                  # Runnable examples (wrapper API)
├── experiments/               # Exploratory scripts
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

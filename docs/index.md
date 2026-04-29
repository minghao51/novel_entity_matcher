# Novel Entity Matcher

Map messy text to canonical entities with automatic novel entity detection and classification.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/minghao51/novel_entity_matcher/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/novel-entity-matcher)](https://pypi.org/project/novel-entity-matcher/)
[![Python Version](https://img.shields.io/pypi/pyversions/novel-entity-matcher)](https://pypi.org/project/novel-entity-matcher/)

## What It Solves

- **Normalize** messy entity strings (typos, aliases, alternate names)
- **Map** text to canonical IDs (e.g. country code matching)
- **Detect** novel entities not present in your known classes
- **Discover** and propose new entity categories automatically
- Run **locally** with Sentence Transformers + SetFit — no cloud API required

Example: `"Deutchland"` → `DE`

## Quick Start

```python
import asyncio
from novelentitymatcher import Matcher

matcher = Matcher(["US", "CA", "DE", "FR", "JP"])
results = asyncio.run(matcher("Deutchland"))
# → MatchResult(id="DE", score=0.92)
```

## Key Features

- **Unified `Matcher` class** — auto-selects between zero-shot, SetFit, BERT, and hybrid modes
- **Novelty Detection** — identifies entities that don't match any known class using kNN, clustering, and statistical strategies
- **Discovery Pipeline** — staged processing with novel class proposal via LLM or heuristic methods
- **Blocking & Reranking** — BM25, TF-IDF, and fuzzy blocking with cross-encoder reranking for scalability
- **Hierarchical Matching** — tree-aware entity resolution with configurable depth and pruning
- **Async API** — high-throughput matching with `async/await` for batch workloads
- **Multiple Backends** — Sentence Transformers, LiteLLM, and static embeddings (Model2Vec)

## Where to Go Next

<div class="grid cards" markdown>

-   :rocket: **Getting Started**
    [Quickstart guide](quickstart.md) — install, create a matcher, and run your first match

-   :books: **API Reference**
    [Auto-generated docs](api/index.md) — full API documentation from source docstrings

-   :bulb: **Guides**
    [Async API](async-guide.md) · [Configuration](configuration.md) · [Models](models.md) · [Matcher Modes](matcher-modes.md)

-   :test_tube: **Experiments**
    [Benchmarking](experiments/index.md) — reproduce results and run your own benchmarks

-   :gear: **Architecture**
    [Internals](architecture.md) — module layout, design decisions, and extension points

-   :map: **Roadmap**
    [Technical Roadmap](technical-roadmap.md) — active development plan and upcoming features

</div>

## Installation

```bash
uv add novel-entity-matcher
```

Optional extras for novelty detection, LLM features, and visualization:

```bash
uv add "novel-entity-matcher[novelty]"
uv add "novel-entity-matcher[llm]"
uv add "novel-entity-matcher[viz]"
```

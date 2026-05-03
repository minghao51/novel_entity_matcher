# Notebooks

Interactive marimo notebooks for entity matching, novelty detection, and benchmarking.

**Source of truth:** [`notebooks/*.py`](https://github.com/minghao51/novel_entity_matcher/tree/main/notebooks) — Quarto `.qmd` versions are auto-rendered from the marimo sources in CI.

| Notebook | Description |
|----------|-------------|
| [Zero-Shot Country Matching Explorer](01_country_matching_explorer.md) | Type a country name with typos, aliases, or alternate languages and see how the Matcher resolves it to a canonical entity — no training required. |
| [Novel Entity Detection Dashboard](02_novel_entity_dashboard.md) | Feed in text queries and see which ones are flagged as novel — not matching any known entity. Uses confidence + KNN distance strategies. |
| [Training Impact Analyzer](03_training_impact_analyzer.md) | Compare zero-shot vs trained matching side-by-side. See how accuracy changes for known and tricky inputs. |
| [Methodology & Benchmarks Overview](04_methodology_benchmarks_overview.md) | Comprehensive overview of all classification modes, novelty detection strategies, parameter sweeps, and performance benchmarks. |

## Run locally

[![marimo](https://img.shields.io/badge/Run%20locally-marimo-2489F4?logo=python)](https://github.com/minghao51/novel_entity_matcher)

```bash
uv run marimo edit notebooks/<name>.py
```

To rebuild docs pages locally after notebook changes:

```bash
uv run quarto render notebooks/
uv run python scripts/generate_notebook_docs.py
uv run mkdocs serve
```

# Notebooks

Interactive marimo notebooks for entity matching, novelty detection, and benchmarking.

**Source of truth:** [`notebooks/*.py`](https://github.com/minghao51/novel_entity_matcher/tree/main/notebooks) — all docs pages and static exports below are auto-generated from these in CI.

| Notebook | Description |
|----------|-------------|
| [Country Matching Explorer](01_country_matching_explorer.md) | Zero-shot entity matching with typo-tolerant country name resolution |
| [Novel Entity Detection Dashboard](02_novel_entity_dashboard.md) | Flag unknown queries using confidence + KNN novelty strategies |
| [Training Impact Analyzer](03_training_impact_analyzer.md) | Compare zero-shot vs trained matching accuracy side-by-side |
| [Methodology & Benchmarks](04_methodology_benchmarks_overview.md) | Comprehensive overview of all classification modes and novelty strategies |

## Run locally

[![marimo](https://img.shields.io/badge/Run%20locally-marimo-2489F4?logo=python)](https://github.com/minghao51/novel_entity_matcher)

```bash
uv run marimo edit notebooks/01_country_matching_explorer.py
```

To rebuild docs pages locally after notebook changes:

```bash
for f in notebooks/*.py; do
  name=$(basename "$f" .py)
  uv run marimo export md "$f" -o "docs/notebooks/${name}.md" -f
  uv run marimo export html "$f" -o "docs/notebooks/${name}.html" -f
done
uv run mkdocs serve
```

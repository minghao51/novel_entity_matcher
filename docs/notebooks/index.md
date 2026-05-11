# Notebooks

Interactive marimo notebooks for entity matching, novelty detection, discovery pipelines, and benchmarking.

**Source of truth:** [`notebooks/*.py`](https://github.com/minghao51/novel_entity_matcher/tree/main/notebooks) — Quarto `.qmd` versions are auto-rendered for static HTML embeds.

| Notebook | Description |
|----------|-------------|
| [Entity Matching Explorer](01_entity_matching_explorer.md) | Learn how the Matcher resolves messy text — typos, aliases, foreign names — to canonical entity IDs using embedding similarity and optional few-shot training. |
| [Novelty Detection Lab](02_novelty_detection_lab.md) | Explore how novelty detection flags inputs that don't belong to any known entity class. Compare strategies side-by-side and tune thresholds. |
| [Discovery Pipeline](03_discovery_pipeline.md) | Walk through the full 5-stage discovery pipeline: match, detect novel inputs, cluster them, extract evidence, and propose new class names. |
| [Benchmarks & Production Guide](04_benchmarks_production_guide.md) | Interactive reference for selecting classification modes, novelty strategies, embedding models, and production configurations. |

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

## Generated Artifacts Policy

- Commit regenerated `docs/notebooks/*.md` stubs whenever notebook titles/descriptions change.
- Commit regenerated `docs/notebooks/html/*.html` after `quarto render notebooks/` for publishable docs updates.
- Commit `notebooks/_freeze/**` only when you intentionally refresh cached execution outputs.

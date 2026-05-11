# Contributing

## Quick Start

Set up the development environment from the repo root:

```bash
uv sync --group dev
```

## Common Commands

```bash
uv run python -m pytest
uv run python -m pytest tests/test_core/test_matcher.py -q
uv run python -m build
uv run ruff check .
```

## Project Conventions

- PyPI distribution name: `novel-entity-matcher`
- Python import path: `novelentitymatcher`
- Repo folder name: `novel_entity_matcher`
- Library/package code lives in `src/novelentitymatcher/`

## Where Changes Go

- New library features: `src/novelentitymatcher/`
- Tests for library changes: `tests/` (mirror package areas where practical)
- User-facing examples: `examples/`
- Exploratory scripts / benchmarks: `experiments/`
- Jupyter notebooks only: `notebooks/`
- Documentation updates: `docs/`

## CLI Mapping

- `novelentitymatcher-ingest` entrypoint is implemented in `src/novelentitymatcher/ingestion/cli.py`

## Notes for Contributors

- Prefer non-breaking changes to public imports in `novelentitymatcher`.
- Keep docs and examples aligned with actual file locations when moving experiments.
- Avoid committing generated artifacts (`dist/`, `docs/build/`, checkpoints, `__pycache__`).
- If a PR changes `.qmd` notebooks, run:
  - `uv run quarto render notebooks/`
  - `uv run python scripts/generate_notebook_docs.py`
  - verify no docs drift with `git diff -- docs/notebooks/`.

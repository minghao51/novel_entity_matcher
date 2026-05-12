# Stack

## Language & Runtime

| Aspect | Value |
|--------|-------|
| Language | Python |
| Runtime version | 3.13 (local), CI tests 3.10–3.12 |
| Minimum version | `>=3.10` (`pyproject.toml:10`) |
| Package manager | uv (`uv.lock`, `uv sync`, `uv run`) |
| Build system | hatchling (`pyproject.toml:2`) |
| Build backend | `hatchling.build` |

## Project

| Aspect | Value |
|--------|-------|
| Package name | `novel-entity-matcher` |
| Version | 0.1.0 |
| License | MIT |
| Distribution | PyPI (Trusted Publishing via `pypa/gh-action-pypi-publish`) |
| Source layout | `src/novelentitymatcher/` |

## Production Dependencies

Core ML/NLP stack:

| Package | Version | Purpose | Key files |
|---------|---------|---------|-----------|
| `torch` | >=2.0.0 | Tensor ops, GPU/MPS fallback, BERT classifier | `core/bert_classifier.py`, `core/matcher.py` |
| `transformers` | >=4.45.0,<5.0.0 | HuggingFace model loading, BERT training | `core/bert_classifier.py` |
| `sentence-transformers` | >=3.0.0 | Dense embeddings, cross-encoder reranking, semantic search | `core/matcher.py`, `core/embedding_matcher.py`, `backends/sentencetransformer.py`, `backends/reranker_st.py` |
| `setfit` | >=1.0.0 | Few-shot text classification | `core/classifier.py`, `novelty/strategies/setfit_impl.py`, `utils/embeddings.py` |
| `datasets` | >=2.14.0 | HuggingFace dataset loading & caching | `core/classifier.py`, `benchmarks/loader.py` |
| `model2vec` | >=0.1.0 | Static embedding models (potion) | `backends/static_embedding.py` |
| `scikit-learn` | >=1.3.0 | Cosine similarity, TF-IDF, LOF, PCA, metrics | `core/blocking.py`, `core/embedding_matcher.py`, `novelty/strategies/lof.py`, `benchmarks/` |
| `numpy` | >=2.0.0 | Array operations throughout | Nearly all modules |
| `pandas` | >=2.0.0 | DataFrames for benchmarks & ingestion | `benchmarks/`, `ingestion/` |
| `networkx` | >=3.0,<4.0 | Graph-based hierarchy & Louvain clustering | `core/hierarchy.py`, `novelty/clustering/graph.py` |

NLP utilities:

| Package | Version | Purpose | Key files |
|---------|---------|---------|-----------|
| `nltk` | >=3.9.4 | Stopwords, lemmatization | `utils/preprocessing.py` |
| `rank-bm25` | >=0.2.2 | BM25 blocking for candidate generation | `core/blocking.py` |
| `rapidfuzz` | >=3.0.0 | Fuzzy string matching for blocking | `core/blocking.py` |

Configuration & validation:

| Package | Version | Purpose | Key files |
|---------|---------|---------|-----------|
| `pydantic` | >=2.0.0 | Data models, config validation, response schemas | `novelty/schemas/`, `novelty/config/`, `pipeline/config.py` |
| `pyyaml` | >=6.0.0 | YAML config loading | `config.yaml` |

Infrastructure:

| Package | Version | Purpose | Key files |
|---------|---------|---------|-----------|
| `requests` | >=2.31.0 | HTTP for data ingestion & benchmark dataset download | `ingestion/base.py`, `benchmarks/loader.py` |
| `optuna` | >=4.8.0 | Hyperparameter optimization for novelty weights | `benchmarks/weight_optimizer.py` |
| `matplotlib` | >=3.9.4 | Benchmark visualization | `benchmarks/visualization.py` |

## Optional Dependencies (`[opinion]`)

Advanced ML & LLM integration — installed via `pip install "novel-entity-matcher[opinion]"`:

| Package | Version | Purpose | Key files |
|---------|---------|---------|-----------|
| `litellm` | >=1.83.7 | Multi-provider LLM API (OpenRouter, OpenAI, Anthropic) | `novelty/proposal/llm.py`, `backends/litellm.py` |
| `dspy` | >=3.2.0 | Prompt optimization via GEPA teleprompter | `novelty/proposal/dspy_module.py`, `novelty/proposal/dspy_optimizer.py` |
| `chromadb` | >=0.5.0 | Vector database backend | `core/vector_store.py` |
| `hnswlib` | >=0.8.0 | ANN index (HNSW) | `novelty/storage/index.py` |
| `faiss-cpu` | >=1.7.4 | ANN index (FAISS) | `novelty/storage/index.py` |
| `hdbscan` | >=0.8.33 | Density-based clustering | `novelty/clustering/backends.py` |
| `umap-learn` | >=0.5 | Dimensionality reduction before clustering | `novelty/clustering/backends.py` |
| `python-igraph` | >=0.11.0 | Graph library for Leiden community detection | `novelty/clustering/graph.py` |
| `leidenalg` | >=0.10.0 | Leiden community detection | `novelty/clustering/graph.py` |
| `aiobreaker` | >=1.1.0 | Circuit breaker for LLM calls | `novelty/proposal/llm.py` |
| `tenacity` | >=9.0.0 | Retry with exponential backoff for LLM calls | `novelty/proposal/llm.py` |
| `tqdm` | >=4.66.0 | Progress bars | Various |
| `seaborn` | >=0.13.2 | Statistical visualization | `benchmarks/visualization.py` |
| `ipywidgets` | >=8.0.0 | Jupyter notebook widgets | — |

## Optional Dependencies (`[docs]`)

| Package | Version | Purpose |
|---------|---------|---------|
| `marimo` | >=0.23.4 | Interactive notebooks |
| `mkdocs` | >=1.6.0 | Static documentation site |
| `mkdocs-material` | >=9.5.0 | Material theme for MkDocs |
| `mkdocstrings[python]` | >=0.25.0 | Python API doc generation |
| `nbclient` | >=0.10.4 | Notebook execution |
| `ipykernel` | >=7.2.0 | Jupyter kernel |

## Dev Dependencies (`[dev]`)

| Package | Version | Purpose |
|---------|---------|---------|
| `ruff` | >=0.1.0 | Linter + formatter |
| `mypy` | >=1.19.1 | Static type checking |
| `pytest` | >=9.0.3 | Test runner |
| `pytest-asyncio` | >=1.2.0 | Async test support |
| `pre-commit` | >=3.6 | Git hook management |
| `pip-audit` | >=2.7.0 | Security vulnerability scanning |
| `build` | >=1.2.2 | PEP 517 package builder |
| `twine` | >=5.1.1 | Package validation & upload |
| `patchright` | >=1.58.0 | Browser automation (Playwright fork) |
| `beautifulsoup4` | >=4.14.3 | HTML parsing |
| `html-to-markdown` | >=1.8.0 | HTML to Markdown conversion |
| `nbformat` | >=5.10.4 | Notebook format handling |
| `types-networkx` | >=3.2 | Stubs for mypy |
| `types-pyyaml` | >=6.0.12 | Stubs for mypy |
| `types-requests` | >=2.32.4 | Stubs for mypy |
| `types-tqdm` | >=4.67.3 | Stubs for mypy |

## Tooling Configuration

### Ruff (`pyproject.toml:109–137`)

- Line length: 88
- Target: py310
- Rules: E, F, I, UP, B, C4, DTZ, T10, ISC, PIE, PT, RUF
- E501 (line length) ignored
- Quotes: double
- Notebooks: all rules disabled
- Tests: DTZ ignored

### Mypy (`pyproject.toml:169–247`)

- Target: Python 3.11
- `check_untyped_defs = true`
- `strict_optional = true`
- Many external packages have `ignore_missing_imports = true`
- Novelty strategy modules have relaxed checking (`ignore_errors = true`)

### Pytest (`pyproject.toml:152–167`)

- Testpaths: `tests/`
- Strict markers enabled
- Maxfail: 10
- Import mode: importlib
- Async mode: auto
- Markers: `unit`, `integration`, `slow`, `e2e`, `hf`, `llm`, `llm_mocked`, `serial`, `network`, `smoke`

### Pre-commit (`.pre-commit-config.yaml`)

- trailing-whitespace, end-of-file-fixer, check-yaml, check-merge-conflict, debug-statements
- Large file checks (1MB general, 5MB for docs/notebooks)
- `uv-lock` hook (keeps lockfile in sync)
- `ruff` lint + format (excludes notebooks)
- `mypy` (local hook)
- `conventional-pre-commit` (commit message convention)
- `quarto-render` (renders changed `.qmd` notebooks)

### Coverage (`pyproject.toml:139–150`)

- Source: `src/`
- Omits: tests, `__pycache__`
- Excludes: `pragma: no cover`, `__repr__`, `NotImplementedError`

## CI/CD (GitHub Actions)

| Workflow | File | Triggers | What it does |
|----------|------|----------|--------------|
| CI | `.github/workflows/ci.yml` | push, PR, dispatch | Pre-commit, mypy, smoke + fast tests, coverage floor (PR), matrix (3.10–3.12 on main), heavy integration, pip-audit, build check |
| Publish | `.github/workflows/publish.yml` | tags `v*` | Build sdist/wheel, validate with twine, publish to PyPI |
| Docs | `.github/workflows/docs.yml` | push to main (docs/src/notebooks), PR | Quarto render notebooks, MkDocs build, deploy to GitHub Pages |

All workflows use `astral-sh/setup-uv@v6` with Python 3.11 (test matrix includes 3.10–3.12).

## CLI Entry Points

Defined in `pyproject.toml:45–48`:

| Command | Module |
|---------|--------|
| `novelentitymatcher-ingest` | `novelentitymatcher.ingestion.cli:main` |
| `novelentitymatcher-bench` | `novelentitymatcher.benchmarks.cli:main` |
| `novelentitymatcher-review` | `novelentitymatcher.novelty.cli:main` |

# Stack — novel-entity-matcher

## Languages & Runtime

| Item | Value | Source |
|------|-------|--------|
| Language | Python 3 | `pyproject.toml` |
| Minimum Python | >=3.10 | `pyproject.toml:10` |
| Local Python | 3.13 | `.python-version` |
| CI Python | 3.10, 3.11, 3.12 (matrix) | `.github/workflows/test.yml:63` |
| Package manager | uv (astral-sh) | `uv.lock`, `.pre-commit-config.yaml:13` |
| Build backend | hatchling >=1.27.0 | `pyproject.toml:2` |

## Frameworks & Libraries

### Core ML / NLP

| Library | Purpose | Files |
|---------|---------|-------|
| **sentence-transformers** >=3.0.0 | Dense text embeddings (SentenceTransformer, CrossEncoder) | `src/novelentitymatcher/core/matcher.py`, `src/novelentitymatcher/utils/embeddings.py`, `src/novelentitymatcher/backends/sentencetransformer.py` |
| **setfit** >=1.0.0 | Few-shot classification via SetFit | `src/novelentitymatcher/core/classifier.py`, `src/novelentitymatcher/novelty/strategies/setfit_impl.py` |
| **transformers** >=4.45.0,<5.0.0 | Hugging Face model hub, tokenizers, BERT classifier | `src/novelentitymatcher/core/bert_classifier.py` |
| **torch** >=2.0.0 | Deep learning backend (PyTorch) | `src/novelentitymatcher/core/bert_classifier.py` |
| **model2vec** >=0.1.0 | Static embedding models (e.g. potion-base) | `src/novelentitymatcher/backends/static_embedding.py` |
| **scikit-learn** >=1.3.0 | Cosine similarity, metrics, ML utilities | `src/novelentitymatcher/core/hybrid.py`, `src/novelentitymatcher/benchmarks/shared.py` |
| **numpy** >=2.0.0 | Array operations throughout | Nearly all modules |
| **pandas** >=2.0.0 | DataFrames for benchmarks and ingestion | `src/novelentitymatcher/benchmarks/runner.py`, `src/novelentitymatcher/benchmarks/loader.py` |
| **nltk** >=3.9.4 | Natural language processing utilities | `pyproject.toml:35` |
| **optuna** >=4.8.0 | Hyperparameter optimization | `pyproject.toml:41` |

### Similarity & Search

| Library | Purpose | Files |
|---------|---------|-------|
| **rank-bm25** >=0.2.2 | BM25 blocking strategy | `pyproject.toml:37` |
| **rapidfuzz** >=3.0.0 | Fuzzy string matching blocking | `pyproject.toml:38` |
| **networkx** >=3.0,<4.0 | Hierarchical entity matching graph | `src/novelentitymatcher/core/hierarchy.py` |

### Data Validation & Config

| Library | Purpose | Files |
|---------|---------|-------|
| **pydantic** >=2.0.0 | Schema validation, BaseModel for proposals/configs | `src/novelentitymatcher/novelty/proposal/llm.py`, `src/novelentitymatcher/novelty/schemas/` |
| **pyyaml** >=6.0.0 | Config file loading (config.yaml) | `src/novelentitymatcher/config.py` |

### LLM Integration (optional `[llm]` extra)

| Library | Purpose | Files |
|---------|---------|-------|
| **litellm** >=1.83.7 | Unified LLM API (OpenRouter, OpenAI, Anthropic) | `src/novelentitymatcher/backends/litellm.py`, `src/novelentitymatcher/novelty/proposal/llm.py` |
| **tenacity** >=9.0.0 | Retry logic for LLM calls | `src/novelentitymatcher/novelty/proposal/llm.py` |
| **aiobreaker** >=1.1.0 | Circuit breaker for LLM resilience | `src/novelentitymatcher/novelty/proposal/llm.py` |

### ANN / Clustering (optional `[novelty]`/`[full]` extras)

| Library | Purpose | Files |
|---------|---------|-------|
| **hnswlib** >=0.8.0 | HNSW approximate nearest neighbor index | `src/novelentitymatcher/novelty/storage/index.py` |
| **faiss-cpu** >=1.7.4 | FAISS ANN index | `src/novelentitymatcher/novelty/storage/index.py` |
| **hdbscan** >=0.8.33 | HDBSCAN clustering | `src/novelentitymatcher/novelty/clustering/backends.py` |
| **umap-learn** >=0.5 | UMAP dimensionality reduction | `pyproject.toml` `[full]` extra |

### HTTP

| Library | Purpose | Files |
|---------|---------|-------|
| **requests** >=2.31.0 | HTTP client for data ingestion, benchmark dataset downloads | `src/novelentitymatcher/ingestion/base.py`, `src/novelentitymatcher/benchmarks/loader.py` |

### Visualization (optional `[viz]` extra)

| Library | Purpose | Files |
|---------|---------|-------|
| **matplotlib** >=3.9.4 | Plotting (core dep + viz extra) | `src/novelentitymatcher/benchmarks/visualization.py` |
| **seaborn** >=0.13.2 | Statistical visualization | `pyproject.toml` `[viz]` extra |

## Dependencies (with versions from manifests)

### Core dependencies — `pyproject.toml:24-43`

```
numpy>=2.0.0
networkx>=3.0,<4.0
pandas>=2.0.0
scikit-learn>=1.3.0
sentence-transformers>=3.0.0
setfit>=1.0.0
datasets>=2.14.0
torch>=2.0.0
transformers>=4.45.0,<5.0.0
nltk>=3.9.4
requests>=2.31.0
pydantic>=2.0.0
rank-bm25>=0.2.2
rapidfuzz>=3.0.0
pyyaml>=6.0.0
model2vec>=0.1.0
optuna>=4.8.0
matplotlib>=3.9.4
```

### Optional extras — `pyproject.toml:50-140`

| Extra | Key Packages |
|-------|-------------|
| `docs` | mkdocs>=1.6.0, mkdocs-material>=9.5.0, mkdocstrings[python]>=0.25.0, marimo>=0.23.4 |
| `notebooks` | marimo>=0.23.4 |
| `jupyter` | tqdm>=4.66.0, ipywidgets>=8.0.0 |
| `novelty` | pydantic>=2.0.0, hnswlib>=0.8.0, hdbscan>=0.8.33, faiss-cpu>=1.7.4 |
| `llm` | litellm>=1.83.7, pydantic>=2.0.0, tenacity>=9.0.0, aiobreaker>=1.1.0 |
| `clustering` | hdbscan>=0.2, umap-learn>=0.5 |
| `viz` | matplotlib>=3.9.4, seaborn>=0.13.2 |
| `full` | pydantic, hnswlib, hdbscan, faiss-cpu, litellm, umap-learn |

### Pinned versions from `uv.lock` (selected, Python >=3.11 resolution)

- torch: 2.11.0
- transformers: (resolved via sentence-transformers)
- accelerate: 1.13.0
- numpy: 2.4.4
- pydantic: (latest compatible)

## Build Tools & Config

| Tool | Config Location | Details |
|------|----------------|---------|
| **uv** | `uv.lock` | Package resolution and locking |
| **hatchling** | `pyproject.toml:1-3` | Build backend, wheel packages `src/novelentitymatcher` |
| **ruff** | `pyproject.toml:173-202` | Linter (E,F,I,UP,B,C4,DTZ,T10,ISC,PIE,PT,RUF) + formatter (double quotes, 88-char line) |
| **mypy** | `pyproject.toml:220-296` | Type checker, target Python 3.11, strict_optional, per-module ignore_missing_imports |
| **pytest** | `pyproject.toml:203-218` | strict-markers, 12 custom markers (unit/integration/slow/e2e/hf/llm/llm_mocked/serial/network/smoke), asyncio_mode=auto |
| **pre-commit** | `.pre-commit-config.yaml` | trailing-whitespace, end-of-file-fixer, check-yaml, uv-lock, ruff (lint+format), mypy, conventional-pre-commit |
| **mkdocs** | `mkdocs.yml` | Material theme, mkdocstrings[python], mike versioning, search plugin |
| **marimo** | `notebooks/*.py` | Interactive notebooks (4 notebooks) |
| **conventional-pre-commit** | `.pre-commit-config.yaml:34-38` | Enforces conventional commit messages |

## Development Dependencies — `pyproject.toml:142-159`

```
beautifulsoup4>=4.14.3
build>=1.2.2
html-to-markdown>=1.8.0
mypy>=1.19.1
patchright>=1.58.0
pip-audit>=2.7.0
pre-commit>=3.6
pytest>=9.0.3
pytest-asyncio>=1.2.0
ruff>=0.1.0
twine>=5.1.1
types-networkx>=3.2
types-pyyaml>=6.0.12.20250915
types-requests>=2.32.4.20260107
types-tqdm>=4.67.3.20260205
```

## CLI Entry Points — `pyproject.toml:45-48`

| Command | Module |
|---------|--------|
| `novelentitymatcher-ingest` | `novelentitymatcher.ingestion.cli:main` |
| `novelentitymatcher-bench` | `novelentitymatcher.benchmarks.cli:main` |
| `novelentitymatcher-review` | `novelentitymatcher.novelty.cli:main` |

# Technology Stack

**Analysis Date:** 2026-04-06

## Languages

**Primary:**
- Python 3.13 — Main application language (supports 3.9–3.12)

## Runtime

**Environment:**
- CPython 3.13

**Package Manager:**
- uv — Fast Python package installer and resolver
- Lockfile: uv.lock (present)

## Frameworks

**Core:**
- SetFit 1.0+ — Few-shot sentence transformer fine-tuning for entity matching
- sentence-transformers 3.0+ — Embedding generation and semantic search
- PyTorch 2.0+ — Deep learning backend for transformers
- transformers 4.45+ — Hugging Face model loading and inference

**ML/Scientific:**
- scikit-learn 1.3+ — Classical ML utilities, metrics, preprocessing
- pandas 2.0+ — Data manipulation and analysis
- numpy 2.0+ — Numerical computing
- networkx 3.0+ — Graph-based hierarchy structures
- rank-bm25 0.2.2+ — BM25 text retrieval
- rapidfuzz 3.0+ — Fast fuzzy string matching

**Novelty Detection:**
- hnswlib 0.8.0+ — Approximate nearest neighbor search
- faiss-cpu 1.7.4+ — Dense vector similarity search
- hdbscan 0.8.33+ — Density-based clustering for novel class discovery
- umap-learn 0.5+ — Dimensionality reduction

**LLM Integration:**
- litellm 1.50.0+ — Unified LLM API (OpenAI, Anthropic, OpenRouter)
- pydantic 2.0+ — Data validation and structured output

**Static Embeddings:**
- model2vec 0.1.0+ — Static model distillation

**Testing:**
- pytest 8.4.2+ — Test framework with markers (integration, slow, hf, llm, e2e)
- pytest-asyncio 1.2.0+ — Async test support

**Build/Dev:**
- hatchling 1.27.0+ — Build backend
- ruff 0.1.0+ — Fast Python linter
- black 23.0+ — Code formatter
- mypy 1.19.1+ — Static type checker
- twine 5.1.1+ — PyPI package uploader
- build 1.2.2+ — Package building

**Visualization:**
- matplotlib 3.9.4+ — Plotting
- seaborn 0.13.2+ — Statistical visualization

**Web Scraping (dev):**
- patchright 1.58.0+ — Browser automation
- beautifulsoup4 4.14.3+ — HTML parsing
- html-to-markdown 1.8.0+ — Content conversion

## Configuration

**Environment:**
- `.env` file for LLM API keys (OpenRouter, Anthropic, OpenAI)
- `.env.example` — Template with documented variables
- `config.yaml` for model and training defaults
- `.python-version` pins Python 3.13

**Build:**
- `pyproject.toml` — Project metadata, dependencies, tool configs
- `uv.lock` — Frozen dependency resolution

**Tool Configs:**
- `[tool.pytest.ini_options]` — Test paths, markers (integration, slow, hf, llm, llm_mocked, e2e), async mode
- `[tool.mypy]` — Type checking (python_version=3.13, strict_optional=true) with per-module overrides for ML libs
- `[tool.hatch.build]` — Wheel and sdist build targets

**Optional Dependency Groups:**
- `jupyter` — tqdm, ipywidgets for notebook progress bars
- `novelty` — pydantic, hnswlib, hdbscan, faiss-cpu for novel class detection
- `llm` — litellm, pydantic for LLM-backed embedding/reranking
- `clustering` — hdbscan, umap-learn for novelty discovery
- `viz` — matplotlib, seaborn for benchmark visualization
- `dev` — black, ruff, pytest, mypy, twine, patchright, bs4
- `full` — pydantic, hnswlib, hdbscan, faiss-cpu, litellm, umap-learn
- `all` — everything combined

## Platform Requirements

**Development:**
- Python 3.9–3.13
- uv package manager
- Optional: GPU for accelerated embedding/training

**Production:**
- PyPI package distribution (`novel-entity-matcher`)
- CLI entrypoints: `novelentitymatcher-ingest`, `novelentitymatcher-bench`, `novelentitymatcher-review`

---

*Stack analysis: 2026-04-06*

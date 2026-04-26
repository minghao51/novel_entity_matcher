# Technology Stack

**Analysis Date:** 2026-04-23

## Languages

**Primary:**
- Python 3.10+ - Core application language
- Python 3.11 - Primary development and CI target
- Python 3.9, 3.10, 3.11, 3.12 - Supported runtime versions

**Secondary:**
- Markdown - Documentation (README, docs/)
- YAML - Configuration files

## Runtime

**Environment:**
- Python 3.10+ (requires-python: ">=3.10")

**Package Manager:**
- uv - Fast Python package manager
- Lockfile: Not explicitly managed (uv sync uses pyproject.toml)

## Frameworks

**Core:**
- SetFit - Few-shot text classification
- Sentence Transformers - Text embeddings
- Transformers (Hugging Face) - Pre-trained models
- PyTorch - Deep learning backend

**Testing:**
- pytest 8.4.2+ - Test framework
- pytest-asyncio 1.2.0+ - Async test support

**Build/Dev:**
- black 23.0.0+ - Code formatting
- ruff 0.1.0+ - Linting and fast Python linter
- mypy 1.19.1+ - Static type checking
- build 1.2.2+ - Package building
- twine 5.1.1+ - PyPI upload

## Key Dependencies

**Critical:**
- numpy 2.0.0+ - Array operations and numerical computing
- pandas 2.0.0+ - Data manipulation
- sentence-transformers 3.0.0+ - Text embeddings
- setfit 1.0.0+ - Few-shot classification
- torch 2.0.0+ - Deep learning framework
- transformers 4.45.0+ - Pre-trained transformer models

**Infrastructure:**
- scikit-learn 1.3.0+ - ML utilities and metrics
- sklearn - Clustering, pairwise distances, feature extraction
- networkx 3.0+ - Graph operations
- requests 2.31.0+ - HTTP client
- pyyaml 6.0.0+ - YAML parsing

**Novelty Detection:**
- hnswlib 0.8.0+ - Approximate nearest neighbor search
- faiss-cpu 1.7.4+ - Vector similarity search
- hdbscan 0.8.33+ - Density-based clustering
- umap-learn 0.5+ - Dimensionality reduction for clustering

**LLM Integration:**
- litellm 1.50.0+ - Multi-provider LLM API client
- pydantic 2.0.0+ - Data validation

**Embeddings & Retrieval:**
- model2vec 0.1.0+ - Static embeddings
- rank-bm25 0.2.2+ - BM25 text retrieval
- rapidfuzz 3.0.0+ - Fuzzy string matching
- nltk 3.9.2+ - NLP utilities

**Benchmarks:**
- matplotlib 3.9.4+ - Plotting
- seaborn 0.13.2+ - Statistical visualization
- tqdm 4.66.0+ - Progress bars
- ipywidgets 8.0.0+ - Jupyter widgets

## Configuration

**Environment:**
- NOVEL_ENTITY_MATCHER_VERBOSE - Enable verbose debug logging
- LLM_API_KEY(s) - For LLM providers (OpenAI, Anthropic, etc.)

**Build:**
- pyproject.toml - Primary configuration (dependencies, scripts, tools)
- hatchling - Build backend

## Platform Requirements

**Development:**
- Python 3.9+
- uv package manager
- macOS MPS fallback enabled for arm64 (PyTorch)

**Production:**
- Python 3.10+
- 4GB+ RAM recommended for embedding models
- Optional: GPU for faster training (CUDA/MPS)

---

*Stack analysis: 2026-04-23*

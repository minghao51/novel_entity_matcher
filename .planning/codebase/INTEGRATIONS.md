# Integrations

## LLM Providers (via LiteLLM)

Multi-provider LLM integration for novel class proposal. LiteLLM abstracts provider differences.

| Provider | Env var | Models used | Key files |
|----------|---------|-------------|-----------|
| OpenRouter | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4`, `openai/gpt-4o` | `novelty/proposal/llm.py:173–178` |
| Anthropic | `ANTHROPIC_API_KEY` | `anthropic/claude-sonnet-4` | `novelty/proposal/llm.py:175` |
| OpenAI | `OPENAI_API_KEY` | `openai/gpt-4o` | `novelty/proposal/llm.py:176` |
| LiteLLM (embeddings/rerank) | `LITELLM_API_KEY` | Configurable | `backends/litellm.py:27,48` |

Provider-to-env mapping: `utils/api_keys.py:3–7`

**LLM configuration** (`novelty/proposal/config.py`):
- `LLM_TIMEOUT` — request timeout (default 30s)
- `LLM_MAX_RETRIES` — max retry attempts (default 5)
- `LLM_CIRCUIT_FAIL_MAX` — failures before circuit breaker opens (default 3)
- `LLM_CIRCUIT_RESET_SECONDS` — circuit breaker recovery (default 60s)
- `LLM_CLASS_PROPOSER_PROVIDER` — default provider
- `LLM_CLASS_PROPOSER_MODEL` — default model

**Resilience features**:
- Circuit breaker via `aiobreaker` (`novelty/proposal/llm.py:52–68`)
- Retry with exponential jitter via `tenacity` (`novelty/proposal/llm.py:1111–1117`)
- Automatic fallback chain across providers (`novelty/proposal/llm.py:1026–1109`)

## DSPy Prompt Optimization (optional)

DSPy replaces handcrafted prompts with optimized Signatures and Modules, trained via GEPA teleprompter.

| Component | File | Purpose |
|-----------|------|---------|
| `ClusterProposalSignature` | `novelty/proposal/dspy_module.py:24` | DSPy Signature defining inputs/outputs |
| `DSPyProposalModule` | `novelty/proposal/dspy_module.py:45` | ChainOfThought module for class proposals |
| `DSPyProposalOptimizer` | `novelty/proposal/dspy_optimizer.py:36` | GEPA optimization with review record training data |
| `records_to_dspy_examples` | `novelty/proposal/dspy_module.py:87` | Converts review records to `dspy.Example` |
| `proposal_metric` | `novelty/proposal/dspy_module.py:129` | Metric function for GEPA optimization |

DSPy is optional — `LLMClassProposer` falls back to manual prompts when no `dspy_module` is provided (`novelty/proposal/llm.py:361–367`).

## HuggingFace Hub

| Usage | File | Details |
|-------|------|---------|
| Model downloads | `utils/embeddings.py:213–232` | `SetFitModel.from_pretrained`, `SentenceTransformer` |
| Dataset loading | `benchmarks/loader.py:25–29` | `datasets.load_dataset`, `huggingface_hub.dataset_info` |
| BERT models | `core/bert_classifier.py:18` | `transformers.AutoModelForSequenceClassification` |
| Static embeddings | `backends/static_embedding.py:48` | `model2vec.StaticModel` |
| Auth token | `.env.example:51` | `HF_TOKEN` (optional, for gated models) |

Models referenced:
- `sentence-transformers/paraphrase-mpnet-base-v2` (`config.yaml:4`)
- `BAAI/bge-m3` (`config.yaml:13`)
- `minishlab/potion-base-8M`, `minishlab/potion-base-32M` (`backends/static_embedding.py:29–30`)
- `distilbert-base-uncased` (`core/bert_classifier.py:65`)

## Vector Store / ANN Backends

Approximate Nearest Neighbor index for similarity search in novelty detection.

| Backend | Package | File | Details |
|---------|---------|------|---------|
| HNSWLib (default) | `hnswlib` | `novelty/storage/index.py:69–85` | Cosine similarity, configurable ef_construction and M |
| FAISS | `faiss-cpu` | `novelty/storage/index.py:89–100` | `IndexFlatIP` (inner product) |
| ChromaDB | `chromadb` | `core/vector_store.py:129–140` | Persistent or ephemeral client |
| Exact search | numpy | `novelty/storage/index.py` | Fallback when no ANN library available |

Default config: `config_registry.py:153–158`

## Clustering Libraries

| Algorithm | Package | File | Purpose |
|-----------|---------|------|---------|
| HDBSCAN | `hdbscan` | `novelty/clustering/backends.py:67–76` | Density-based clustering (primary) |
| UMAP + HDBSCAN | `umap-learn` + `hdbscan` | `novelty/clustering/backends.py:266–325` | Dimensionality reduction then clustering |
| Leiden | `python-igraph` + `leidenalg` | `novelty/clustering/graph.py:93–148` | Community detection on similarity graph |
| Louvain | `networkx` | `novelty/clustering/graph.py:194` | Community detection alternative |
| OPTICS | scikit-learn | `novelty/clustering/backends.py` | Ordered-based clustering |

## External Data Sources (Ingestion)

HTTP data ingestion for reference entity lists via `requests`.

| Dataset | Source URL | Fetcher | File |
|---------|-----------|---------|------|
| Universities | `github.com/Hipo/university-domains-list` | `UniversitiesFetcher` | `ingestion/universities.py` |
| O*NET Occupations | `www.onetcenter.org/dl/30_2/occupation_data.zip` | `OccupationsFetcher` | `ingestion/occupations.py:18` |
| UNSPSC Products | `unstats.un.org/unsd/services/v2/` | `UNSPSCFetcher` | `ingestion/products.py:19` |
| NAICS Industries | `github.com/erickogore/country-code-json`, `github.com/datasets/industry-codes` | `IndustriesFetcher` | `ingestion/industries.py:19–22` |
| ISO 4217 Currencies | `datahub.io/core/currency-codes/r/codes-all.csv` | `CurrenciesFetcher` | `ingestion/currencies.py:12` |
| ISO 639 Languages | `datahub.io/core/language-codes/r/language-codes-full.csv` | `LanguagesFetcher` | `ingestion/languages.py:12` |
| IANA Timezones | `github.com/eggert/tz/main/zone.tab` | `TimezonesFetcher` | `ingestion/timezones.py:12` |

All fetchers extend `BaseFetcher` (`ingestion/base.py`) with async support, rate limiting, and size validation.

## Environment Variables

| Variable | Default | Purpose | File |
|----------|---------|---------|------|
| `OPENROUTER_API_KEY` | — | OpenRouter LLM API key | `utils/api_keys.py:4` |
| `ANTHROPIC_API_KEY` | — | Anthropic API key | `utils/api_keys.py:5` |
| `OPENAI_API_KEY` | — | OpenAI API key | `utils/api_keys.py:6` |
| `LITELLM_API_KEY` | — | LiteLLM embedding/rerank key | `backends/litellm.py:27,48` |
| `LLM_CLASS_PROPOSER_PROVIDER` | `openrouter` | Default LLM provider | `.env.example:26` |
| `LLM_CLASS_PROPOSER_MODEL` | `anthropic/claude-sonnet-4` | Default LLM model | `.env.example:30` |
| `LLM_TIMEOUT` | `30` | Request timeout (seconds) | `novelty/proposal/config.py:25–31` |
| `LLM_MAX_RETRIES` | `5` | Max retry attempts | `novelty/proposal/config.py:33–38` |
| `LLM_CIRCUIT_FAIL_MAX` | `3` | Circuit breaker threshold | `novelty/proposal/config.py:40–46` |
| `LLM_CIRCUIT_RESET_SECONDS` | `60` | Circuit breaker reset | `novelty/proposal/config.py:48–55` |
| `HF_TOKEN` | — | HuggingFace token (optional) | `.env.example:51` |
| `NOVEL_ENTITY_MATCHER_VERBOSE` | `false` | Verbose logging | `__init__.py:19`, `core/matcher.py:80` |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` (macOS ARM) | CPU fallback for MPS ops | `core/matcher.py:15`, `backends/static_embedding.py:12` |

Environment management: `pydantic-settings` with `.env` file support (`novelty/proposal/config.py:65–70`).

## ML Models & Frameworks

| Framework | Purpose | Key files |
|-----------|---------|-----------|
| PyTorch | Backend for transformers, tensor ops, device management (CPU/CUDA/MPS) | `core/bert_classifier.py` |
| Sentence Transformers | Dense embeddings (`encode`), cross-encoder reranking, semantic search | `core/matcher.py`, `core/embedding_matcher.py`, `backends/sentencetransformer.py` |
| SetFit | Few-shot classification via sentence embedding fine-tuning | `core/classifier.py`, `novelty/strategies/setfit_impl.py` |
| HuggingFace Transformers | BERT-based sequence classification | `core/bert_classifier.py` |
| Model2Vec | Static embedding distillation (potion models) | `backends/static_embedding.py` |
| scikit-learn | Cosine similarity, TF-IDF, LOF, PCA, train/test split, metrics | `core/blocking.py`, `core/embedding_matcher.py`, `novelty/strategies/lof.py` |
| NLTK | Stopword removal, lemmatization | `utils/preprocessing.py` |
| BM25 (rank-bm25) | Sparse retrieval for candidate blocking | `core/blocking.py` |
| RapidFuzz | Fuzzy string matching for blocking | `core/blocking.py` |
| Optuna | Bayesian hyperparameter optimization | `benchmarks/weight_optimizer.py` |

## Monitoring & Observability

| Component | File | Purpose |
|-----------|------|---------|
| `get_logger` | `utils/logging_config.py` | Structured logging with configurable verbosity |
| API key redaction | `exceptions.py` (`_redact_api_keys`) | Prevents API key leakage in logs |
| LLM call logging | `novelty/proposal/llm.py` | Logs model attempts, retryable errors, fallbacks |
| Circuit breaker state | `novelty/proposal/llm.py:258–261` | Tracks LLM provider health |
| Drift detection | `novelty/drift/` | Snapshot-based distribution monitoring |

## Documentation Pipeline

| Tool | File | Purpose |
|------|------|---------|
| MkDocs Material | `mkdocs.yml` | Static docs site |
| mkdocstrings | `mkdocs.yml:99` | Python API reference from docstrings |
| mike | `mkdocs.yml:148` | Versioned docs deployment |
| Quarto | `notebooks/_quarto.yml`, `Makefile` | Notebook rendering with freeze cache |
| Marimo | `pyproject.toml:52` | Interactive notebook authoring |

Deployed to GitHub Pages via `.github/workflows/docs.yml`.

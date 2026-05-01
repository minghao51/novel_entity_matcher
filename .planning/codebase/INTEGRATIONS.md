# Integrations — novel-entity-matcher

## External APIs

### LLM Providers (via litellm)

All LLM calls go through **litellm** (`src/novelentitymatcher/backends/litellm.py`, `src/novelentitymatcher/novelty/proposal/llm.py`), which provides a unified interface to multiple providers:

| Provider | Env Key | Models Used | Files |
|----------|---------|-------------|-------|
| **OpenRouter** | `OPENROUTER_API_KEY` | `anthropic/claude-sonnet-4`, any OpenRouter-supported model | `src/novelentitymatcher/novelty/proposal/llm.py`, `.env.example:10` |
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-sonnet-4` | `.env.example:14` |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o` | `.env.example:18` |

Configuration env vars:
- `LLM_CLASS_PROPOSER_PROVIDER` — default provider (`openrouter`, `anthropic`, `openai`)
- `LLM_CLASS_PROPOSER_MODEL` — default model name
- `LLM_TIMEOUT` — request timeout (default 30s)
- `LLM_MAX_RETRIES` — max retry attempts (default 5)
- `LLM_CIRCUIT_FAIL_MAX` — consecutive failures before circuit breaker opens (default 3)
- `LLM_CIRCUIT_RESET_SECONDS` — circuit breaker reset duration (default 60s)

Source: `.env.example`, `src/novelentitymatcher/novelty/proposal/llm.py`, `scripts/setup_llm.sh`

Resilience features:
- **Retry** via `tenacity` — exponential backoff with jitter (`src/novelentitymatcher/novelty/proposal/llm.py:19-27`)
- **Circuit breaker** via `aiobreaker` — prevents cascading failures (`src/novelentitymatcher/novelty/proposal/llm.py:46-57`)
- **Model fallback** — tries `primary_model` then `fallback_models` sequentially (`src/novelentitymatcher/novelty/proposal/llm.py:977-978`)

### Hugging Face Hub

| Integration | Purpose | Files |
|-------------|---------|-------|
| Model downloads | SentenceTransformer, SetFit, BERT model weights | `src/novelentitymatcher/utils/embeddings.py`, `src/novelentitymatcher/core/classifier.py` |
| Token (optional) | `HF_TOKEN` env var for gated/private models | `.env.example:51` |
| Benchmark datasets | `datasets` library for loading HF datasets | `pyproject.toml:31` |

### Data Source APIs (Ingestion)

Each fetcher in `src/novelentitymatcher/ingestion/` uses `requests` to download reference data:

| Fetcher | Source | File |
|---------|--------|------|
| **Universities** | External university list APIs | `src/novelentitymatcher/ingestion/universities.py` |
| **Products** | Product reference data | `src/novelentitymatcher/ingestion/products.py` |
| **Occupations** | Occupation/taxonomy data | `src/novelentitymatcher/ingestion/occupations.py` |
| **Industries** | Industry classification data | `src/novelentitymatcher/ingestion/industries.py` |
| **Currencies** | Currency reference data | `src/novelentitymatcher/ingestion/currencies.py` |
| **Languages** | Language reference data | `src/novelentitymatcher/ingestion/languages.py` |
| **Timezones** | Timezone reference data | `src/novelentitymatcher/ingestion/timezones.py` |

All use the `BaseFetcher` pattern from `src/novelentitymatcher/ingestion/base.py` with `requests.get()` and configurable timeouts (default 60s).

### Benchmark Dataset Downloads

Entity resolution datasets are downloaded at benchmark time:

| Source | Files |
|--------|-------|
| HTTP download of CSV files (tableA, tableB, test pairs) | `src/novelentitymatcher/benchmarks/loader.py:169-199` |

Uses `requests` with 60s timeout, 100MB size limit, content-type validation.

## Databases

**No traditional database integration.** The project is a library, not a service.

### Local Storage

| Storage Type | Purpose | Files |
|-------------|---------|-------|
| **CSV files** | Training data, processed ingestion data, benchmark results | `data/`, `src/novelentitymatcher/ingestion/base.py` |
| **JSON files** | Default config, country codes | `src/novelentitymatcher/data/default_config.json`, `src/novelentitymatcher/data/country_codes.json` |
| **YAML config** | User configuration overrides | `config.yaml`, `src/novelentitymatcher/config.py` |
| **ANN indexes** | In-memory HNSW/FAISS indexes (no persistent DB) | `src/novelentitymatcher/novelty/storage/index.py` |
| **File-based persistence** | Proposal review and class storage | `src/novelentitymatcher/novelty/storage/persistence.py`, `src/novelentitymatcher/novelty/storage/review.py` |
| **ML checkpoints** | Model weights (safetensors, .pt, .bin) — gitignored | `.gitignore:44-49` |

## Auth Providers

**No auth provider integration.** The library authenticates to external APIs via API keys in environment variables:

| Service | Auth Method | Env Var | File |
|---------|-------------|---------|------|
| OpenRouter | API key | `OPENROUTER_API_KEY` | `.env.example:10` |
| Anthropic | API key | `ANTHROPIC_API_KEY` | `.env.example:14` |
| OpenAI | API key | `OPENAI_API_KEY` | `.env.example:18` |
| Hugging Face | Token (optional) | `HF_TOKEN` | `.env.example:51` |

API key redaction is handled by `_redact_api_keys()` in `src/novelentitymatcher/exceptions.py:136-137` to prevent accidental logging of secrets.

## Third-party Services

### CI/CD (GitHub Actions)

| Workflow | Purpose | File |
|----------|---------|------|
| **Lint** | ruff, mypy, pre-commit, build verification, pip-audit security scan | `.github/workflows/lint.yml` |
| **Test** | Smoke tests, fast unit tests, heavy integration tests (Python 3.10-3.12 matrix) | `.github/workflows/test.yml` |
| **Publish** | Build sdist/wheel, publish to PyPI via Trusted Publishing | `.github/workflows/publish.yml` |
| **Deploy Docs** | MkDocs build, marimo notebook export (md + html), GitHub Pages deployment | `.github/workflows/docs.yml` |

### PyPI

- Package: `novel-entity-matcher` — published via `pypa/gh-action-pypi-publish` with Trusted Publishing (OIDC)
- Source: `.github/workflows/publish.yml:32-33`

### GitHub Pages

- Documentation site hosted at `https://minghao51.github.io/novel_entity_matcher/`
- Versioned via `mike` (declared in `mkdocs.yml:141-142`)
- Source: `.github/workflows/docs.yml`, `mkdocs.yml`

### Pre-trained Model Sources

All models are loaded from Hugging Face Hub via `sentence-transformers`, `transformers`, `setfit`, or `model2vec`:

| Model Alias | Full Name | Backend | File |
|-------------|-----------|---------|------|
| `potion-8m` | minishlab/potion-base-8M | static | `src/novelentitymatcher/config_registry.py:4-9` |
| `potion-32m` | minishlab/potion-base-32M | static | `src/novelentitymatcher/config_registry.py:11-16` |
| `bge-m3` | BAAI/bge-m3 | sentence-transformers | `src/novelentitymatcher/config_registry.py:34-39` |
| `bge-base` | BAAI/bge-base-en-v1.5 | sentence-transformers | `src/novelentitymatcher/config_registry.py:29-33` |
| `mpnet` | sentence-transformers/all-mpnet-base-v2 | sentence-transformers | `src/novelentitymatcher/config_registry.py:47-52` |
| `minilm` | sentence-transformers/all-MiniLM-L6-v2 | sentence-transformers | `src/novelentitymatcher/config_registry.py:54-58` |
| `nomic` | nomic-ai/nomic-embed-text-v1 | sentence-transformers | `src/novelentitymatcher/config_registry.py:41-45` |
| `distilbert` | distilbert-base-uncased | bert | `src/novelentitymatcher/config_registry.py:59-66` |
| `tinybert` | huawei-noah/TinyBERT_General_4L_312D | bert | `src/novelentitymatcher/config_registry.py:68-76` |

## Webhooks/Callbacks

### User-facing Callbacks

| Callback | Purpose | File |
|----------|---------|------|
| **MetricEvent callbacks** | User-provided callbacks for custom metric handling | `src/novelentitymatcher/monitoring/metrics.py:14-33` |
| **Pipeline stage hooks** | `StageContext` / `StageResult` for pipeline extensibility | `src/novelentitymatcher/pipeline/` |

The library does **not** receive webhooks. It emits events through callback interfaces that users can subscribe to. No incoming HTTP server or webhook listener exists.

### Pipeline Observability

- `MetricEvent` dataclass with standard metric names (`METRIC_MATCH_LATENCY`, `METRIC_NOVELTY_RATE`, etc.)
- Source: `src/novelentitymatcher/monitoring/metrics.py:37-40`

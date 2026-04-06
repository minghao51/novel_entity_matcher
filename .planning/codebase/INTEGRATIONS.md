# External Integrations

**Analysis Date:** 2026-04-06

## APIs & External Services

**LLM Providers:**
- OpenRouter — Multi-provider LLM gateway (recommended)
  - SDK/Client: litellm
  - Auth: `OPENROUTER_API_KEY`
- Anthropic — Claude models for class proposal
  - SDK/Client: litellm
  - Auth: `ANTHROPIC_API_KEY`
- OpenAI — GPT models for class proposal
  - SDK/Client: litellm
  - Auth: `OPENAI_API_KEY`

**Model Hub:**
- Hugging Face — Model downloads (sentence-transformers, SetFit, BGE, etc.)
  - SDK/Client: huggingface_hub, transformers, datasets
  - Auth: `HF_TOKEN` (optional)

## Data Storage

**Databases:**
- None — In-memory and file-based storage only

**Vector/ANN Indexes:**
- FAISS (CPU) — Dense vector similarity search for novelty detection
- HNSWLib — Approximate nearest neighbor indexing
- Connection: In-memory, persisted via pickle/JSON
- Client: faiss-cpu, hnswlib

**File Storage:**
- Local filesystem — Model weights, embeddings, benchmark data, review records
- Formats: pickle, JSON, CSV

**Caching:**
- None — No external caching layer

## Authentication & Identity

**Auth Provider:**
- None — Library/package, no user authentication

**API Key Management:**
- Environment variables via `.env` file
- Keys passed directly to litellm backend calls

## Monitoring & Observability

**Error Tracking:**
- None — Custom exception hierarchy (SemanticMatcherError, ValidationError, TrainingError, MatchingError, ModeError)

**Logs:**
- Python logging via custom `logging_config` module
- Configurable verbosity via `NOVEL_ENTITY_MATCHER_VERBOSE` env var

## CI/CD & Deployment

**Hosting:**
- PyPI — Package distribution via Trusted Publishing

**CI Pipeline:**
- GitHub Actions
  - `lint.yml` — Ruff linting, Black formatting, build validation
  - `test.yml` — pytest (fast PR checks + full matrix on main)
  - `publish.yml` — Build and publish on version tags

## Environment Configuration

**Required env vars:**
- `OPENROUTER_API_KEY` — OpenRouter API key (recommended default)
- `ANTHROPIC_API_KEY` — Anthropic API key (optional)
- `OPENAI_API_KEY` — OpenAI API key (optional)
- `LLM_CLASS_PROPOSER_PROVIDER` — LLM provider selection (default: openrouter)
- `LLM_CLASS_PROPOSER_MODEL` — Model identifier (default: anthropic/claude-sonnet-4)

**Optional env vars:**
- `HF_TOKEN` — Hugging Face token for gated model downloads
- `NOVEL_ENTITY_MATCHER_VERBOSE` — Enable verbose logging

**Secrets location:**
- `.env` file (local development)
- GitHub Actions secrets (CI/CD)
- PyPI trusted publishing (no secrets needed)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Data Ingestion Sources

Built-in entity data loaders (local, no external API):

| Source | Module |
|--------|--------|
| Currencies | `ingestion/currencies.py` |
| Industries | `ingestion/industries.py` |
| Languages | `ingestion/languages.py` |
| Occupations | `ingestion/occupations.py` |
| Products | `ingestion/products.py` |
| Timezones | `ingestion/timezones.py` |
| Universities | `ingestion/universities.py` |

---

*Integration audit: 2026-04-06*

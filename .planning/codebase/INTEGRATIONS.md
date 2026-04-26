# External Integrations

**Analysis Date:** 2026-04-23

## APIs & External Services

**Machine Learning Models:**
- Hugging Face Model Hub - Pre-trained sentence transformer and BERT models
- SDK/Client: sentence-transformers, transformers
- Auth: None (public models), API key for private repos

**LLM Providers (via LiteLLM):**
- OpenAI - GPT models for class proposal
- Anthropic - Claude models for class proposal
- Other providers supported via litellm abstraction
- SDK/Client: litellm
- Auth: Provider-specific API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.)

**Data Sources (Ingestion):**
- DataHub - ISO language and currency codes
- UNSD - UNSPSC product/service categories
- IANA - Timezone database
- O*NET - SOC occupation codes
- GitHub - Industry codes (community datasets)
- SDK/Client: requests
- Auth: None (public APIs)

## Data Storage

**Databases:**
- None (local filesystem only)

**File Storage:**
- Local filesystem - Model checkpoints, embeddings, proposals, benchmarks
- Paths: ./checkpoints/, ./proposals/, ./artifacts/, ./experiments/

**Caching:**
- None (in-memory only during execution)

## Authentication & Identity

**Auth Provider:**
- None (no user authentication)

## Monitoring & Observability

**Error Tracking:**
- None (custom exception handling only)

**Logs:**
- Custom logging via novelentitymatcher.utils.logging_config
- Levels: DEBUG, INFO, WARNING, ERROR
- Optional file handler for debugging
- Verbosity controlled by NOVEL_ENTITY_MATCHER_VERBOSE environment variable

## CI/CD & Deployment

**Hosting:**
- PyPI (pypi.org) - Python package distribution
- GitHub (github.com/minghao51/novel_entity_matcher) - Source code

**CI Pipeline:**
- GitHub Actions - Automated testing and linting
- Workflows:
  - lint.yml - Ruff, Black, build validation
  - test.yml - Pytest (unit, integration, slow)
  - publish.yml - PyPI publishing

**Test Matrix:**
- Python 3.9, 3.10, 3.11, 3.12 (main branch)
- Python 3.11 (PRs and pushes to non-main branches)

## Environment Configuration

**Required env vars:**
- None for basic functionality
- NOVEL_ENTITY_MATCHER_VERBOSE - Optional debug logging

**Optional env vars (LLM features):**
- OPENAI_API_KEY - OpenAI API access
- ANTHROPIC_API_KEY - Anthropic API access
- [PROVIDER]_API_KEY - Other LLM provider keys

**Secrets location:**
- Environment variables only (no secrets in code)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- LLM API calls (via litellm) to providers like OpenAI/Anthropic
- HTTP requests to public data sources during ingestion

---

*Integration audit: 2026-04-23*

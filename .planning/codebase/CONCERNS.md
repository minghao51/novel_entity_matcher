# Codebase Concerns

**Analysis Date:** 2026-05-01

## Tech Debt

**mypy `ignore_errors` overrides for multiple modules:**
- Issue: Several module groups have `ignore_errors = true` in `pyproject.toml`, suppressing all type errors without fixing them
- Files: `pyproject.toml:278-296`
- Impact: Type safety regressions in novelty strategies (`self_knowledge_impl`, `prototypical_impl`, `oneclass_impl`, `setfit_impl`), clustering, storage, and hybrid matcher go undetected
- Fix approach: Progressively fix type errors and remove individual modules from the override list

**Duplicate API key resolution logic:**
- Issue: `_get_api_keys_from_env()` in `llm.py:261-275` and `_provider_to_env_var()` in `llm.py:277-284` maintain the same provider-to-env mapping separately; `litellm.py:23-26` and `litellm.py:44-47` each have their own `_get_api_key()` with different env var (`LITELLM_API_KEY`)
- Files: `src/novelentitymatcher/novelty/proposal/llm.py:261-291`, `src/novelentitymatcher/backends/litellm.py:23-26,44-47`
- Impact: Divergent env var handling across backends; adding a new provider requires editing multiple places
- Fix approach: Consolidate API key resolution into a single shared utility

**Duplicated novelty orchestration paths:**
- Issue: `NovelEntityMatcher` (entity_matcher.py) and `DiscoveryPipeline` (discovery.py) both orchestrate Matcher + NoveltyDetector + LLMClassProposer with overlapping but non-identical logic
- Files: `src/novelentitymatcher/novelty/entity_matcher.py:63-556`, `src/novelentitymatcher/pipeline/discovery.py:59-573`
- Impact: Feature drift between the two entry points; bug fixes may not propagate to both
- Fix approach: Extract shared orchestration into a single internal module both classes delegate to

**Fallback stubs for optional dependencies reduce safety:**
- Issue: `tenacity` and `aiobreaker` fallback stubs in `llm.py:27-57` silently disable retry/circuit-breaker when the optional `llm` extra is not installed; no warning is emitted
- Files: `src/novelentitymatcher/novelty/proposal/llm.py:18-57`
- Impact: Users may unknowingly run without retry protection, causing transient failure storms
- Fix approach: Emit a `logger.warning` when fallback stubs are activated

**`except Exception: pass` blocks swallow errors:**
- Issue: Broad exception handlers with `pass` silently swallow errors in `__init__.py:24-26` (logging config), `llm.py:509` (attribute parsing), and other locations
- Files: `src/novelentitymatcher/__init__.py:24-26`, `src/novelentitymatcher/novelty/proposal/llm.py:509-510`
- Impact: Hard-to-debug failures when logging setup or attribute parsing breaks
- Fix approach: Replace with narrow exception types or at minimum log the exception

## Security Concerns

**Encrypted `.env` file contains API keys (mitigated by dotenvx):**
- Risk: `.env` file contains encrypted API keys for OpenRouter, Anthropic, OpenAI; `.env.keys` contains the private decryption key
- Files: `.env:17` (encrypted OPENROUTER_API_KEY), `.env.keys:6` (DOTENV_PRIVATE_KEY)
- Current mitigation: `.env` and `.env.*` are in `.gitignore`; only `.env.example` is tracked by git; encryption via dotenvx is used
- Recommendations: Confirmed safe — keys are encrypted and not in version control. Consider adding a pre-commit hook to prevent accidental `.env` commits

**API key redaction pattern may miss key formats:**
- Risk: The regex `_API_KEY_PATTERN` in `exceptions.py:131-133` only matches `sk-or-v1-`, `sk-ant-`, `sk-` (20+ chars), and `hf_` prefixed keys; other provider key formats (e.g., Google AI, Azure) would not be redacted
- Files: `src/novelentitymatcher/exceptions.py:131-137`
- Current mitigation: Covers the three main providers used in the codebase
- Recommendations: Add patterns for additional providers as they are integrated; consider a broader catch-all for high-entropy strings in log output

**SHA-1 used for reference signature hash:**
- Risk: SHA-1 is used to compute reference corpus signatures in `detector.py:59-66`; while not used for security purposes, it is deprecated for collision resistance
- Files: `src/novelentitymatcher/novelty/core/detector.py:59`
- Current mitigation: Used only for change detection, not cryptographic purposes
- Recommendations: Low priority — replace with SHA-256 for consistency with best practices

## Performance Issues

**`iterrows()` used in benchmark runners:**
- Problem: `DataFrame.iterrows()` is extremely slow for large datasets, used in 6 benchmark/data pipeline files
- Files: `src/novelentitymatcher/benchmarks/runner.py:297,401,506`, `src/novelentitymatcher/benchmarks/classification/evaluator.py:60`, `src/novelentitymatcher/benchmarks/visualization.py:295`, `src/novelentitymatcher/benchmarks/novelty/evaluator.py:81,91`
- Cause: `iterrows()` creates Series objects per row; vectorized operations are 100-1000x faster
- Improvement path: Replace with vectorized pandas/numpy operations or `.itertuples()`

**Full-reference re-initialization on every detect call when corpus changes:**
- Problem: `NoveltyDetector.detect_novel_samples()` recomputes SHA-1 hash of all reference embeddings on every call and re-initializes all strategies if the signature differs (`detector.py:144-151`)
- Files: `src/novelentitymatcher/novelty/core/detector.py:144-151`
- Cause: Hash computation includes `tobytes()` over the full embedding array plus all labels
- Improvement path: Cache the signature externally and only recompute when the corpus is known to have changed; avoid `tobytes()` for large arrays

**HNSW index has fixed `max_elements` capacity:**
- Problem: `ANNIndex` is initialized with a fixed `max_elements` (default 100K) and raises `ValueError` when exceeded (`index.py:120-123`); no dynamic resizing
- Files: `src/novelentitymatcher/novelty/storage/index.py:120-123`
- Cause: HNSWlib requires pre-allocated index size
- Improvement path: Implement dynamic re-indexing or over-provision with a configurable multiplier

**`_vectors` np.vstack on every add:**
- Problem: `ANNIndex.add_vectors()` uses `np.vstack` to grow the internal vector array on every call (`index.py:128`), causing O(n) copies
- Files: `src/novelentitymatcher/novelty/storage/index.py:128`
- Cause: Pre-allocated numpy arrays are not used; vstack copies all existing data
- Improvement path: Pre-allocate with `max_elements` capacity or use a list buffer with periodic consolidation

## Code Quality Issues

**Very large files exceeding 1000 lines:**
- Files: `src/novelentitymatcher/benchmarks/novelty_bench.py` (1328 lines), `src/novelentitymatcher/novelty/proposal/llm.py` (1181 lines), `src/novelentitymatcher/benchmarks/runner.py` (1093 lines)
- Why fragile: High cognitive load for maintenance; multiple responsibilities in single files
- Safe modification: Extract logical sections into separate modules (e.g., separate prompt building from API calling in `llm.py`)
- Test coverage: Covered but changes risk side effects across large surface areas

**`Config.__getattr__` silently creates nested Config objects:**
- Files: `src/novelentitymatcher/config.py:167-173`
- Why fragile: Accessing any attribute that exists in `_config` returns a value; typos on non-existent keys raise `AttributeError` but dict-valued keys return a new `Config` wrapper, which can mask bugs
- Safe modification: Add an explicit `__dir__` and consider a typed config schema

**`Matcher` class has many delegated property pairs:**
- Files: `src/novelentitymatcher/core/matcher.py:296-341`
- Why fragile: Six pairs of public/private property accessors (`embedding_matcher`/`_embedding_matcher`, etc.) with manual setter delegation to `MatcherComponentFactory` internals
- Safe modification: Consolidate through a single component registry pattern

**`bert_classifier.py` imports `torch` at module level inside methods:**
- Files: `src/novelentitymatcher/core/bert_classifier.py:176,392`
- Why fragile: `torch` is imported inside `train()` and `load()` methods rather than at module level, but the class already depends on `transformers` which depends on `torch`; the conditional import adds complexity without clear benefit since the module already fails without torch transitively

## Missing Features / Gaps

**No async LLM proposal:**
- Problem: `LLMClassProposer.propose_classes()` is synchronous; litellm supports `acompletion` but it is not used
- Files: `src/novelentitymatcher/novelty/proposal/llm.py:293-329`
- Blocks: High-throughput async discovery pipelines must block on LLM calls

**No retry configuration for ANN index persistence:**
- Problem: `ANNIndex` does not support save/load; the index is ephemeral and must be rebuilt on restart
- Files: `src/novelentitymatcher/novelty/storage/index.py` (no `save`/`load` methods)
- Blocks: Production deployments that need to warm-start from a pre-built index

**No input sanitization for LLM prompts:**
- Problem: User-provided entity names and sample texts are interpolated directly into LLM prompts without sanitization
- Files: `src/novelentitymatcher/novelty/proposal/llm.py:330-460` (prompt building methods)
- Blocks: Risk of prompt injection when matching against untrusted input text

**No structured logging output:**
- Problem: All logging uses f-string formatted messages; no JSON structured logging option for production log aggregation
- Files: `src/novelentitymatcher/utils/logging_config.py`
- Blocks: Integration with observability platforms (ELK, Datadog, etc.)

**Missing test coverage for ingestion modules:**
- Problem: The ingestion modules (`universities.py` 384 lines, `products.py` 379 lines, `currencies.py`, `industries.py`, `languages.py`, `occupations.py`, `timezones.py`) have minimal tests — only `test_timezones.py` (5 lines) exists
- Files: `tests/unit/ingestion/` (nearly empty)
- Risk: Web scraping/data fetching changes break silently
- Priority: Low (ingestion is offline tooling)

**No concurrency limit for strategy detection:**
- Problem: `NoveltyDetector.detect_novel_samples()` runs all configured strategies sequentially with no option for parallel execution
- Files: `src/novelentitymatcher/novelty/core/detector.py:158-167`
- Risk: Detection latency scales linearly with number of strategies
- Priority: Medium

---

*Concerns audit: 2026-05-01*

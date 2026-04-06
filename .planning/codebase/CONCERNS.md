# Codebase Concerns

**Analysis Date:** 2026-04-06

## Tech Debt

**Duplicate orchestration logic between NovelEntityMatcher and DiscoveryPipeline:**
- Issue: `novelty/entity_matcher.py` (601 lines) and `pipeline/discovery.py` (621 lines) contain near-identical code for match pipelines — `_build_discovery_pipeline`/`_build_orchestrator`, `_collect_match_result_sync/async`, `_build_match_result`, and `_derive_existing_classes`. Both classes construct the same 5 pipeline stages in the same order.
- Files: `src/novelentitymatcher/novelty/entity_matcher.py`, `src/novelentitymatcher/pipeline/discovery.py`
- Impact: Any fix or feature addition must be duplicated; drift between the two APIs is inevitable
- Fix approach: Extract shared pipeline construction into a single base class or factory; have one class delegate to the other

**Large/Complex Files:**
- Issue: Several files exceed 600+ lines, making them hard to maintain and test
- Files: `src/novelentitymatcher/core/matcher.py` (1226 lines), `src/novelentitymatcher/benchmarks/runner.py` (1087 lines), `src/novelentitymatcher/pipeline/adapters.py` (640 lines), `src/novelentitymatcher/novelty/proposal/llm.py` (638 lines), `src/novelentitymatcher/core/hierarchy.py` (625 lines), `src/novelentitymatcher/pipeline/discovery.py` (621 lines), `src/novelentitymatcher/novelty/entity_matcher.py` (601 lines)
- Impact: Difficult to review changes, higher bug risk, slower onboarding
- Fix approach: Extract cohesive sub-components into separate modules; aim for <400 lines per file

**Broad Exception Handling:**
- Issue: 33 instances of `except Exception` across the codebase, many without specific error handling or logging
- Files: `src/novelentitymatcher/__init__.py:23,31`, `src/novelentitymatcher/pipeline/adapters.py:198,213,591`, `src/novelentitymatcher/core/matcher.py:1147,1191`, `src/novelentitymatcher/core/bert_classifier.py:129,166`, `src/novelentitymatcher/benchmarks/runner.py:617,727,794,986`, `src/novelentitymatcher/novelty/proposal/llm.py:401,494`, and others
- Impact: Swallows unexpected errors, makes debugging difficult, hides real failures
- Fix approach: Replace with specific exception types; add logging for caught exceptions

**Print Statements in Production Code:**
- Issue: 108 `print()` calls in source code instead of proper logging
- Files: Multiple files across `src/novelentitymatcher/`
- Impact: Inconsistent output, no log levels, hard to filter/disable in production
- Fix approach: Replace with `get_logger(__name__)` calls using the existing logging infrastructure

**Strategy auto-registration via side-effect imports:**
- Issue: `novelty/config/base.py:159-171` imports all strategy modules inside `validate()` to trigger decorator-based registration. This is a hidden side-effect — calling validate changes global state.
- Files: `src/novelentitymatcher/novelty/config/base.py`
- Impact: Import order matters; strategies may be double-registered if validate() is called multiple times
- Fix approach: Use explicit registration at module load time or a central registry function

**Inconsistent import styles:**
- Issue: Mix of absolute imports (`from novelentitymatcher.utils.logging_config import ...`) and relative imports (`from ..utils.logging_config import ...`) across the codebase.
- Files: `src/novelentitymatcher/novelty/entity_matcher.py:38`, `src/novelentitymatcher/pipeline/discovery.py:40`, `src/novelentitymatcher/novelty/proposal/llm.py:20` vs. most other files
- Impact: Confusing for contributors; potential issues if package is installed under a different name
- Fix approach: Standardize on relative imports throughout

**Wildcard Import in Public API:**
- Issue: `api.py` uses `from novelentitymatcher import *` pattern internally
- Files: `src/novelentitymatcher/api.py`
- Impact: Unclear dependencies, potential for importing unintended symbols
- Fix approach: Use explicit imports throughout

## Known Bugs

**No known bugs identified from static analysis.** No TODO/FIXME/HACK comments found in the codebase indicating known issues.

## Security Considerations

**NumPy `allow_pickle=True` — arbitrary code execution:**
- Risk: `numpy.load` with `allow_pickle=True` can execute arbitrary Python code from a crafted `.npy`/`.npz` file. If model artifacts are ever loaded from untrusted sources (e.g., user-uploaded, downloaded), this is a remote code execution vector.
- Files:
  - `src/novelentitymatcher/novelty/strategies/prototypical_impl.py:229,233`
  - `src/novelentitymatcher/novelty/strategies/self_knowledge_impl.py:331`
- Current mitigation: None visible
- Recommendations: Use `allow_pickle=False` where possible (prototypes/covariances are plain dicts); for SparseAutoencoder weights, consider `torch.save`/`torch.load` with `weights_only=True` or a safe serialization format

**API Keys in Memory:**
- Risk: API keys stored as instance attributes (`self.api_key`) could be exposed via debugging, logging, or memory dumps
- Files: `src/novelentitymatcher/backends/litellm.py:16-20,30-35`, `src/novelentitymatcher/novelty/proposal/llm.py:111`
- Current mitigation: Keys passed via constructor or read from environment variables
- Recommendations: Avoid storing keys as instance attributes; use key managers or credential providers; ensure keys are never logged

**External HTTP Requests Without Validation:**
- Risk: Multiple ingestion modules fetch data from external URLs without response validation or SSL verification checks
- Files: `src/novelentitymatcher/ingestion/universities.py`, `src/novelentitymatcher/ingestion/products.py`, `src/novelentitymatcher/ingestion/occupations.py`, `src/novelentitymatcher/ingestion/industries.py`, `src/novelentitymatcher/ingestion/timezones.py`, `src/novelentitymatcher/ingestion/languages.py`, `src/novelentitymatcher/ingestion/currencies.py`
- Current mitigation: Timeouts set (30-60s)
- Recommendations: Add response status code validation, content-type checks, and size limits

**Environment Variable Access:**
- Risk: Direct `os.environ` access for setting environment variables (PYTORCH_ENABLE_MPS_FALLBACK)
- Files: `src/novelentitymatcher/core/matcher.py`, `src/novelentitymatcher/backends/static_embedding.py`
- Current mitigation: Uses `setdefault()` which is safe
- Recommendations: Low risk; current approach is acceptable

## Performance Bottlenecks

**Model Loading:**
- Problem: Multiple files load ML models/tokenizers without caching or lazy loading patterns
- Files: `src/novelentitymatcher/core/bert_classifier.py:120-131`, `src/novelentitymatcher/core/matcher.py`
- Cause: Model downloads/initialization on every instantiation
- Improvement path: Implement singleton or factory pattern with model caching

**TF-IDF Computation:**
- Problem: Custom TF-IDF implementation in adapters iterates over all tokens for each document
- Files: `src/novelentitymatcher/pipeline/adapters.py:393-419`
- Cause: O(n*m) complexity where n=documents, m=unique tokens
- Improvement path: Use scipy sparse matrices or sklearn's TfidfVectorizer

**Reference embedding recomputation:**
- Problem: `get_reference_corpus()` in `_EntityMatcher` computes embeddings lazily but caches them on the instance. If the classifier model changes or is retrained, stale embeddings may be used.
- Files: `src/novelentitymatcher/core/matcher.py:255-281`
- Cause: No invalidation mechanism when training data or model changes
- Improvement path: Add a version counter or hash to invalidate cached embeddings

## Fragile Areas

**Matcher Core:**
- Files: `src/novelentitymatcher/core/matcher.py`
- Why fragile: 1226-line file with complex state management, multiple mode handling (zero-shot, head-only, full, hybrid, bert), and broad exception handlers
- Safe modification: Add comprehensive tests before changes; use small, incremental refactors
- Test coverage: Unknown; needs verification

**Pipeline Adapters:**
- Files: `src/novelentitymatcher/pipeline/adapters.py`
- Why fragile: Custom implementations of TF-IDF, TextRank, and keyword extraction with multiple exception handlers
- Safe modification: Test with diverse input data; validate edge cases
- Test coverage: Unknown

**LLM Proposal System:**
- Files: `src/novelentitymatcher/novelty/proposal/llm.py`
- Why fragile: Complex JSON schema validation, multiple LLM provider support, retry logic with broad exception handling
- Safe modification: Test with mock providers; validate schema changes carefully
- Test coverage: Has tests in `tests/test_llm_proposer.py`

**NovelEntityMatcher ↔ DiscoveryPipeline duplication:**
- Files: `src/novelentitymatcher/novelty/entity_matcher.py`, `src/novelentitymatcher/pipeline/discovery.py`
- Why fragile: Two classes with overlapping APIs can diverge silently. `_coerce_novel_sample_report` in NovelEntityMatcher is a coercion shim that masks type mismatches.
- Safe modification: Change both classes in the same PR; add integration tests that exercise both APIs identically
- Test coverage: No tests verify equivalence between the two APIs

**Pydantic schema validation with LLM output:**
- Files: `src/novelentitymatcher/novelty/proposal/llm.py:25-48`
- Why fragile: `LLMProposalSchema` expects a strict JSON structure from LLM responses. If the LLM deviates (even slightly), validation fails and retries are needed.
- Safe modification: Keep retry logic robust; add schema versioning
- Test coverage: Tests mock LLM responses but don't cover all malformed output patterns

## Scaling Limits

**Memory Usage:**
- Current capacity: Embeddings stored in memory; no streaming or chunked processing visible
- Limit: Large datasets will exhaust RAM
- Scaling path: Implement batched processing, disk-backed storage, or ANN index streaming

**Ingestion Modules:**
- Current capacity: Single-threaded HTTP fetching with sequential processing
- Limit: Slow for large external datasets
- Scaling path: Add async/concurrent fetching with rate limiting

**LLM proposal token budget:**
- Current capacity: `token_budget=256` (default in `PipelineConfig`) limits how many novel samples are sent to the LLM
- Limit: Large novel batches get truncated; only first ~256 tokens worth of samples are analyzed
- Scaling path: Hierarchical summarization — cluster first, then send cluster summaries to LLM

## Dependencies at Risk

**No critical dependency risks identified from static analysis.** Review `pyproject.toml` and run `pip-audit` or `safety check` for current vulnerability status.

## Missing Critical Features

**Input Validation:**
- Problem: Limited input validation on public API methods
- Blocks: Reliable error messages for malformed inputs
- Recommendation: Add pydantic validation or explicit type/value checks at API boundaries

**Rate Limiting:**
- Problem: No rate limiting for LLM API calls or external HTTP requests
- Blocks: Production reliability under load
- Recommendation: Add retry with exponential backoff and rate limiting

**Observability/Metrics:**
- Problem: No structured metrics (latency, throughput, error rates) are emitted
- Blocks: Production monitoring and SLO tracking

## Test Coverage Gaps

**Ingestion Modules:**
- What's not tested: External data fetching, parsing edge cases, network failures
- Files: `src/novelentitymatcher/ingestion/universities.py`, `src/novelentitymatcher/ingestion/products.py`, `src/novelentitymatcher/ingestion/occupations.py`, `src/novelentitymatcher/ingestion/industries.py`, `src/novelentitymatcher/ingestion/timezones.py`, `src/novelentitymatcher/ingestion/languages.py`, `src/novelentitymatcher/ingestion/currencies.py`
- Risk: Breakage when external data sources change format or become unavailable
- Priority: Medium

**BERT Classifier:**
- What's not tested: Model training edge cases, tokenizer failures, GPU/CPU fallback
- Files: `src/novelentitymatcher/core/bert_classifier.py`
- Risk: Silent failures during model training
- Priority: Medium

**Hybrid mode matching:**
- What's not tested: The `_match_hybrid` path combining embedding + classifier + reranker
- Files: `src/novelentitymatcher/core/matcher.py`
- Risk: Regression in the most sophisticated matching mode
- Priority: High

**DiscoveryPipeline vs NovelEntityMatcher equivalence:**
- What's not tested: That both classes produce identical results for the same inputs
- Files: `src/novelentitymatcher/novelty/entity_matcher.py`, `src/novelentitymatcher/pipeline/discovery.py`
- Risk: Silent divergence between the two public APIs
- Priority: High

**Strategy persistence (save/load):**
- What's not tested: Round-trip serialization of PrototypicalStrategy and SelfKnowledgeStrategy with `allow_pickle=True`
- Files: `src/novelentitymatcher/novelty/strategies/prototypical_impl.py`, `src/novelentitymatcher/novelty/strategies/self_knowledge_impl.py`
- Risk: Corrupted or incompatible saved models
- Priority: Medium

**Error Handling Paths:**
- What's not tested: Broad `except Exception` branches, network timeout scenarios, malformed API responses
- Files: Multiple files (see Security Considerations)
- Risk: Unhandled edge cases in production
- Priority: High

---

*Concerns audit: 2026-04-06*

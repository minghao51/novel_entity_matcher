# Novel Entity Matcher — Current State

Last updated: 2026-05-12

## What's Implemented

| Feature | Module | Status |
|---------|--------|--------|
| Embedding-based matching (zero-shot) | `core/embedding_matcher.py` | Full — sync + async, thresholding |
| SetFit classification (head-only / full) | `core/classifier.py` | Full — fit, predict, add_class |
| BERT sequence classification | `core/bert_classifier.py` | Full — training, save/load, fp16+MPS guard |
| Hybrid blocking + retrieval | `core/hybrid.py` | Full — TF-IDF + embedding rerank |
| Hierarchical matching | `core/hierarchy.py` | Full — ancestry, paths, DAG traversal |
| Blocking strategies (TF-IDF, BM25, Fuzzy) | `core/blocking.py` | Full |
| Reranker (cross-encoder) | `core/reranker.py` | Full |
| Vector store (in-memory + ChromaDB) | `core/vector_store.py` | Full (ChromaDB optional) |
| Text normalization | `core/normalizer.py` | Full |
| Strategy pattern for matching modes | `core/matching_strategy.py` | Full — 5 strategies + facade |
| Session-example `matcher.py` (God class) | `core/matcher.py` | Full — 1147 lines, 69 methods |
| Embedding backends (ST, static, LiteLLM) | `backends/` | Full — 3 backends |
| Novelty detection (16 strategies) | `novelty/strategies/` | Full — OOD, one-class, energy, LOF, Mahalanobis, GMM, KNN, confidence, pattern, prototypical, SetFit, SetFit centroid, self-knowledge, conformal, clustering, ReAct |
| Singular/plural signal fusion | `novelty/core/signal_combiner.py` | Full |
| Score calibrator | `novelty/core/score_calibrator.py` | Full |
| Adaptive weight optimization | `novelty/core/adaptive_weights.py` | Full |
| Clustering (HDBSCAN, UMAP, Leiden, Louvain) | `novelty/clustering/` | Full — scalable, incremental, graph, stability, validation |
| Drift detection (snapshot-based) | `novelty/drift/` | Full |
| ANN index (HNSWLib, FAISS, exact) | `novelty/storage/index.py` | Full — pluggable backends |
| Proposal/review workflow (LLM, DSPy) | `novelty/proposal/` | Full — retrieval-augmented, DSPy optimization, conflict resolution |
| Active learning (sampling, annotation) | `novelty/active_learning/` | Full |
| Evaluation framework | `novelty/evaluation/` | Full — splitters, metrics, evaluator |
| Discovery pipeline | `pipeline/discovery.py` | Full — match, discover, approve, promote |
| Pipeline orchestration | `pipeline/orchestrator.py` | Full — sequential stage execution |
| Pipeline contracts + stages | `pipeline/contracts.py`, `pipeline/stages/` | Full |
| NovelEntityMatcher (matcher-first) | `novelty/entity_matcher.py` | Full |
| Ingestion (7 datasets: unis, occupations, products, industries, currencies, languages, timezones) | `ingestion/` | Full — concurrent fetchers |
| Benchmark runner + CLI | `benchmarks/` | Full — entity resolution, classification, novelty, infra, weight optimization |
| Monitoring / metrics | `monitoring/` | Full |
| Config management | `config.py`, `novelty/config/` | Full — Pydantic settings + YAML |
| Lazy-loading public API | `__init__.py`, `api.py` | Full — deferred imports via `__getattr__` |

## Stubbed / Unimplemented

All `NotImplementedError`, `TODO`, `FIXME`, `HACK`, `XXX` queries return **zero results** — the codebase contains none.

Placeholder `pass` statements found (all intentional, not stubs):

| Location | Context |
|----------|---------|
| `backends/base.py:9` | Abstract method body (ABC pattern) |
| `core/matching_strategy.py:143,236` | `build_index()` no-op for head-only/hybrid strategies |
| `core/bert_classifier.py:222` | Silence unused gradient debug output |
| `core/bert_classifier.py:394` | Graceful fallback when `torch` not installed |
| `novelty/proposal/llm.py:62,68` | Circuit breaker stub when `aiobreaker` not available |
| `novelty/storage/review.py:194` | Empty `retrain_callback` — no retrain logic on promotion |
| `benchmarks/novelty_bench.py:400,434` | Silent error suppression in LOF/OneClass benchmarks |
| `benchmarks/weight_optimizer.py:92,116` | Bypass trials when Optuna not installed |
| `backends/static_embedding.py:54` | model2vec import fallback catch-all |
| `novelty/clustering/incremental.py:194` | Placeholder in incremental update path |

No unfinished features detected — all public API surface is wired.

## Known Bugs

| Severity | Issue | Location |
|----------|-------|----------|
| **High** | LLM response `choices[0].message.content` accessed without null check — empty `choices` or `None` content crashes JSON parsing | `novelty/proposal/llm.py:1169` |
| **High** | Persistence deserialization uses bare `data["timestamp"]` / `data["novel_sample_report"]` with no `KeyError` guard — corrupt YAML produces opaque failure | `novelty/storage/persistence.py:252` |
| **Medium** | `novelty_bench.py` methods catch `ValueError, RuntimeError` and silently `pass` — LOF and OneClassSVM failures invisible | `benchmarks/novelty_bench.py:399-434` |
| **Medium** | 6 bare `except Exception` handlers mask real error types across the codebase | `__init__.py:33`, `utils/embeddings.py:67`, `utils/benchmarks.py:258,437`, `novelty/proposal/llm.py:1093`, `ingestion/cli.py:108` |
| **Low** | `mypy` `warn_return_any = false` suppresses warnings for all `Any` return types (`config.py`, `backends/litellm.py`, vector store) | `pyproject.toml:219` |
| **Low** | Static embedding backend integration tests skipped (`reason="RikkaBotan model requires additional SSE module dependencies"`) | `tests/integration/backends/test_static_embedding.py:10` |
| **Low** | MPS fp16 warning uses `warnings.warn()` line 184 but ModelCache eviction uses raw `min()` — not consistent | `core/bert_classifier.py:182-186` |

## Security Concerns

| Severity | Issue | Location |
|----------|-------|----------|
| **High** | `__import__` with user-configurable module name in ingestion CLI — attack surface if `name` were ever user-controlled (currently hardcoded dict, but no `AttributeError` guard) | `ingestion/cli.py:92-96` |
| **Medium** | `trust_remote_code=True` passed to `get_cached_sentence_transformer` — model repos execute arbitrary code at load time | `backends/static_embedding.py:62`, `utils/embeddings.py:159-175` |
| **Medium** | `.env.keys` contains `DOTENV_PRIVATE_KEY` for `.env` decryption — gitignored via `.env.*` + `!.env.example`, but misconfigured `.gitignore` would leak | `.env.keys` (gitignored) |
| **Low** | API keys passed through env vars (`OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) — logged at debug level via exception redaction in `_redact_api_keys` | `exceptions.py` |
| **Low** | No fuzzing or adversarial input tests in benchmark loader for CSV/YAML deserialization | `benchmarks/loader.py` |

No hardcoded secrets or API keys found in source code.

## Performance Issues

| Issue | Location | Impact |
|-------|----------|--------|
| Redundant embedding encoding in benchmarks — each benchmark method independently encodes train/val/test splits (15+ encodings for `run_depth("full")`) | `benchmarks/novelty_bench.py` (all benchmark methods) | **Critical** — encoding is dominant cost, pre-encoding would cut runtime 10x |
| `ModelCache._estimate_model_bytes` returns `0` on any exception — cached entry never evicted, 4 GB budget never enforced | `utils/embeddings.py:67-68` | **High** — memory leak under failure |
| Mahalanobis covariance inversion uses `np.linalg.inv` without singularity check — ill-conditioned on small/high-dim data | `benchmarks/novelty_bench.py:365` | **Medium** — runtime error on edge cases |
| N+1 pattern: `_build_strategy_outputs` re-initializes 4 strategies per call, called separately for val/test and each ensemble method | `benchmarks/novelty_bench.py:236-296` | **Medium** — redundant object construction |
| Strategy `initialize()` + `detect()` in 16 strategies independently handle embeddings — no shared helper pattern beyond abstract base | `novelty/strategies/*_impl.py` | **Low** — code duplication, not runtime |
| MPS fp16 fallback recomputes `torch.backends.mps.is_available()` on every `fit` call | `core/bert_classifier.py:174-187` | **Low** — negligible overhead |
| `hierarchy.py` uses nested iteration for ancestry/descendant traversal — no adjacency caching for repeated lookups | `core/hierarchy.py:68-74,423-491` | **Low** — fine for typical entity counts (<10K) |

## Maintenance Issues

| Issue | Detail | Severity |
|-------|--------|----------|
| **37 `type: ignore` comments** across 19 source files | Most in `benchmarks/` (10), `core/matching_strategy.py` (7), `novelty/proposal/llm.py` (3), `core/hybrid.py` (2), `core/vector_store.py` (2), `core/bert_classifier.py` (2) | **High** — suppresses real typing issues |
| **4 mypy override blocks with `ignore_errors = true`** | Entire modules bypass type checking: `novelty.strategies.*`, `novelty.clustering.*`, `novelty.storage.*`, `core.hybrid`, `self_knowledge_impl`, `prototypical_impl`, `oneclass_impl`, `setfit_impl` | **High** — any type regression in 30+ files goes undetected |
| `disallow_untyped_defs = false` | All functions can omit type annotations — weakens mypy value | **Medium** |
| `warn_return_any = false` | Suppresses all warnings for `Any` return types | **Medium** |
| Python version mismatch: `requires-python = ">=3.10"` vs `mypy == "3.11"` | `pyproject.toml:216` and `pyproject.toml:6` | **Low** |
| `novelty_bench.py` (1444 lines) — heavy duplication across 15 benchmark methods | Same encode-prepare-run-report pattern repeated; template method would halve file | **Medium** |
| `llm.py` (1268 lines) — single class with 19 methods, 6 exception categories in fallback chain | `LLMClassProposer` manages model selection, retry, circuit breaker, DSPy, summarization, schema discovery, response parsing | **Medium** |
| `runner.py` (1096 lines) — `run_novelty_on_processed` method alone is 196 lines | | **Medium** |
| `Matcher` god class (1147 lines, 69 methods) — 5 modes, implicit state machine, no formal state validation | `core/matcher.py` | **Medium** |
| Depth-dependent import chain: `pipeline/` → `novelty/` → `core/` → `utils/` | Changes in low-level utilities cascade unpredictably | **Medium** |
| Backwards-compatible aliases: `_coerce_texts`, `_extract_top_prediction_metadata`, `_resolve_threshold` | `core/matcher.py:49-51` — no external callers | **Low** |
| `print()` instead of `logger.info()` in `novelty_bench.py` lines 671, 739, 805, 889 | Inconsistent with rest of codebase | **Low** |
| Hardcoded `_SCORE_KEYS` / `_FLAG_KEYS` lists — manual update needed for new strategies | `novelty/core/signal_combiner.py:21-62` | **Low** |
| Heavy ML deps (`torch`, `transformers`, `sentence-transformers`, `setfit`) as core dependencies, not optional | `pyproject.toml:24-43` — ~4 GB download even for static matching only | **Low** |
| Large data files in `data/raw/` (NAICS, UNSPSC, universities) bundled in git | Should be downloaded at build time | **Low** |
| Accumulating output dirs: `proposals/`, `benchmark_results/`, `artifacts/`, `experiments/` | Gitignored but no auto-cleanup | **Low** |

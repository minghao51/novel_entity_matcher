# Concerns

Tech debt, bugs, security, performance, and maintainability issues found in the codebase.

---

## Critical

### C-01: LLM response access without null checks

`src/novelentitymatcher/novelty/proposal/llm.py:1169`

```python
return response.choices[0].message.content
```

`choices` may be empty, and `content` can be `None` (e.g., tool-call responses). Will raise `IndexError` or return `None` into JSON parsing, producing opaque failures.

### C-02: Persistence deserialization lacks KeyError guards

`src/novelentitymatcher/novelty/storage/persistence.py:252`

```python
timestamp = datetime.fromisoformat(data["timestamp"])
```

`_dict_to_report` accesses `data["timestamp"]`, `data["novel_sample_report"]["novel_samples"]`, etc. without defensive checks. A corrupt or version-mismatched YAML file raises raw `KeyError` with no context.

### C-03: mypy ignores errors on broad module swaths

`pyproject.toml:229-247`

Four mypy override blocks set `ignore_errors = true` for:
- `novelentitymatcher.novelty.strategies.*`
- `novelty.clustering.*`
- `novelty.storage.*`
- `core.hybrid`
- `self_knowledge_impl`, `prototypical_impl`, `oneclass_impl`, `setfit_impl`

These modules have zero type-checking. Any type regression goes undetected.

---

## Security

### S-01: Dynamic `__import__` in ingestion CLI

`src/novelentitymatcher/ingestion/cli.py:92`

```python
module = __import__(
    f"novelentitymatcher.ingestion.{name}",
    fromlist=[name.capitalize() + "Fetcher"],
)
fetcher_cls = getattr(module, name.capitalize() + "Fetcher")
```

`name` comes from the hardcoded `INGESTORS` dict, so the attack surface is limited. However, `getattr` has no `AttributeError` handling if the expected class name doesn't exist in the module.

### S-02: `trust_remote_code` in model loading

`src/novelentitymatcher/utils/embeddings.py:159-175`

`get_cached_sentence_transformer` accepts `trust_remote_code=False` by default. Callers can pass `True`, which executes arbitrary code from model repos. Not a vulnerability today, but a risk surface.

### S-03: `.env.keys` contains private decryption key

`.env.keys:6` contains `DOTENV_PRIVATE_KEY` for decrypting `.env`. The file is gitignored via `.env.*` (with `!.env.example` exception), but a misconfigured `.gitignore` would leak it.

---

## Performance

### P-01: Redundant embedding encoding in benchmarks

`src/novelentitymatcher/benchmarks/novelty_bench.py`

Every benchmark method (`benchmark_knn`, `benchmark_mahalanobis`, `benchmark_lof`, `benchmark_oneclass_svm`, `benchmark_isolation_forest`, `benchmark_setfit_centroid`, `benchmark_uncertainty`, `benchmark_self_knowledge`, `benchmark_prototypical`, `benchmark_setfit`, `benchmark_ensemble`, `benchmark_ensemble_adaptive`, `benchmark_signal_combiner`) independently calls:
```python
train_emb = self.encode_texts(split.train_texts)
val_emb = self.encode_texts(split.val_texts)
test_emb = self.encode_texts(split.test_texts)
```

For a `run_depth("full")` call, the same texts are encoded 12+ times. Encoding is the dominant cost (seconds per call). Pre-encoding once and passing embeddings through would cut total benchmark runtime by an order of magnitude.

### P-02: Unbounded model cache with imprecise eviction

`src/novelentitymatcher/utils/embeddings.py:23-76`

`ModelCache` uses `_estimate_model_bytes` which estimates memory from config attributes, returning `0` on any exception (line 67-68). If estimation fails, entries are cached with zero memory footprint and never evicted. The 4 GB default budget is never exhausted.

### P-03: Mahalanobis covariance inversion without singularity check

`src/novelentitymatcher/benchmarks/novelty_bench.py:365`

```python
global_cov = np.cov(train_emb, rowvar=False) + 1e-6 * np.eye(train_emb.shape[1])
cov_inv = np.linalg.inv(global_cov)
```

With small training sets or high-dimensional embeddings, the regularized covariance may still be ill-conditioned. Should use `np.linalg.pinv` or check condition number.

### P-04: N+1 pattern in novelty benchmark strategy construction

`src/novelentitymatcher/benchmarks/novelty_bench.py:236-296`

`_build_strategy_outputs` re-initializes four strategies (ConfidenceStrategy, KNNDistanceStrategy, PatternScorer, SetFitCentroidStrategy) for every call. Called separately for val and test, for every ensemble method.

---

## Code Complexity

### X-01: `novelty_bench.py` — 1444 lines, heavy duplication

The `NoveltyBenchmark` class repeats the same pattern across 15 benchmark methods:
1. Encode texts
2. Prepare labels
3. Run strategy
4. Compute metrics
5. Build result

Only steps 3-4 vary. A template method or shared helper would halve the file size.

### X-02: `llm.py` — 1268 lines, fallback chain complexity

`LLMClassProposer` manages model selection, retry logic, circuit breaker, DSPy delegation, hierarchical summarization, schema discovery, and response parsing in a single class. The `_call_llm_with_fallback` method (lines 1026-1109) has 6 exception categories with different handling. The class has 4 public methods and 15 private methods.

### X-03: `runner.py` — 1096 lines, orchestration sprawl

`BenchmarkRunner` handles entity resolution, classification, novelty, processed-ood novelty, auto-thresholding, and result persistence. The `run_novelty_on_processed` method alone is 196 lines (lines 806-1001).

### X-04: `Matcher` god class — 708 lines, 5 modes, implicit state machine

`src/novelentitymatcher/core/matcher.py`

The `Matcher` class manages:
- Training mode detection (`_detect_training_mode`)
- Strategy selection (`_get_strategy`, `_select_matcher`)
- Sync/async matching (`match`, `match_async`, `_match_sync_impl`, `_match_async_impl`)
- Metadata matching (`_match_with_metadata`, `_match_with_metadata_async`)
- Batch processing (delegated to `_BatchEngine`)
- Diagnostics (delegated to `_DiagnosisEngine`)
- Component lifecycle (`_components`)
- Async executor lifecycle

State (`_training_mode`, `_detected_mode`, `_active_matcher`, `_has_training_data`) is managed implicitly across `__init__`, `fit`, `fit_async`, and `match`. No formal state machine or validation of state transitions.

### X-05: Novelty strategy implementations share no common helper

Each strategy (knn_distance, confidence, pattern, uncertainty, energy, setfit, etc.) independently implements `initialize()` and `detect()` with nearly identical boilerplate for embedding handling and metric extraction. No shared base helper exists beyond the abstract `BaseNoveltyStrategy`.

---

## Missing Error Handling

### E-01: Benchmark methods silently swallow errors

`src/novelentitymatcher/benchmarks/novelty_bench.py:399,433`

```python
except (ValueError, RuntimeError):
    pass
```

LOF and OneClassSVM benchmark methods catch exceptions and silently discard results. Failures are invisible to the user.

### E-02: Bare `except Exception` catch-alls

6 instances across the codebase:
- `src/novelentitymatcher/novelty/proposal/llm.py:1093` — defensive wrapper
- `src/novelentitymatcher/utils/embeddings.py:67` — model size estimation
- `src/novelentitymatcher/__init__.py:33` — version fallback
- `src/novelentitymatcher/utils/benchmarks.py:258,437` — monkeypatch-exercised paths
- `src/novelentitymatcher/ingestion/cli.py:108` — concurrent ingestion

Each masks the real exception type and makes debugging harder.

### E-03: Ingestion CLI `getattr` without `AttributeError` guard

`src/novelentitymatcher/ingestion/cli.py:96`

```python
fetcher_cls = getattr(module, name.capitalize() + "Fetcher")
```

If a module doesn't export a class with the expected name, this raises `AttributeError` with no helpful message.

---

## Test Coverage Gaps

### T-01: 43 source files have no dedicated test files

Key untested modules:
- All novelty strategies: `knn_distance.py`, `confidence.py`, `pattern.py`, `uncertainty.py`, `energy.py`, `setfit.py`, `setfit_centroid.py`, `prototypical.py`, `clustering.py`, `oneclass.py`
- All novelty strategy implementations: `oneclass_impl.py`, `prototypical_impl.py`, `setfit_impl.py`, `self_knowledge_impl.py`, `pattern_impl.py`
- All clustering: `scalable.py`, `backends.py`, `graph.py`, `incremental.py`, `stability.py`, `params.py`
- Novelty schemas: `models.py`, `results.py`, `reports.py`
- Novelty storage: `index.py`, `review.py`
- Novelty proposal: `llm.py`, `retrieval.py`, `schema_enforcement.py`
- Novelty evaluation: `evaluator.py`, `splitters.py`
- Pipeline stages: `drift_hook.py`
- Top-level: `api.py`, `config_registry.py`, `exceptions.py`

### T-02: Static embedding backend tests are skipped

`tests/integration/backends/test_static_embedding.py:10`

```python
@pytest.mark.skip(reason="RikkaBotan model requires additional SSE module dependencies")
```

An entire backend has no active test coverage.

### T-03: Integration tests dominate, unit coverage may be thin

912 total test functions, but many are integration tests requiring model downloads or network access. Unit test isolation for core logic (strategy detection, signal combination, weight optimization) is unclear.

---

## Tech Debt

### D-01: Python version inconsistency

`pyproject.toml` declares `requires-python = ">=3.10"` but `tool.mypy.python_version = "3.11"`. Type checking may not match runtime behavior on 3.10 or 3.12.

### D-02: `disallow_untyped_defs = false` in mypy config

`pyproject.toml:180`

All functions can be defined without type annotations, weakening the value of mypy. Progress toward strict typing is not tracked.

### D-03: Backwards-compatible aliases add noise

`src/novelentitymatcher/core/matcher.py:49-51`

```python
_coerce_texts = coerce_texts
_extract_top_prediction_metadata = extract_top_prediction_metadata
_resolve_threshold = resolve_threshold
```

These private-module aliases for functions moved to `matcher_shared.py` serve no purpose if no external caller depends on them.

### D-04: `novelty_bench.py` uses `print()` instead of logging

Lines 671, 739, 805, 889, etc. use `print(f"...")` for progress output. Should use `logger.info()` for consistency with the rest of the codebase and to allow output control.

### D-05: Hardcoded strategy score/flag key lists

`src/novelentitymatcher/novelty/core/signal_combiner.py:21-62`

`_SCORE_KEYS` and `_FLAG_KEYS` are hardcoded lists that must be manually updated when a new strategy is added. Adding a new strategy requires touching this file, or the new signals are silently ignored in fusion.

### D-06: Package bundles heavy ML dependencies unconditionally

`pyproject.toml:24-43` — `torch`, `transformers`, `sentence-transformers`, `setfit`, `optuna`, `matplotlib` are all core dependencies. Users who only need static embedding matching still download ~4 GB of ML libraries. The `opinion` extras group adds more, but the split is not clean.

---

## Maintainability

### M-01: Deep import chains

`pipeline/` → `novelty/` → `core/` → `utils/` creates a 4-layer dependency chain. Changes in low-level utilities can cascade unpredictably through novelty detection and pipeline orchestration.

### M-02: Large data files in `data/raw/`

`data/raw/industries/naics_2022.json`, `data/raw/products/unspsc.json`, and `data/raw/universities/universities.json` are bundled in the repo. These may be large and should be downloaded at build/runtime rather than stored in version control.

### M-03: Accumulating output directories

`proposals/`, `benchmark_results/`, `artifacts/`, and `experiments/` accumulate generated files. While gitignored, they create clutter in local development and are not cleaned automatically.

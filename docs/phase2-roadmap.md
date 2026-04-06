# Phase 2 Roadmap: Signal Fusion, Pipeline Contracts & Discovery Quality

**Last Updated:** 2026-04-04  
**Status:** Active  
**Predecessor:** [technical-roadmap.md](./technical-roadmap.md)

## Purpose

This document defines the next phase of implementation work for Novel Entity Matcher. It builds on the existing novelty detection infrastructure and roadmap phases, focusing on six concrete improvements that raise detection quality, improve architecture clarity, and reduce operational cost.

## Initiatives

### 1. Model-Based Signal Fusion

**Problem:** The current `SignalCombiner` uses fixed weighted averages. Weights like `knn=0.45`, `confidence=0.35` are hand-tuned and don't adapt to dataset characteristics or capture interactions between signals.

**Target:** Replace the mathematical weighted average with a learnable meta-learner that combines strategy outputs into a final novelty decision.

**Approach:**
1. Start with **logistic regression** as the meta-learner — interpretable, low overfitting risk, works with small labeled OOD datasets
2. Train on existing benchmark datasets (ag_news, go_emotions) using held-out OOD samples
3. Extract per-sample strategy scores as features, novelty label as target
4. Keep the mathematical fallback for cases where no training data is available

**Success Criteria:**
- AUROC improvement of ≥5% over fixed-weight baseline on ag_news
- No regression on go_emotions or entity-name datasets
- Fallback to weighted average when no labeled OOD data exists
- Model weights are inspectable (feature importance via coefficients)

**Files to Modify:**
- `src/novelentitymatcher/novelty/core/signal_combiner.py` — add `MetaLearnerCombiner` class
- `src/novelentitymatcher/novelty/config/base.py` — add `combine_method="meta_learner"` option
- `src/novelentitymatcher/novelty/config/weights.py` — add meta-learner configuration fields

---

### 2. Switch Default Retrieval Model to Potion-32M

**Problem:** The README lists `potion-32m` as available but defaults to `potion-8m` for retrieval. Potion-32M offers better quality with acceptable speed.

**Target:** Make `potion-32m` the default retrieval model.

**Approach:**
1. Update the default model alias in `Matcher` and related configuration
2. Verify all benchmark scripts and tests still pass
3. Update documentation to reflect the new default

**Success Criteria:**
- `Matcher(mode="zero-shot")` uses `potion-32m` by default
- All existing tests pass
- Benchmark results show no regression in match quality
- Documentation updated consistently

**Files to Modify:**
- `src/novelentitymatcher/core/matcher.py` — default model parameter
- `src/novelentitymatcher/config_registry.py` — default model entry
- `docs/models.md` — update default documentation
- `README.md` — update default mention

---

### 3. SetFit Centroid Distance Continuous Scoring

**Problem:** The docs claim SetFit Centroid Distance achieves AUROC 0.886, but the current `setfit` strategy only produces binary `is_novel` flags. The continuous distance-to-centroid score is lost.

**Target:** Implement a proper centroid distance strategy that outputs continuous novelty scores, matching the documented benchmark methodology.

**Approach:**
1. Create `SetFitCentroidStrategy` that computes minimum cosine distance to class centroids in SetFit embedding space
2. Output continuous `centroid_distance_score` metric (not just binary flag)
3. Calibrate threshold using reference set percentile (existing approach in `setfit_impl.py`)
4. Register as `setfit_centroid` strategy with weight 0.45 (matching documented importance)

**Success Criteria:**
- AUROC ≥ 0.88 on ag_news benchmark (matching documented 0.886)
- Continuous scores usable by signal fusion (not just binary flags)
- Works with SetFit full training mode (requires fine-tuned embeddings)
- Graceful fallback when SetFit model not available

**Files to Create/Modify:**
- `src/novelentitymatcher/novelty/strategies/setfit_centroid.py` — new strategy
- `src/novelentitymatcher/novelty/config/strategies.py` — add `SetFitCentroidConfig`
- `src/novelentitymatcher/novelty/core/signal_combiner.py` — add `setfit_centroid` to weight map
- `docs/methodology/overview.md` — clarify distinction between setfit and setfit_centroid

---

### 4. Internal Pipeline Contracts

**Problem:** The current architecture chains matching, novelty detection, and discovery through `NovelEntityMatcher` without clear stage boundaries. This makes it hard to reorder, replace, or independently test stages.

**Target:** Define explicit stage I/O interfaces so the discovery pipeline becomes composable.

**Approach:**
1. Define `StageContext` — carries embeddings, labels, metadata between stages
2. Define `StageResult` — standardized output shape for each stage
3. Define `PipelineStage` protocol — `initialize()`, `execute()`, `validate()`
4. Create adapters around existing capabilities rather than rewriting:
   - `EmbeddingStage` — wraps static/sentence transformer backends
   - `MatchingStage` — wraps Matcher routes
   - `NoveltyStage` — wraps multi-strategy detector
   - `ClusteringStage` — wraps HDBSCAN/k-means backends
5. Keep `Matcher` and `NovelEntityMatcher` as public facades over the stage architecture

**Success Criteria:**
- End-to-end discovery flow executes through stage contracts
- Each stage is independently testable with mock inputs/outputs
- Existing public APIs continue to work unchanged
- Stage interfaces documented with type hints and docstrings

**Files to Create/Modify:**
- `src/novelentitymatcher/pipeline/stages.py` — stage protocol and context/result types
- `src/novelentitymatcher/pipeline/adapters.py` — adapters for existing capabilities
- `src/novelentitymatcher/pipeline/orchestrator.py` — basic pipeline executor
- `src/novelentitymatcher/pipeline/__init__.py` — public exports

---

### 5. Adaptive Strategy Weights

**Problem:** Fixed strategy weights don't account for dataset characteristics. kNN works well for dense clusters but poorly for sparse data; confidence scores are random for some datasets but informative for others.

**Target:** Learn or estimate optimal strategy weights based on dataset characteristics rather than using fixed values.

**Approach:**
1. Compute dataset characteristics:
   - **Class separability**: average inter-class vs intra-class distance ratio
   - **Sample density**: average kNN distance within classes
   - **Embedding dimensionality**: effective dimensionality via PCA explained variance
   - **Class balance**: entropy of class distribution
2. Map characteristics to weight adjustments using a simple rule-based system initially:
   - High separability → increase centroid distance weight
   - High density → increase kNN weight
   - Low sample count → decrease LOF/Mahalanobis weight (unstable with few samples)
   - High dimensionality → increase uncertainty weight
3. Optionally train a weight predictor on benchmark datasets for future improvement

**Success Criteria:**
- Adaptive weights outperform fixed weights on ≥2 of 3 benchmark datasets
- Weight adjustments are logged and inspectable
- No regression on any dataset compared to fixed baseline
- Configuration option to force fixed weights when needed

**Files to Create/Modify:**
- `src/novelentitymatcher/novelty/core/adaptive_weights.py` — new module
- `src/novelentitymatcher/novelty/config/weights.py` — add `adaptive=True` option
- `src/novelentitymatcher/novelty/core/signal_combiner.py` — integrate adaptive weight resolution

---

### 6. Cluster-Level Evidence Extraction

**Problem:** LLM proposal generation operates at the sample level, which is expensive and produces inconsistent proposals. The roadmap calls for cluster-level evidence extraction before LLM calls.

**Target:** Add a cluster summarization layer that extracts statistical evidence from novel clusters before passing to LLM proposers.

**Approach:**
1. For each novel cluster, extract:
   - **Statistical keywords**: top-N TF-IDF or chi-squared terms distinguishing cluster from reference
   - **Representative examples**: 3-5 samples closest to cluster centroid
   - **Cluster statistics**: size, density, average distance to nearest known class
   - **Token-budget-aware context**: pack evidence into a bounded context window
2. Pass summarized evidence to LLM proposer instead of raw sample lists
3. Reduce LLM calls from one-per-sample to one-per-cluster

**Success Criteria:**
- LLM cost reduced by ≥80% (one call per cluster vs per sample)
- Proposal quality maintained or improved (measured by human review or benchmark)
- Evidence extraction is deterministic and reproducible
- Works with or without LLM (evidence useful for human review even without LLM)

**Files to Create/Modify:**
- `src/novelentitymatcher/novelty/proposal/evidence_extractor.py` — new module
- `src/novelentitymatcher/novelty/proposal/cluster_proposer.py` — cluster-level LLM proposer
- `src/novelentitymatcher/novelty/config/proposal.py` — add evidence extraction config
- `src/novelentitymatcher/novelty/schemas.py` — add `ClusterEvidence` schema

---

## Sequencing & Dependencies

```
1. Model-Based Signal Fusion ──────────────────────────────────┐
2. Potion-32M Default ─────────────────────────────────────────┤ (independent, can run in parallel)
3. SetFit Centroid Distance ───────────────────────────────────┤
                                                              │
4. Pipeline Contracts ────────────────────────────────────────┤ (depends on stable strategy outputs)
5. Adaptive Strategy Weights ─────────────────────────────────┘ (depends on signal combiner work from #1)
                                                              │
6. Cluster-Level Evidence Extraction ──────────────────────────┘ (depends on pipeline contracts from #4)
```

**Execution Order:**
1. Start with #2 (Potion-32M default) — smallest change, immediate benefit
2. Parallelize #1, #3 — both improve signal quality independently
3. #5 (Adaptive weights) — builds on signal combiner improvements from #1
4. #4 (Pipeline contracts) — architectural foundation
5. #6 (Evidence extraction) — depends on #4 for cluster stage interface

## Testing Strategy

Each initiative must include:
- Unit tests for new functionality
- Benchmark comparison against current baseline (AUROC, DR@1%, latency)
- No regression on existing test suite
- Contract tests for pipeline stage I/O (initiative #4)

## Definition of Done

All six initiatives are complete when:
- All tests pass (existing + new)
- Benchmarks show improvement or no regression on ag_news, go_emotions, and entity-name datasets
- Documentation updated (methodology overview, models guide, architecture doc)
- Public APIs remain backward compatible
- Technical roadmap updated to reflect completed work

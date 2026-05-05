# Technical Roadmap: Novel Entity Matcher

**Last Updated:** 2026-05-05
**Status:** Active technical plan
**Version Path:** 0.1.0 package today → 1.0.0 architecture target
**Supersedes:** `technical-roadmap.md` (2026-03-24), `20260505-technical_roadmap.md` (draft)

---

## Table of Contents

1. [Context & Architecture](#context--architecture)
2. [Current vs Target Gap Analysis](#current-vs-target-gap-analysis)
3. [Completed Milestones](#completed-milestones)
4. [Quick Wins](#quick-wins)
5. [B: Novelty & OOD Depth](#b-novelty--ood-depth)
6. [C: Clustering & Discovery](#c-clustering--discovery)
7. [D: Pipeline & Infrastructure](#d-pipeline--infrastructure)
8. [E: LLM & HITL](#e-llm--hitl)
9. [Deferred Features](#deferred-features)
10. [Dependency Graph](#dependency-graph)
11. [Suggested Sequencing](#suggested-sequencing)
12. [Risks and Mitigations](#risks-and-mitigations)

---

## Context & Architecture

### Repository Baseline

The published package is at `0.1.0` in `pyproject.toml`. Two public surfaces exist:
- `Matcher` — primary general-purpose matching API
- `NovelEntityMatcher` — novelty-aware orchestration layer
- `DiscoveryPipeline` — pipeline-first discovery entry point

The README positions the library as a text-to-entity matcher with optional novelty detection, async support, benchmarking, and optional LLM-backed proposal features.

### Implemented Architecture

```
core/
  unified Matcher, zero-shot / SetFit / BERT / hybrid routes
  async execution, blocking, reranking, hierarchy, monitoring, normalization
novelty/
  modular detector core + strategy registry
  strategies: confidence, KNN distance, clustering, one-class, pattern,
  prototypical, self-knowledge, Mahalanobis, LOF
  clustering (HDBSCAN / sOPTICS / UMAP+HDBSCAN), proposal (LLM),
  schema enforcement, storage (ANN + review), config, evidence extraction
backends/
  static embeddings, sentence-transformer, reranker, LiteLLM integration
benchmarks/
  runner / registry / CLI scaffolding
ingestion/
  dataset-specific modules + CLI
pipeline/
  5-stage orchestrator: match → OOD → cluster → evidence → propose
  PipelineBuilder, PipelineOrchestrator, PipelineConfig
  DiscoveryPipeline as pipeline-first public API
```

### Strengths Already Present

- Unified matcher API with sync and async flows
- Training-aware model resolution and static-embedding defaults
- Multi-strategy novelty subsystem with pluggable strategies, evaluation, schemas, persistence
- ANN (HNSWlib / FAISS / exact), scalable clustering, LLM proposal with circuit breaker
- 5-stage pipeline orchestrator with named stages and stable stage I/O
- HITL review lifecycle: pending → approved/rejected → promoted
- Schema-aware proposal with attribute discovery
- Broad test coverage across public API surface

### Architectural Principles

1. Pipeline-first orchestration as the main internal design.
2. Existing matcher and novelty functionality reused where sound.
3. Breaking API changes acceptable when they simplify the architecture materially.
4. Configuration drives stage selection and optional capabilities.
5. LLM usage stays optional, bounded, and cost-aware.
6. Promotion of new classes is explicit, auditable, and computationally efficient.

---

## Current vs Target Gap Analysis

| Area | Current State | Target State | Gap | Status |
|---|---|---|---|---|
| Matching API | `Matcher` is the main public interface | Matching becomes one subsystem within a broader pipeline | Medium | **Done** — `DiscoveryPipeline` wraps `Matcher` |
| Novelty detection | Multi-strategy detector exists | Conformalized OOD with p-value calibration | Medium | Partial — Mahalanobis has calibration, rest need Q1 |
| Discovery orchestration | `NovelEntityMatcher` chains stages | Explicit pipeline orchestrator with named stages | High | **Done** — 5-stage `PipelineOrchestrator` |
| Clustering | Scalable clustering exists | Clustering as standard pipeline stage with interchangeable backends | Medium | **Done** — `ClusteringBackendRegistry`, but needs Leiden/Louvain |
| Evidence extraction | `ClusterEvidenceExtractor` (TF-IDF/centroid/combined) | Cluster summarization and keyword centroids | Medium | **Done** — see `novelty/extraction/evidence.py` |
| LLM proposals | LLM proposer with hierarchical summarization | Cluster-level judge/proposer with attribute discovery | Medium | **Done** — `propose_from_clusters_with_schema` |
| Schema enforcement | Pydantic models with retry | Retry-aware schema enforcement | Medium | **Done** — `_run_structured_cluster_proposal` with retry |
| HITL workflow | Review lifecycle with persistence | Active Learning Loop with instant index updates | High | Partial — review exists, active learning pending (E2) |
| Configuration | Config registries exist | Stage-oriented pipeline configuration | Medium | **Done** — `PipelineConfig` + `PipelineBuilder` |
| Documentation | Multiple roadmap narratives, some stale | One active technical roadmap | Completed | **This document** |
| OOD score normalization | Raw scores combined by `SignalCombiner` | Calibrated [0,1] scores per strategy | High | **Pending** — Q1 |
| Energy-based OOD | Not implemented | Energy score + ReAct strategy | High | **Pending** — B2 |
| Concept drift | Not implemented | Distribution snapshots + drift scorer + pipeline hook | High | **Pending** — B1 |
| Incremental entities | Full rebuild required | `add_entities` without retraining | High | **Pending** — D2 |
| Vector DB integration | In-memory ANN only | Pluggable VectorStore protocol + ChromaDB | Medium | **Pending** — D1 |
| Proposal conflict resolution | No overlap detection | Pairwise overlap detector + resolver | Medium | **Pending** — E1 |
| Cluster stability | Cohesion/separation only | Bootstrap Jaccard stability + filter stage | Medium | **Pending** — C2 |
| Graph community detection | Not implemented | Leiden / Louvain on k-NN similarity graph | Medium | **Pending** — C3 |

---

## Completed Milestones

Items from the previous roadmap (2026-03-24) that are now **done**:

| Previous Phase | Item | Completed As |
|---|---|---|
| Phase 2: Pipeline Contracts | Stage context, stage result, orchestrator contract | `pipeline/contracts.py`, `pipeline/orchestrator.py` |
| Phase 2: Pipeline Contracts | Adapters around existing capabilities | `pipeline/adapters.py` (5 stage adapters) |
| Phase 3: Community Detection | Clustering backends pluggable through stage contract | `ClusteringBackendRegistry`, `CommunityDetectionStage` |
| Phase 3: Evidence Extraction | Statistical keywords, representative examples, token budgets | `ClusterEvidenceExtractor` (tfidf/centroid/combined) |
| Phase 3: Schema Evolution | LLM proposer discovers common attributes/fields | `propose_from_clusters_with_schema` |
| Phase 3: Schema Enforcement | Retry-aware validation of LLM outputs | `_build_retry_prompt` with Pydantic schema |
| Phase 4: Review Persistence | Review-state persistence for proposed classes | `ProposalReviewManager` with JSON storage |
| Phase 4: Promotion Mechanics | Promotion APIs | `promote_proposal()` with default promoter |
| Phase 5: Public Pipeline API | Pipeline-first entry point | `DiscoveryPipeline` |
| Immediate: Unify Discovery | Shared orchestration core | `DiscoveryPipeline` + `NovelEntityMatcher` share `DiscoveryBase` |
| Immediate: Refactor Matcher | Decompose matcher.py | `MatcherRuntimeState`, `MatcherComponentFactory`, engine classes |
| Medium: Model Caching | Global model registry | `get_cached_sentence_transformer`, `get_cached_setfit_model` |
| Medium: Attribute Discovery | Enhanced LLM proposer | `propose_from_clusters_with_schema` + `DiscoveredAttribute` |

---

## Quick Wins

> High-impact, low-effort items shippable in 1–2 days each.

### Q1. OOD Score Calibration

**Why:** Strategy outputs live on different scales (confidence ∈ [0,1], KNN distance ∈ [0,2], Mahalanobis ∈ [0,∞)). `SignalCombiner` combines raw scores — producing unpredictable composite novelty scores.

**Deliverable:** `OODScoreCalibrator` — per-strategy min-max + quantile normalization.

```python
# novelty/core/score_calibrator.py

class OODScoreCalibrator:
    """Normalize OOD strategy outputs to a shared [0, 1] scale."""

    def __init__(self, method: str = "minmax"):
        self.method = method
        self._stats: dict[str, dict[str, float]] = {}

    def fit(self, strategy_scores: dict[str, np.ndarray]) -> OODScoreCalibrator:
        for strategy_id, scores in strategy_scores.items():
            self._stats[strategy_id] = {
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "p5": float(np.percentile(scores, 5)),
                "p95": float(np.percentile(scores, 95)),
            }
        return self

    def transform(self, strategy_id: str, scores: np.ndarray) -> np.ndarray:
        stats = self._stats.get(strategy_id)
        if stats is None:
            return scores
        if self.method == "minmax":
            lo, hi = stats["p5"], stats["p95"]
            if hi - lo < 1e-9:
                return np.zeros_like(scores)
            return np.clip((scores - lo) / (hi - lo), 0, 1)
        raise ValueError(f"Unknown method: {self.method}")
```

**Integration point:** `NoveltyDetector.detect_novel_samples` — calibrate each strategy's metric dict before passing to `SignalCombiner`.

**Test strategy:**
- Unit: verify `[0,1]` output for each method with random scores
- Unit: verify constant scores → all zeros
- Integration: run full pipeline with calibrator enabled, assert novelty scores ∈ [0,1]

---

### Q2. Embedding Cache with LRU Eviction

**Why:** Model-level caching only. Production workloads re-encode identical texts across calls.

**Deliverable:** `LRUEmbeddingCache` — hash-keyed embedding store with max-size eviction.

```python
# utils/embedding_cache.py

class LRUEmbeddingCache:
    """LRU cache for text embeddings with configurable max size."""

    def __init__(self, max_entries: int = 10_000, dim: int | None = None):
        self.max_entries = max_entries
        self.dim = dim
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, text: str) -> np.ndarray | None:
        if text in self._cache:
            self._cache.move_to_end(text)
            self._hits += 1
            return self._cache[text]
        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        if text in self._cache:
            self._cache.move_to_end(text)
        self._cache[text] = embedding
        if len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)

    def get_batch(
        self, texts: list[str]
    ) -> tuple[list[np.ndarray | None], list[int]]:
        """Return (cached_or_none_per_text, uncached_indices)."""
        results: list[np.ndarray | None] = []
        uncached: list[int] = []
        for i, text in enumerate(texts):
            emb = self.get(text)
            results.append(emb)
            if emb is None:
                uncached.append(i)
        return results, uncached
```

**Integration point:** Wrap `model.encode()` calls in `EmbeddingMatcher.build_index()` and `Matcher.match()`.

**Test strategy:**
- Unit: verify cache hit/miss counts
- Unit: verify LRU eviction at capacity
- Unit: verify batch API correctness

---

### Q3. Expose HDBSCAN Condensed Tree

**Why:** HDBSCAN computes a full hierarchy internally but `ScalableClusterer` only returns flat labels. Exposing the dendrogram enables multi-resolution analysis.

**Deliverable:** Add `get_condensed_tree()` and `extract_clusters_at_stability()` to `ScalableClusterer`.

```python
# In ScalableClusterer:

def get_condensed_tree(self) -> dict[str, Any]:
    """Return HDBSCAN condensed tree for visualization / multi-res selection."""
    if self._backend_instance is None:
        raise RuntimeError("Must call fit_predict first")
    return self._backend_instance.get_condensed_tree()

def extract_clusters_at_stability(
    self, min_persistence: float = 0.1
) -> tuple[np.ndarray, dict]:
    """Re-extract clusters at a different stability threshold."""
    ...
```

**Integration point:** `HDBSCANBackend` already has access to `clusterer.condensed_tree_`. Expose through the backend protocol.

**Test strategy:**
- Unit: verify tree structure contains expected keys
- Unit: verify re-extraction produces valid labels
- Visual: plot dendrogram in notebook

---

## B: Novelty & OOD Depth

### Epic B1: Temporal / Concept Drift Detection

**Problem:** As the entity landscape evolves, the trained matcher silently degrades. No signal tells the operator "your model is stale."

**Stories:**

#### B1.1 — Reference Distribution Snapshot

```python
# novelty/drift/distribution_snapshot.py

@dataclass
class DistributionSnapshot:
    """Statistical summary of the reference embedding distribution."""
    timestamp: datetime
    n_points: int
    mean: np.ndarray
    covariance: np.ndarray
    per_class_stats: dict[str, dict[str, np.ndarray]]
    embedding_hash: str

    @classmethod
    def from_embeddings(
        cls, embeddings: np.ndarray, labels: list[str]
    ) -> DistributionSnapshot:
        mean = embeddings.mean(axis=0)
        cov = np.cov(embeddings.T)
        per_class = {}
        for label in set(labels):
            mask = np.array(labels) == label
            class_embs = embeddings[mask]
            per_class[label] = {
                "mean": class_embs.mean(axis=0),
                "cov": np.cov(class_embs.T),
                "count": int(mask.sum()),
            }
        hasher = hashlib.sha256()
        hasher.update(embeddings.tobytes())
        return cls(
            timestamp=datetime.now(),
            n_points=len(embeddings),
            mean=mean,
            covariance=cov,
            per_class_stats=per_class,
            embedding_hash=hasher.hexdigest(),
        )

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            path,
            mean=self.mean,
            covariance=self.covariance,
            metadata=json.dumps({
                "timestamp": self.timestamp.isoformat(),
                "n_points": self.n_points,
                "embedding_hash": self.embedding_hash,
                "per_class_stats": {
                    k: {"mean": v["mean"].tolist(), "count": v["count"]}
                    for k, v in self.per_class_stats.items()
                },
            }),
        )

    @classmethod
    def load(cls, path: str | Path) -> DistributionSnapshot: ...
```

**Test strategy:**
- Unit: snapshot equality for identical inputs
- Unit: snapshot difference for perturbed inputs
- Integration: save/load round-trip preserves all fields

#### B1.2 — Drift Scorer

```python
# novelty/drift/drift_scorer.py

@dataclass
class DriftReport:
    global_drift_score: float
    per_class_drift: dict[str, float]
    drift_detected: bool
    method: str
    recommendation: str  # "retrain" | "add_entities" | "monitor"

class DriftScorer:
    """Compare current distribution against a baseline snapshot."""

    def __init__(self, method: str = "mmd_linear"):
        self.method = method

    def score(self, current: np.ndarray, baseline: DistributionSnapshot) -> DriftReport:
        # MMD with linear kernel for speed, KL-gaussian for accuracy
        # Per-class drift via cosine distance between centroids
        ...
```

**Algorithm options:**
- **MMD (Maximum Mean Discrepancy):** Linear kernel for speed, RBF for accuracy.
- **KL-divergence approximation:** Fit Gaussians, compute closed-form KL.
- **Per-class drift:** Cosine distance between centroids.

**Test strategy:**
- Unit: identical → score ≈ 0, shifted → > threshold
- Property: MMD is symmetric

#### B1.3 — Drift-Aware Pipeline Hook

```python
class DriftCheckStage(PipelineStage):
    name = "drift_check"

    def run(self, context: StageContext) -> StageResult:
        current = context.artifacts.get("query_embeddings")
        baseline = DistributionSnapshot.load(self.baseline_path)
        report = DriftScorer(method="mmd_linear").score(current, baseline)
        if report.drift_detected:
            logger.warning("Drift detected: %.3f — %s", report.global_drift_score, report.recommendation)
        return StageResult(artifacts={"drift_report": report}, metadata={...})
```

**Integration:** Optional pipeline stage, disabled by default. `PipelineConfig(drift_check_enabled=True)`.

**Test strategy:** Integration: full pipeline with drift stage, verify report in artifacts.

---

### Epic B2: Energy-Based OOD Detection

**Problem:** Distance/threshold heuristics are suboptimal. Energy-based scoring (Liu et al., NeurIPS 2020) is provably better aligned with input density.

#### B2.1 — Energy Score Strategy

```python
# novelty/core/strategies/energy.py

class EnergyOODStrategy(BaseStrategy):
    """Energy(x) = -T * log sum_i exp(logit_i(x) / T)
    Lower energy = more likely in-distribution."""

    def initialize(self, reference_embeddings, reference_labels, config):
        for label in set(reference_labels):
            mask = np.array(reference_labels) == label
            self._centroids[label] = reference_embeddings[mask].mean(axis=0)
        ref_logits = self._compute_logits(reference_embeddings)
        ref_energies = self._compute_energy(ref_logits)
        self._threshold = float(np.mean(ref_energies) - 2 * np.std(ref_energies))

    def _compute_logits(self, embeddings):
        centroid_matrix = np.array([self._centroids[l] for l in self._centroids])
        return cosine_similarity(embeddings, centroid_matrix) * self._scale

    def _compute_energy(self, logits):
        return -self._temperature * np.log(np.sum(np.exp(logits / self._temperature), axis=1))

    def detect(self, texts, embeddings, predicted_classes, confidences, **kwargs):
        logits = self._compute_logits(embeddings)
        energies = self._compute_energy(logits)
        flags = set(np.where(energies < self._threshold)[0])
        ...
```

**Key params:** `T=1.0` (temperature), `scale=25.0` (logit scaling).

**Test strategy:** Unit: in-distribution > OOD energy. Benchmark: AUROC vs KNN-distance.

#### B2.2 — Energy + ReAct Hybrid

```python
class ReActStrategy(BaseStrategy):
    """ReAct-style OOD: trim top activations then compute energy."""

    def _trim_features(self, embeddings):
        threshold = np.percentile(embeddings, self._trim_percentile * 100)
        trimmed = embeddings.copy()
        trimmed[trimmed > threshold] = threshold
        return trimmed

    def detect(self, texts, embeddings, predicted_classes, confidences, **kwargs):
        return self._energy_strategy.detect(
            texts, self._trim_features(embeddings), predicted_classes, confidences, **kwargs
        )
```

**Test strategy:** Benchmark: ReAct vs vanilla energy AUROC.

---

### Epic B3: Class-Conditional Fine-Grained OOD

**Problem:** Current Mahalanobis uses diagonal covariance only. Richer class models catch subtle near-OOD.

#### B3.1 — Mixture of Gaussians OOD

```python
# novelty/core/strategies/mixture_gaussian.py

class MixtureGaussianStrategy(BaseStrategy):
    """Fit per-class multivariate Gaussians, score via log-likelihood."""

    def initialize(self, reference_embeddings, reference_labels, config):
        dim = reference_embeddings.shape[1]
        for label in set(reference_labels):
            mask = np.array(reference_labels) == label
            class_embs = reference_embeddings[mask]
            self._class_models[label] = {
                "mean": class_embs.mean(axis=0),
                "cov": np.cov(class_embs.T) + np.eye(dim) * 1e-6,
                "prior": len(class_embs) / len(reference_labels),
            }

    def _log_likelihood(self, x, label):
        model = self._class_models[label]
        diff = x - model["mean"]
        cov_inv = np.linalg.inv(model["cov"])
        return -0.5 * (diff @ cov_inv @ diff) + np.log(model["prior"])
```

**Test strategy:** Unit: in-distribution > near-OOD. Unit: regularization prevents singular covariance. Benchmark: AUROC vs Mahalanobis.

---

## C: Clustering & Discovery

### Epic C1: Online / Incremental Clustering

**Problem:** Full refit required for every new batch. No incremental assignment.

#### C1.1 — Incremental Point Assignment

```python
# novelty/clustering/incremental.py

class IncrementalClusterer:
    """Assign new points to existing clusters; create new clusters when needed."""

    def assign(self, new_embeddings: np.ndarray) -> np.ndarray:
        # 1. Try assigning to existing centroids (cosine sim > threshold)
        # 2. Cluster unassigned among themselves via HDBSCAN
        # 3. Update centroids and state
        ...
```

**Test strategy:** Unit: existing → correct cluster. Unit: novel → new IDs. Unit: centroids update.

#### C1.2 — Cluster Merge Detection

```python
def detect_merges(centroids, merge_threshold=0.85) -> list[tuple[int, int]]:
    """Detect cluster pairs that should merge based on centroid similarity."""
    ...
```

**Test strategy:** Unit: identical → merge. Unit: distant → no merge.

---

### Epic C2: Cluster Stability Analysis

**Problem:** Single HDBSCAN run may produce unstable clusters. No robustness signal.

#### C2.1 — Bootstrap Stability Scorer

```python
# novelty/clustering/stability.py

class ClusterStabilityScorer:
    """Assess cluster stability via bootstrap resampling.
    Score > 0.7 = stable, < 0.3 = unstable."""

    def score(self, embeddings, base_labels, clusterer_factory) -> dict[int, float]:
        # n_bootstrap subsampled runs, per-cluster best Jaccard match averaged
        ...
```

**Test strategy:** Unit: separable → >0.9. Unit: noise → <0.3. Property: scores ∈ [0,1].

#### C2.2 — Stability-Gated Discovery Stage

```python
class StabilityFilterStage(PipelineStage):
    name = "stability_filter"
    # Insert between CommunityDetectionStage and ClusterEvidenceStage
    # Filter clusters with stability < 0.5
```

**Test strategy:** Integration: only stable clusters reach evidence stage.

---

### Epic C3: Graph-based Community Detection

**Problem:** HDBSCAN struggles with variable-density clusters, overlapping communities, non-globular shapes.

#### C3.1 — Similarity Graph Builder

```python
# novelty/clustering/graph.py

class SimilarityGraphBuilder:
    """Build a k-NN similarity graph from embeddings. Returns igraph.Graph."""

    def build(self, embeddings):
        knn = kneighbors_graph(embeddings, n_neighbors=self.k, metric=self.metric, mode="distance")
        knn.data = 1.0 - knn.data  # distance → similarity
        knn = knn.maximum(knn.T)    # symmetrize
        # Build igraph from sparse matrix
        ...
```

#### C3.2 — Leiden Community Detection Backend

```python
class LeidenBackend(ClusteringBackend):
    def fit_predict(self, X, min_cluster_size=5, metric="cosine", **kwargs):
        graph = SimilarityGraphBuilder(k=self.k, metric=metric).build(X)
        partition = graph.community_leiden(weights="weight", resolution_parameter=self.resolution)
        labels = np.array(partition.membership)
        # Filter small communities → -1
        ...
```

**Registration:** `ClusteringBackendRegistry.register("leiden", LeidenBackend)`

**Test strategy:** Unit: well-separated → correct labels. Benchmark: Leiden vs HDBSCAN on variable-density data.

#### C3.3 — Louvain Backend

Same interface, `graph.community_multilevel()`. Registered as `"louvain"`.

---

## D: Pipeline & Infrastructure

### Epic D1: Vector DB Integration

**Problem:** ANN index is in-memory only. No persistence, scaling, or hybrid metadata+vector search.

#### D1.1 — Vector Store Protocol

```python
# core/vector_store.py

class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: np.ndarray, metadata: list[dict] | None = None) -> None: ...
    def query(self, vector: np.ndarray, top_k: int = 10, filter: dict | None = None) -> list[dict]: ...
    def delete(self, ids: list[str]) -> None: ...
    def count(self) -> int: ...
```

#### D1.2 — ChromaDB Backend

```python
class ChromaVectorStore:
    def __init__(self, collection_name="entities", persist_directory=None): ...

    def upsert(self, ids, vectors, metadata=None):
        self._collection.upsert(ids=ids, embeddings=vectors.tolist(), metadatas=metadata)

    def query(self, vector, top_k=10, filter=None):
        results = self._collection.query(query_embeddings=..., n_results=top_k, where=filter)
        return [{"id": id, "score": score, "metadata": meta} for id, score, meta in ...]
```

**Test strategy:** Unit: upsert+query round-trip, metadata filter, delete. Integration: as EmbeddingMatcher backend.

#### D1.3 — In-Memory Backend (Default)

Wraps existing `ANNIndex` to satisfy `VectorStore` protocol. Zero external deps.

```python
class InMemoryVectorStore:
    def __init__(self, dim, backend="hnswlib", **kwargs):
        self._index = ANNIndex(dim=dim, backend=backend, **kwargs)
```

---

### Epic D2: Incremental Entity Addition

**Problem:** Adding entities requires full index rebuild + classifier retrain.

#### D2.1 — Incremental Embedding Index Update

```python
# In EmbeddingMatcher:
def add_entities(self, new_entities):
    new_embeddings = self.model.encode(new_texts)
    self.embeddings = np.vstack([self.embeddings, new_embeddings])
    if self._ann_index is not None:
        self._ann_index.add_vectors(new_embeddings, labels=new_ids)
```

**Test strategy:** Unit: new entities appear. Property: existing results unchanged.

#### D2.2 — Incremental Classifier Update

```python
# In SetFitClassifier:
def add_class(self, class_name, examples):
    embeddings = self.model.encode(examples)
    if self._use_sentence_transformer_fallback:
        self.class_centroids[class_name] = embeddings.mean(axis=0)
        self.labels.append(class_name)
    else:
        self._pending_new_class_examples[class_name] = examples  # queue for retrain
```

**Test strategy:** Unit: new class in labels, predict returns it, existing classes unaffected.

#### D2.3 — Pipeline-Level add_entities

```python
# In DiscoveryPipeline:
def add_entities(self, new_entities):
    self.matcher.add_entities(new_entities)
    self.entities.extend(new_entities)
    self.detector.reset()
```

**Test strategy:** Integration: add → match → verify. discover after add → verify new classes in reference.

---

## E: LLM & HITL

### Epic E1: Conflict Resolution Between Proposals

**Problem:** Proposals may overlap ("Technology Startup" vs "SaaS Company" for same clusters).

#### E1.1 — Proposal Overlap Detector + Resolver

```python
# novelty/proposal/conflict_resolver.py

@dataclass
class ProposalConflict:
    proposal_a: ClassProposal
    proposal_b: ClassProposal
    overlap_type: str       # "subset" | "superset" | "partial" | "duplicate"
    overlap_score: float    # 0-1
    shared_cluster_ids: list[int]
    recommendation: str     # "merge" | "keep_both" | "keep_a" | "keep_b"

class ProposalConflictResolver:
    def detect_conflicts(self, proposals) -> list[ProposalConflict]:
        # Pairwise: cluster overlap + name embedding similarity + example Jaccard
        ...

    def resolve(self, proposals) -> list[ClassProposal]:
        # Remove lower-confidence proposals from conflicts
        ...
```

**Test strategy:** Unit: identical→conflict, disjoint→no conflict, subset/superset classification. Integration: pipeline with overlapping clusters.

---

### Epic E2: Active Learning Loop

**Problem:** No mechanism to surface uncertain samples for annotation.

#### E2.1 — Uncertainty Sampler

```python
# novelty/active_learning/sampler.py

class UncertaintySampler:
    """Strategies: entropy, margin, least_confident."""

    def sample(self, texts, confidences, predicted_classes, n_samples=10) -> list[dict]:
        # Score by uncertainty, return top-n
        ...
```

#### E2.2 — Annotation Interface Adapter

```python
# novelty/active_learning/annotation.py

@dataclass
class AnnotationResult:
    text: str
    assigned_label: str  # "existing_class" or "novel:<new_class>"
    annotator: str

class AnnotationCollector:
    def apply_annotations(self, annotations, matcher):
        # Parse labels, add new entities + training data, partial retrain
        ...
```

**Test strategy:** Unit: entropy selects lowest-confidence. Integration: full loop: sample → annotate → apply → verify improvement.

---

## Deferred Features

> Valuable but postponed to reduce current scope complexity.

### Deferred 1: LLM Cost Tracking & Budget Enforcement

**Rationale:** Useful for production cost control but not blocking any other feature.

**Sketch:** `LLMCostTracker` wrapping `_call_litellm` — per-call token counting via `response.usage`, pricing table per model, budget cap raising `BudgetExceededError`.

### Deferred 2: PII Redaction Before LLM

**Rationale:** Important for compliance but current use cases are entity names/titles, not PII-heavy.

**Sketch:** `PIIRedactor` — regex patterns for email, phone, SSN, credit card. Apply in `sanitize_prompt_input()`.

### Deferred 3: Few-Shot Proposal Prompting via DSPy/GEPA

**Rationale:** Custom example store + handcrafted few-shot prompts are fragile. DSPy or GEPA can optimize proposal prompts programmatically.

**Sketch:**
- Replace handcrafted prompts with a DSPy `Module` (Signature → Predict → BootstrapFewShot)
- GEPA for genetic prompt evolution if DSPy optimization plateaus
- Training signal: approved vs rejected proposals from `ProposalReviewManager`
- Requires `dspy` and/or `gepa` as optional dependencies

### Deferred 4: Streaming / Real-time Pipeline & Event Hooks

**Rationale:** Significant architectural complexity (async queues, consumers, backpressure). Current sync/single-query async covers primary use case.

**Sketch:**
- `MicroBatchProcessor`: async queue → accumulate → `match_batch` → resolve futures
- `EventSource` protocol: `KafkaEventSource`, `RedisStreamSource`, `FileWatchSource`
- `EventConsumer` bridge: event source → micro-batch processor
- Backpressure: configurable queue size + drop policy

---

## Dependency Graph

```
Quick Wins (independent):
  Q1 (OOD Calibration) ← no deps
  Q2 (Embedding Cache) ← no deps
  Q3 (HDBSCAN Condensed Tree) ← no deps

B1 (Drift Detection):
  B1.1 (Distribution Snapshot) ← no deps
  B1.2 (Drift Scorer) ← B1.1
  B1.3 (Pipeline Hook) ← B1.2

B2 (Energy OOD):
  B2.1 (Energy Score) ← Q1
  B2.2 (ReAct Hybrid) ← B2.1, B3.1

B3 (Class-Conditional):
  B3.1 (MoG Strategy) ← Q1

C1 (Incremental Clustering):
  C1.1 (Point Assignment) ← no deps
  C1.2 (Merge Detection) ← C1.1

C2 (Stability Analysis):
  C2.1 (Bootstrap Scorer) ← no deps
  C2.2 (Stability Filter Stage) ← C2.1

C3 (Graph Community Detection):
  C3.1 (Similarity Graph) ← no deps
  C3.2 (Leiden Backend) ← C3.1
  C3.3 (Louvain Backend) ← C3.1

D1 (Vector DB):
  D1.1 (Vector Store Protocol) ← no deps
  D1.2 (ChromaDB Backend) ← D1.1
  D1.3 (In-Memory Backend) ← D1.1

D2 (Incremental Entities):
  D2.1 (Embedding Update) ← D1.1
  D2.2 (Classifier Update) ← D2.1
  D2.3 (Pipeline-Level) ← D2.1, D2.2

E1 (Conflict Resolution):
  E1.1 (Overlap Detector) ← no deps

E2 (Active Learning):
  E2.1 (Uncertainty Sampler) ← no deps
  E2.2 (Annotation Collector) ← E2.1, D2.3
```

---

## Suggested Sequencing

### Phase 1 — Foundation (Week 1–2)

| Item | Days | Reason |
|---|---|---|
| Q1: OOD Score Calibration | 1 | Unblocks all OOD improvements |
| Q2: Embedding Cache | 2 | Performance prerequisite |
| Q3: HDBSCAN Condensed Tree | 1 | Standalone value |
| D1.1: Vector Store Protocol | 1 | Unlocks D1.2, D2.1 |
| D1.3: In-Memory Backend | 1 | Wraps existing ANN |

### Phase 2 — OOD Depth (Week 3–4)

| Item | Days | Reason |
|---|---|---|
| B2.1: Energy OOD | 3 | High-value standalone strategy |
| B3.1: MoG Strategy | 3 | Better class-conditional modeling |
| B1.1: Distribution Snapshot | 2 | Foundation for drift |
| B2.2: ReAct Hybrid | 2 | Combines B2.1 + B3.1 |
| B1.2: Drift Scorer | 2 | Depends on B1.1 |
| B1.3: Drift Pipeline Hook | 1 | Depends on B1.2 |

### Phase 3 — Clustering & Discovery (Week 5–6)

| Item | Days | Reason |
|---|---|---|
| C3.1: Similarity Graph Builder | 2 | Foundation for C3.2/C3.3 |
| C3.2: Leiden Backend | 2 | Primary graph community method |
| C2.1: Bootstrap Stability | 3 | Quality assurance for clusters |
| C1.1: Incremental Assignment | 3 | Production clustering |
| C2.2: Stability Filter Stage | 1 | Depends on C2.1 |
| C1.2: Merge Detection | 1 | Depends on C1.1 |
| C3.3: Louvain Backend | 1 | Alternative to Leiden |

### Phase 4 — Infrastructure (Week 7–8)

| Item | Days | Reason |
|---|---|---|
| D1.2: ChromaDB Backend | 3 | Persistent vector storage |
| D2.1: Incremental Embeddings | 2 | Depends on D1.1 |
| D2.2: Incremental Classifier | 3 | Depends on D2.1 |
| D2.3: Pipeline-Level add_entities | 1 | Depends on D2.1, D2.2 |

### Phase 5 — LLM & HITL (Week 9–10)

| Item | Days | Reason |
|---|---|---|
| E1.1: Proposal Conflict Resolver | 3 | Standalone |
| E2.1: Uncertainty Sampler | 2 | Standalone |
| E2.2: Annotation Collector | 3 | Depends on E2.1, D2.3 |

### Phase 6 — Deferred (Future)

| Item | Reason |
|---|---|
| LLM Cost Tracking | Production cost control |
| PII Redaction | Compliance for personal-data domains |
| DSPy/GEPA Proposal Optimization | When LLM proposal quality bottlenecks |
| Streaming / Event-Driven Pipeline | When real-time consumption required |

---

## Risks and Mitigations

| Risk | Why It Matters | Mitigation |
|---|---|---|
| Doc drift returns | Repo already had roadmap statements mismatching implementation | Treat roadmap updates as part of feature work |
| Premature API redesign | Public pipeline API too early could freeze weak contracts | Internal stage contracts proven first, then exposed |
| Discovery complexity balloons | Novelty, clustering, LLM, review flows sprawl | Explicit stage boundaries, verify each independently |
| LLM cost/latency creep | Proposal generation expensive at sample level | Cluster-level evidence extraction + bounded prompting |
| Promotion workflow mistakes | Poorly governed promotion corrupts known entity index | Explicit approval state + auditable promotion |
| Experimental method sprawl | Too many novelty strategies blur supported path | Label experimental vs supported, benchmark before promoting |
| OOD score miscalibration | New strategies add uncalibrated signals | Q1 calibrator ships before any new strategy |

---

## Out of Scope

- **A1–A5 (Matching & Classification):** Calibration, cascading, ensembles, rules layer, adaptive thresholds — not prioritized this cycle.
- **B3.2 (KDE OOD):** Marginal improvement over MoG at high dimensionality cost.
- **F1 (Multi-lingual):** Requires embedding model change — cross-cutting concern.
- **F2 (XAI):** Model-specific interpretability (SHAP, attention visualization).
- **F3 (Domain Adaptation):** Unsupervised fine-tuning pipeline (TSDAE, GPL).
- **F4 (Entity Resolution):** Different problem class (dedup vs classification).
- **F5 (Data Augmentation):** Low priority given few-shot classifier exists.

---

## Definition of Done

This roadmap remains the active source of truth until replaced. Update when:
- The primary public API changes
- The stage architecture materially changes
- Promotion/review workflows become more concrete
- Package extras, supported backends, or test strategy change

When a future roadmap replaces this one, it should supersede explicitly and archive this file.

> Archived on 2026-03-24. Superseded by [`docs/technical-roadmap.md`](../../technical-roadmap.md) as the active technical roadmap.

# Technical Roadmap: Migration to Modular Discovery Pipeline

**Repository:** github.com/minghao51/novel_entity_matcher  
**Author:** minghao51  
**Version:** v0.3.x → v1.0.0 Target  
**Domain:** Healthcare / Insurance · Singapore  
**Status:** Beta (Breaking Changes Allowed)  
**Last Updated:** 2026-03-24  

---

## Table of Contents

1. [Executive Summary](#01-executive-summary)
2. [Current State Analysis](#02-current-state-analysis)
3. [Gap Analysis: Current vs Target](#03-gap-analysis-current-vs-target)
4. [Migration Strategy](#04-migration-strategy)
5. [Target Pipeline Architecture](#05-target-pipeline-architecture)
6. [Implementation Specifications](#06-implementation-specifications)
7. [Target Module Layout](#07-target-module-layout)
8. [Migration Phases & Timeline](#08-migration-phases--timeline)
9. [Breaking Changes Log](#09-breaking-changes-log)
10. [Open Design Decisions](#10-open-design-decisions)
11. [Testing Strategy](#11-testing-strategy)
12. [Appendix: Code Examples](#12-appendix-code-examples)

---

## 01. Executive Summary

This document outlines the technical roadmap for migrating `novel_entity_matcher` from its current architecture to a modular, stage-based Discovery Pipeline. The migration aims to transform the codebase into a swappable, experiment-friendly system that follows an **ML-first, LLM-last** philosophy, where LLMs are invoked only once per cluster rather than once per entity, dramatically reducing API costs while maintaining high-quality novelty detection.

> **⚠️ Beta Status:** This project is in beta. Breaking changes are **expected and allowed**. No backward compatibility guarantees apply. The goal is to ship the correct architecture, not preserve legacy APIs.

### Key Objectives

1. **Modularize** the existing novelty detection system into discrete, swappable pipeline stages
2. **Enhance** OOD detection with statistical methods (Mahalanobis, LOF) beyond simple threshold cutoffs
3. **Introduce** community detection as a cost-optimization layer before LLM invocation
4. **Add** statistical keyword extraction (KeyBERT/c-TF-IDF) to compress context for LLM calls
5. **Implement** HITL (Human-in-the-Loop) staging and promotion workflow for taxonomy evolution
6. **Redesign** the public API around the pipeline as the primary interface

### Core Philosophy

> **The Discovery Gap:** The current system routes unknown entities through a cosine threshold and marks them as low-confidence, but does nothing systematic with them. There is no clustering of rejected entities, no automated naming, and no mechanism to promote a novel concept back into the known index. This migration closes that gap.

---

## 02. Current State Analysis

### 2.1 Existing Architecture Overview

The current codebase has the following high-level structure:

```
src/novelentitymatcher/
├── core/                          # Matching logic
│   ├── matcher.py                 # Unified Matcher class
│   ├── embedding_matcher.py       # Zero-shot embedding matching
│   ├── entity_matcher.py          # SetFit/BERT training
│   ├── classifier.py              # SetFit classifier wrapper
│   ├── bert_classifier.py         # BERT classifier
│   ├── hybrid.py                  # Hybrid retrieval + reranking
│   ├── blocking.py                # Blocking strategies
│   └── ...
│
├── novelty/                       # Novelty detection (already modular!)
│   ├── core/
│   │   ├── detector.py            # NoveltyDetector with strategy pattern
│   │   └── strategies.py          # Strategy registry
│   ├── strategies/                # Pluggable detection strategies
│   │   ├── base.py                # NoveltyStrategy ABC
│   │   ├── confidence.py          # Confidence threshold
│   │   ├── knn_distance.py        # KNN distance-based
│   │   ├── clustering.py          # HDBSCAN-based clustering
│   │   ├── setfit.py              # SetFit uncertainty
│   │   ├── prototypical.py        # Prototypical networks
│   │   ├── oneclass.py            # One-class SVM
│   │   ├── pattern.py             # Pattern-based detection
│   │   └── self_knowledge.py      # Self-knowledge detection
│   ├── clustering/
│   │   ├── scalable.py            # ScalableClusterer (HDBSCAN, sOPTICS)
│   │   └── validation.py          # Cluster validation
│   ├── proposal/
│   │   ├── llm.py                 # LLMClassProposer via LiteLLM
│   │   └── retrieval.py           # RetrievalAugmentedProposer
│   ├── schemas/                   # Pydantic models
│   ├── storage/                   # Persistence layer
│   └── entity_matcher.py          # NovelEntityMatcher orchestration
│
├── backends/                      # Embedding/Reranker backends
│   ├── base.py                    # EmbeddingBackend, RerankerBackend ABCs
│   ├── static_embedding.py        # Static embeddings (potion-8m)
│   ├── sentencetransformer.py     # SentenceTransformer
│   ├── litellm.py                 # LiteLLM backend
│   └── reranker_st.py             # Cross-encoder reranker
│
├── config.py                      # Configuration loader
├── config_registry.py             # Model/mode registry
└── ...
```

### 2.2 Existing Strengths

The current architecture already has several key components aligned with the target state:

| Component | Current Implementation | Target Alignment |
|-----------|----------------------|------------------|
| **Strategy Pattern** | `NoveltyStrategy` ABC with `StrategyRegistry` | ✅ Ready for pipeline stages |
| **Clustering** | `ScalableClusterer` with HDBSCAN/sOPTICS | ✅ Partially ready (needs Leiden) |
| **LLM Integration** | `LLMClassProposer` via LiteLLM | ✅ Ready, needs schema enforcement |
| **Pydantic Models** | `ClassProposal`, `NovelClassAnalysis` | ✅ Ready for expansion |
| **ANN Index** | `ANNIndex` with HNSW/FAISS | ✅ Ready |
| **Persistence** | `save_proposals`, `load_proposals` | ✅ Ready for staging table |
| **Async API** | `fit_async`, `match_async` | ✅ Ready |

### 2.3 Key Classes Analysis

#### `NoveltyDetector` (novelty/core/detector.py)

The existing detector already implements:
- Strategy initialization and lifecycle management
- Signal combination via `SignalCombiner`
- Metadata building via `MetadataBuilder`
- Hash-based reference signature for re-initialization detection

**Gap:** This is oriented toward novelty detection, not a full discovery pipeline with LLM proposal and HITL integration.

#### `NovelEntityMatcher` (novelty/entity_matcher.py)

This class orchestrates:
1. `Matcher` for classification
2. `NoveltyDetector` for novelty scoring
3. `LLMClassProposer` for class naming

**Gap:** Not structured as a stage-based pipeline. Missing OOD filter separation, statistical extraction, and HITL workflow.

#### `ScalableClusterer` (novelty/clustering/scalable.py)

Supports:
- HDBSCAN (default for <100K points)
- sOPTICS (accelerated for 100K-1M points)
- UMAP+HDBSCAN (for >1M points)

**Gap:** Missing Leiden graph-based clustering for hierarchical domains.

---

## 03. Gap Analysis: Current vs Target

### 3.1 Stage-by-Stage Comparison

| Stage | Target Component | Current Status | Gap Severity |
|-------|-----------------|----------------|--------------|
| **Stage 1: Vectorization** | Swappable embedder interface | `EmbeddingBackend` ABC exists | 🟢 Minor - needs pipeline integration |
| **Stage 2: OOD Filter** | Mahalanobis/LOF OOD detection | Only `knn_distance` strategy | 🟡 Medium - needs Mahalanobis implementation |
| **Stage 3: Community Detection** | HDBSCAN/Leiden clustering | `ScalableClusterer` has HDBSCAN | 🟡 Medium - needs Leiden addition |
| **Stage 4: Statistical Extractor** | KeyBERT/c-TF-IDF | Not implemented | 🔴 High - entirely new |
| **Stage 5: LLM Judge** | Cluster-level LLM invocation | `LLMClassProposer` exists but sample-level | 🟡 Medium - needs batching + schema |
| **Stage 6: Schema Enforcer** | Pydantic validation + retry | Basic Pydantic, no retry logic | 🟡 Medium - needs retry mechanism |
| **HITL Workflow** | Staging table + promotion | Persistence exists, no promotion | 🔴 High - entirely new |

### 3.2 Detailed Gap Analysis

#### Stage 1: Vectorization Interface

**Current:**
- `EmbeddingBackend` ABC with `encode()` method
- Implementations: `StaticEmbeddingBackend`, `SentenceTransformerBackend`, `LiteLLMBackend`

**Missing:**
- Pipeline stage wrapper that exposes `BaseEmbedder` interface
- Configuration injection via `config.yaml`

**Recommendation:** Create `SetFitEmbedder` that wraps existing backends with SetFit-specific logic for void exploitation.

#### Stage 2: OOD Filter

**Current:**
- `knn_distance` strategy computes KNN distances
- `confidence` strategy checks prediction confidence
- No per-class covariance estimation

**Missing:**
- Mahalanobis distance: `Dₘ(x) = √[(x−μ)ᵀ Σ⁻¹(x−μ)]` per known class
- Local Outlier Factor (LOF) as fallback
- `ood_threshold` and `ood_method` configuration

**Recommendation:** Implement `MahalanobisOODFilter` as a new strategy, with `LOFFilter` as alternative.

#### Stage 3: Community Detection

**Current:**
- `ScalableClusterer` with HDBSCAN/sOPTICS/UMAP+HDBSCAN
- Auto-backend selection based on dataset size
- Quality metrics: cohesion, separation, silhouette

**Missing:**
- Leiden graph-based clustering for hierarchical structures
- KNN graph construction for Leiden

**Recommendation:** Add `LeidenClusterer` using `leidenalg` or `igraph`. Make it available via configuration.

#### Stage 4: Statistical Extractor

**Current:**
- No keyword extraction before LLM

**Missing:**
- KeyBERT for top-K n-gram extraction per cluster
- c-TF-IDF for cluster-specific term importance
- Token budget management for LLM calls

**Recommendation:** Implement `KeyBERTExtractor` and `CTFIDFExtractor` as optional Stage 4 components.

#### Stage 5: LLM Judge

**Current:**
- `LLMClassProposer` takes `List[NovelSampleMetadata]`
- Samples limited to 20 in prompt
- JSON response parsing with Pydantic validation

**Missing:**
- Cluster-level batching (1,000 entities → 5 LLM calls)
- Three-field structured output: verify, name, explain
- Cascaded model routing (Haiku → Sonnet escalation)

**Recommendation:** Refactor `LLMClassProposer` to accept cluster-level batches with micro-batch prompts.

#### Stage 6: Schema Enforcer

**Current:**
- Pydantic models for response validation
- No retry-on-validation-failure

**Missing:**
- Automatic retry with corrected prompt
- `max_retries` configuration
- `PydanticOutputParser` wrapper

**Recommendation:** Implement `PydanticOutputParser` with retry logic.

#### HITL Workflow

**Current:**
- `save_proposals()` for persistence
- `export_summary()` for markdown export

**Missing:**
- Staging table with weighted confidence
- Auto-promotion at >95% confidence
- FastAPI endpoints for review
- Index promotion + SetFit retrain trigger

**Recommendation:** Create `hitl/` module with `StagingTable`, `ProposalAPI`, and `IndexPromoter`.

---

## 04. Migration Strategy

### 4.1 Core Principles

1. **Clean Architecture Over Compatibility:** Ship the correct design, not a compatibility layer. Breaking changes are acceptable and expected.
2. **Incremental Refactoring:** Migrate one stage at a time, with each stage independently testable.
3. **Strategy Reuse:** Leverage existing `NoveltyStrategy` pattern for pipeline stages.
4. **Configuration-Driven:** All stage selection via `config.yaml`, not code changes.
5. **Delete Dead Code:** Remove unused code paths rather than deprecating them.

### 4.2 Migration Approach

```
Phase 1: Abstract Pipeline Interfaces (Weeks 1-2)
    └── Create pipeline/ module with ABCs
    └── Wire stage chain into DiscoveryPipeline orchestrator
    └── Refactor Matcher to use pipeline internally
    
Phase 2: OOD Filter Implementation (Weeks 3-4)
    └── MahalanobisOODFilter
    └── LocalOutlierFactorFilter
    └── Benchmark vs cosine threshold
    └── Remove old threshold-based NIL detection
    
Phase 3: Community Detection + Extractor (Weeks 5-7)
    └── LeidenClusterer
    └── KeyBERTExtractor
    └── c-TF-IDF Extractor
    └── Remove sample-level LLM call path
    
Phase 4: LLM Judge + Schema Enforcer (Weeks 8-10)
    └── Cluster-level batching
    └── PydanticOutputParser with retry
    └── Delete old LLMClassProposer
    
Phase 5: HITL Dashboard + Promotion (Weeks 11-14)
    └── StagingTable
    └── FastAPI endpoints
    └── IndexPromoter with retrain trigger
```

### 4.3 New API Design

The `DiscoveryPipeline` becomes the primary interface:

```python
# NEW API (v1.0.0)
from novelentitymatcher import DiscoveryPipeline

pipeline = DiscoveryPipeline.from_config("config.yaml")
pipeline.fit(entities=entities, training_data=training_data)
report = pipeline.discover(["query1", "query2", ...])

# Match results still available
results = pipeline.match(["query"])  # Returns MatchResult

# Low-level stage access for experiments
embeddings = pipeline.stages["embed"].encode(["text"])
ood_scores = pipeline.stages["ood"].score(embeddings)
```

---

## 05. Target Pipeline Architecture

### 5.1 Stage Interfaces

All stages implement a common interface:

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass
import numpy as np

@dataclass
class StageContext:
    """Shared context passed between stages."""
    texts: List[str]
    embeddings: np.ndarray
    reference_embeddings: np.ndarray
    reference_labels: List[str]
    metadata: Dict[str, Any]

@dataclass  
class StageResult:
    """Result from a single stage."""
    data: Any
    metrics: Dict[str, float]
    passed_indices: np.ndarray  # Indices passed to next stage
    flagged_indices: np.ndarray  # Indices flagged for special handling

class PipelineStage(ABC):
    """Base class for all pipeline stages."""
    
    stage_id: str
    stage_name: str
    
    @abstractmethod
    def initialize(self, context: StageContext) -> None:
        """Initialize stage with reference data."""
        pass
    
    @abstractmethod
    def process(self, context: StageContext) -> StageResult:
        """Process inputs and return results."""
        pass
    
    @abstractmethod
    def get_config_schema(self) -> type:
        """Return Pydantic config schema for this stage."""
        pass
```

### 5.2 Pipeline Orchestrator

```python
class DiscoveryPipeline:
    """
    Primary API for entity matching and novel class discovery.
    
    Stage execution flow:
    1. Vectorization → embeddings
    2. OOD Filter → NIL pool
    3. Community Detection → clusters
    4. Statistical Extractor → keywords
    5. LLM Judge → proposals
    6. Schema Enforcer → validated proposals
    """
    
    def __init__(
        self,
        stages: List[PipelineStage],
        config: Optional[Dict[str, Any]] = None,
    ):
        self.stages = stages
        self.config = config or {}
        self._stage_map = {s.stage_id: s for s in stages}
        
    @classmethod
    def from_config(cls, path: str) -> "DiscoveryPipeline":
        """Load pipeline from YAML configuration."""
        ...
    
    def fit(
        self,
        entities: List[Dict[str, Any]],
        training_data: Optional[List[Dict]] = None,
    ) -> "DiscoveryPipeline":
        """Fit the pipeline with known entities and optional training data."""
        ...
        
    def match(
        self,
        texts: List[str],
        top_k: int = 1,
    ) -> List["MatchResult"]:
        """Match texts against known entities."""
        ...
        
    def discover(
        self,
        texts: List[str],
        run_llm: bool = True,
        context: Optional[str] = None,
    ) -> "DiscoveryReport":
        """Execute full discovery pipeline on texts."""
        # Build initial context
        ctx = self._build_context(texts)
        
        # Execute stages sequentially
        results = {}
        for stage in self.stages:
            stage_result = stage.process(ctx)
            results[stage.stage_id] = stage_result
            ctx = self._update_context(ctx, stage_result)
        
        return self._build_report(results)
    
    @property
    def stages(self) -> Dict[str, PipelineStage]:
        """Access individual stages for experimentation."""
        return self._stage_map
```

### 5.3 Configuration Schema

```yaml
# config.yaml
pipeline:
  stages:
    - id: embed
      implementation: SetFitEmbedder
      config:
        model: potion-8m
        normalize: true
        
    - id: ood
      implementation: MahalanobisOODFilter
      config:
        threshold: 0.95
        fallback: LocalOutlierFactorFilter
        min_samples_per_class: 5
        
    - id: cluster
      implementation: HDBSCANClusterer
      config:
        min_cluster_size: 5
        min_samples: 3
        metric: cosine
        
    - id: extract
      implementation: KeyBERTExtractor
      config:
        top_n: 5
        ngram_range: [1, 2]
        optional: true  # Skip if LLM budget concerns
        
    - id: judge
      implementation: LLMJudgeAgent
      config:
        model: openai/gpt-4o-mini
        fallback_model: anthropic/claude-3-haiku
        max_retries: 2
        batch_size: 10  # Clusters per LLM call
        
    - id: schema
      implementation: PydanticOutputParser
      config:
        strict: true
        auto_reject_invalid: false

hitl:
  enabled: true
  staging_table: ./data/proposals/staging.json
  auto_promote_threshold: 0.95
  retrain_on_promote: true
```

---

## 06. Implementation Specifications

### 6.1 Stage 1: Vectorization Interface

#### SetFitEmbedder

**File:** `src/novelentitymatcher/pipeline/stage1_embed/setfit_embedder.py`

**Purpose:** Convert raw string mentions into dense vectors. SetFit-trained vectors push known classes apart, creating voids for OOD detection.

**Interface:**

```python
class SetFitEmbedder(PipelineStage):
    """Vectorization using SetFit-compatible embeddings."""
    
    stage_id = "embed"
    stage_name = "Vectorization"
    
    def __init__(
        self,
        model: str = "potion-8m",
        normalize: bool = True,
        backend: Optional[EmbeddingBackend] = None,
    ):
        self.model_name = model
        self.normalize = normalize
        self._backend = backend
        self._model = None
    
    def initialize(self, context: StageContext) -> None:
        """Load model if not provided."""
        if self._backend is None:
            from ...backends import get_backend_for_model
            self._backend = get_backend_for_model(self.model_name)
    
    def process(self, context: StageContext) -> StageResult:
        """Encode texts to embeddings."""
        embeddings = self._backend.encode(context.texts)
        
        if self.normalize:
            embeddings = embeddings / np.linalg.norm(
                embeddings, axis=1, keepdims=True
            )
        
        return StageResult(
            data=embeddings,
            metrics={
                "n_texts": len(context.texts),
                "embedding_dim": embeddings.shape[1],
            },
            passed_indices=np.arange(len(context.texts)),
            flagged_indices=np.array([]),
        )
```

**Swappable Options:**
- `SentenceTransformerEmbedder` - Standard SentenceTransformers
- `OpenAIEmbedder` - Via LiteLLM
- `VoyageEmbedder` - Via LiteLLM
- `BgeM3Embedder` - Multilingual support

### 6.2 Stage 2: OOD Filter

#### MahalanobisOODFilter

**File:** `src/novelentitymatcher/pipeline/stage2_ood/mahalanobis.py`

**Purpose:** Statistical out-of-distribution detection using Mahalanobis distance per known class.

**Mathematical Foundation:**

$$D_M(x) = \sqrt{(x-\mu_c)^T \Sigma_c^{-1} (x-\mu_c)}$$

Where:
- $x$ is the query embedding
- $\mu_c$ is the mean embedding of class $c$
- $\Sigma_c$ is the covariance matrix for class $c$

**Interface:**

```python
class MahalanobisOODFilter(PipelineStage):
    """OOD detection using per-class Mahalanobis distance."""
    
    stage_id = "ood"
    stage_name = "OOD Filter"
    
    def __init__(
        self,
        threshold: float = 0.95,  # Percentile threshold
        min_samples_per_class: int = 5,
        fallback_to_lof: bool = True,
        regularization: float = 1e-6,
    ):
        self.threshold = threshold
        self.min_samples = min_samples_per_class
        self.fallback_to_lof = fallback_to_lof
        self.regularization = regularization
        
        # Per-class statistics
        self._class_means: Dict[str, np.ndarray] = {}
        self._class_covs: Dict[str, np.ndarray] = {}
        self._class_thresholds: Dict[str, float] = {}
        self._lof_fallback: Optional[LocalOutlierFactor] = None
    
    def initialize(self, context: StageContext) -> None:
        """Fit per-class covariance matrices."""
        labels = np.array(context.reference_labels)
        embeddings = context.reference_embeddings
        
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            class_embeddings = embeddings[mask]
            
            if len(class_embeddings) < self.min_samples:
                # Mark for LOF fallback
                continue
            
            # Compute statistics
            self._class_means[label] = np.mean(class_embeddings, axis=0)
            cov = np.cov(class_embeddings.T) + self.regularization * np.eye(
                class_embeddings.shape[1]
            )
            self._class_covs[label] = cov
            
            # Compute in-sample distances for threshold calibration
            inv_cov = np.linalg.inv(cov)
            diffs = class_embeddings - self._class_means[label]
            distances = np.sqrt(np.sum(diffs @ inv_cov * diffs, axis=1))
            self._class_thresholds[label] = np.percentile(
                distances, self.threshold * 100
            )
        
        # Train LOF for classes with insufficient samples
        if self.fallback_to_lof:
            valid_mask = np.array([
                l in self._class_means for l in labels
            ])
            if valid_mask.sum() < len(labels):
                self._lof_fallback = LocalOutlierFactor(
                    novelty=True, n_neighbors=20
                )
                self._lof_fallback.fit(embeddings[~valid_mask])
    
    def _compute_distance(
        self,
        embedding: np.ndarray,
        predicted_class: str,
    ) -> float:
        """Compute Mahalanobis distance to predicted class."""
        if predicted_class not in self._class_means:
            return float('inf')
        
        mu = self._class_means[predicted_class]
        cov = self._class_covs[predicted_class]
        
        diff = embedding - mu
        inv_cov = np.linalg.inv(cov)
        
        return float(np.sqrt(diff @ inv_cov @ diff))
    
    def process(self, context: StageContext) -> StageResult:
        """Flag embeddings outside class confidence ellipsoids."""
        embeddings = context.embeddings
        predicted_classes = context.metadata.get("predicted_classes", [])
        
        distances = []
        flags = []
        
        for i, (emb, pred_class) in enumerate(
            zip(embeddings, predicted_classes)
        ):
            dist = self._compute_distance(emb, pred_class)
            distances.append(dist)
            
            if pred_class in self._class_thresholds:
                is_ood = dist > self._class_thresholds[pred_class]
            elif self._lof_fallback is not None:
                # Use LOF for fallback
                lof_score = self._lof_fallback.score_samples([emb])[0]
                is_ood = lof_score < -1.5
            else:
                is_ood = True  # Unknown class = OOD
            
            flags.append(is_ood)
        
        flagged_indices = np.where(flags)[0]
        passed_indices = flagged_indices  # OOD samples go to clustering
        
        return StageResult(
            data={
                "distances": np.array(distances),
                "is_ood": np.array(flags),
            },
            metrics={
                "n_ood": len(flagged_indices),
                "ood_rate": len(flagged_indices) / len(embeddings),
                "mean_distance": np.mean(distances),
            },
            passed_indices=passed_indices,
            flagged_indices=flagged_indices,
        )
```

**Risk Mitigation:**
- Mahalanobis requires sufficient per-class samples (≥5)
- Add `LocalOutlierFactorFilter` as automatic fallback for sparse classes
- Regularize covariance matrices to avoid singularity

### 6.3 Stage 3: Community Detection

#### LeidenClusterer

**File:** `src/novelentitymatcher/pipeline/stage3_cluster/leiden_clusterer.py`

**Purpose:** Graph-based clustering for hierarchical domains using the Leiden algorithm.

**Algorithm:**
1. Build KNN graph from embeddings
2. Apply Leiden community detection
3. Return cluster assignments with quality metrics

**Interface:**

```python
class LeidenClusterer(PipelineStage):
    """Graph-based clustering using Leiden algorithm."""
    
    stage_id = "cluster"
    stage_name = "Community Detection"
    
    def __init__(
        self,
        k_neighbors: int = 15,
        resolution: float = 1.0,
        n_iterations: int = 2,
        metric: str = "cosine",
    ):
        self.k_neighbors = k_neighbors
        self.resolution = resolution
        self.n_iterations = n_iterations
        self.metric = metric
        
        self._graph: Optional[igraph.Graph] = None
        self._labels: Optional[np.ndarray] = None
    
    def _build_knn_graph(
        self,
        embeddings: np.ndarray,
    ) -> igraph.Graph:
        """Build KNN graph from embeddings."""
        from sklearn.neighbors import NearestNeighbors
        
        n_samples = len(embeddings)
        k = min(self.k_neighbors, n_samples - 1)
        
        # Find KNN
        nn = NearestNeighbors(n_neighbors=k + 1, metric=self.metric)
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
        
        # Build edge list
        edges = []
        weights = []
        for i in range(n_samples):
            for j, (neighbor, dist) in enumerate(zip(indices[i, 1:], distances[i, 1:])):
                if i < neighbor:  # Avoid duplicate edges
                    edges.append((i, neighbor))
                    # Convert distance to similarity weight
                    weights.append(1.0 / (1.0 + dist))
        
        # Create igraph
        graph = igraph.Graph(n=n_samples, edges=edges)
        graph.es["weight"] = weights
        
        return graph
    
    def process(self, context: StageContext) -> StageResult:
        """Cluster OOD embeddings using Leiden."""
        # Get OOD embeddings from previous stage
        ood_indices = context.metadata.get("ood_indices", [])
        if len(ood_indices) == 0:
            return StageResult(
                data={"labels": np.array([]), "n_clusters": 0},
                metrics={"n_clusters": 0, "n_noise": 0},
                passed_indices=np.array([]),
                flagged_indices=np.array([]),
            )
        
        ood_embeddings = context.embeddings[ood_indices]
        
        # Build graph
        self._graph = self._build_knn_graph(ood_embeddings)
        
        # Apply Leiden
        import leidenalg
        partition = leidenalg.find_partition(
            self._graph,
            leidenalg.RBConfigurationVertexPartition,
            resolution_parameter=self.resolution,
            n_iterations=self.n_iterations,
        )
        
        self._labels = np.array(partition.membership)
        
        # Compute metrics
        unique_labels = np.unique(self._labels)
        n_clusters = len(unique_labels)
        n_noise = np.sum(self._labels == -1) if -1 in unique_labels else 0
        
        # Map back to original indices
        cluster_assignments = np.full(len(context.texts), -1)
        cluster_assignments[ood_indices] = self._labels
        
        return StageResult(
            data={
                "labels": cluster_assignments,
                "ood_indices": ood_indices,
                "n_clusters": n_clusters,
            },
            metrics={
                "n_clusters": n_clusters,
                "n_noise": n_noise,
                "avg_cluster_size": len(ood_indices) / max(n_clusters, 1),
            },
            passed_indices=ood_indices,
            flagged_indices=np.array([]),
        )
```

**When to use Leiden vs HDBSCAN:**
- **Leiden:** Hierarchical domains (e.g., "Cardiovascular" → "Interventional Cardiology")
- **HDBSCAN:** Flat, density-varying clusters; simpler setup

### 6.4 Stage 4: Statistical Extractor

#### KeyBERTExtractor

**File:** `src/novelentitymatcher/pipeline/stage4_extract/keybert_extractor.py`

**Purpose:** Extract representative n-grams from each cluster to compress context for LLM calls.

**Interface:**

```python
class KeyBERTExtractor(PipelineStage):
    """Keyword extraction using KeyBERT for cluster summarization."""
    
    stage_id = "extract"
    stage_name = "Statistical Extractor"
    
    def __init__(
        self,
        top_n: int = 5,
        ngram_range: Tuple[int, int] = (1, 2),
        diversity: float = 0.5,
        use_mmr: bool = True,
    ):
        self.top_n = top_n
        self.ngram_range = ngram_range
        self.diversity = diversity
        self.use_mmr = use_mmr
        
        self._model = None
    
    def initialize(self, context: StageContext) -> None:
        """Load KeyBERT model."""
        try:
            from keybert import KeyBERT
            self._model = KeyBERT(model="all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "keybert is required. Install with: pip install keybert"
            )
    
    def process(self, context: StageContext) -> StageResult:
        """Extract keywords per cluster."""
        cluster_labels = context.metadata.get("cluster_labels", [])
        texts = np.array(context.texts)
        
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        
        cluster_keywords = {}
        
        for cluster_id in unique_clusters:
            # Get cluster texts
            mask = cluster_labels == cluster_id
            cluster_texts = texts[mask]
            
            if len(cluster_texts) == 0:
                continue
            
            # Combine texts for keyword extraction
            combined = " ".join(cluster_texts)
            
            # Extract keywords
            if self.use_mmr:
                keywords = self._model.extract_keywords(
                    combined,
                    keyphrase_ngram_range=self.ngram_range,
                    use_mmr=True,
                    diversity=self.diversity,
                    top_n=self.top_n,
                )
            else:
                keywords = self._model.extract_keywords(
                    combined,
                    keyphrase_ngram_range=self.ngram_range,
                    top_n=self.top_n,
                )
            
            cluster_keywords[int(cluster_id)] = [
                {"keyword": kw, "score": float(score)}
                for kw, score in keywords
            ]
        
        return StageResult(
            data={"cluster_keywords": cluster_keywords},
            metrics={
                "n_clusters_processed": len(cluster_keywords),
                "avg_keywords_per_cluster": np.mean([
                    len(kws) for kws in cluster_keywords.values()
                ]) if cluster_keywords else 0,
            },
            passed_indices=context.metadata.get("ood_indices", []),
            flagged_indices=np.array([]),
        )
```

### 6.5 Stage 5: LLM Judge

#### LLMJudgeAgent

**File:** `src/novelentitymatcher/pipeline/stage5_judge/llm_judge.py`

**Purpose:** Verify clusters, propose names, and generate explanations with minimal LLM calls.

**Key Design:**
- **Cluster-level batching:** One LLM call per cluster, not per entity
- **Three-field output:** `is_valid_cluster`, `class_name`, `explanation`
- **Cascaded routing:** Haiku for initial check, Sonnet for dense medical clusters

**Interface:**

```python
class LLMJudgeAgent(PipelineStage):
    """LLM-based cluster verification and naming."""
    
    stage_id = "judge"
    stage_name = "LLM Judge"
    
    # Model routing config
    MODEL_ROUTING = {
        "fast": "anthropic/claude-3-haiku",
        "balanced": "openai/gpt-4o-mini",
        "medical": "anthropic/claude-3.5-sonnet",
    }
    
    def __init__(
        self,
        model: str = "balanced",
        fallback_models: Optional[List[str]] = None,
        max_retries: int = 2,
        sample_size: int = 10,  # Samples per cluster in prompt
        token_budget: int = 4000,
    ):
        self.model = self.MODEL_ROUTING.get(model, model)
        self.fallback_models = fallback_models or [
            "openai/gpt-4o-mini",
            "anthropic/claude-3-haiku",
        ]
        self.max_retries = max_retries
        self.sample_size = sample_size
        self.token_budget = token_budget
    
    def _build_cluster_prompt(
        self,
        cluster_id: int,
        samples: List[str],
        keywords: List[Dict],
        nearest_known: List[str],
    ) -> str:
        """Build micro-batch prompt for a single cluster."""
        samples_text = "\n".join(f"- {s}" for s in samples[:self.sample_size])
        keywords_text = ", ".join(k["keyword"] for k in keywords[:5])
        nearest_text = ", ".join(nearest_known[:3])
        
        return f"""Analyze this cluster of novel entities:

CLUSTER ID: {cluster_id}
SAMPLE ENTITIES:
{samples_text}

STATISTICAL KEYWORDS: {keywords_text}
NEAREST KNOWN CLASSES: {nearest_text}

Your task:
1. Verify if this is a coherent cluster (not noise)
2. Propose a concise class name (2-4 words)
3. Explain how it differs from nearest known classes

Respond as JSON:
{{
  "is_valid_cluster": true/false,
  "class_name": "proposed name",
  "confidence": 0.0-1.0,
  "explanation": "2-sentence rationale"
}}"""
    
    def process(self, context: StageContext) -> StageResult:
        """Judge each cluster with minimal LLM calls."""
        cluster_labels = context.metadata.get("cluster_labels", [])
        texts = np.array(context.texts)
        keywords = context.metadata.get("cluster_keywords", {})
        
        unique_clusters = np.unique(cluster_labels)
        unique_clusters = unique_clusters[unique_clusters >= 0]
        
        proposals = []
        total_llm_calls = 0
        
        for cluster_id in unique_clusters:
            # Get cluster data
            mask = cluster_labels == cluster_id
            cluster_texts = texts[mask].tolist()
            cluster_keywords = keywords.get(int(cluster_id), [])
            
            # Get nearest known classes
            nearest_known = self._get_nearest_known_classes(
                context.embeddings[mask],
                context.reference_embeddings,
                context.reference_labels,
            )
            
            # Build prompt
            prompt = self._build_cluster_prompt(
                cluster_id, cluster_texts, cluster_keywords, nearest_known
            )
            
            # Call LLM with fallback
            response, model_used = self._call_llm(prompt)
            total_llm_calls += 1
            
            # Parse response
            proposal = self._parse_response(response, cluster_id, cluster_texts)
            proposal["model_used"] = model_used
            proposals.append(proposal)
        
        return StageResult(
            data={"proposals": proposals},
            metrics={
                "n_clusters_judged": len(proposals),
                "total_llm_calls": total_llm_calls,
                "valid_clusters": sum(1 for p in proposals if p.get("is_valid_cluster")),
            },
            passed_indices=np.array([]),
            flagged_indices=np.array([]),
        )
```

### 6.6 Stage 6: Schema Enforcer

#### PydanticOutputParser

**File:** `src/novelentitymatcher/pipeline/stage6_schema/parser.py`

**Purpose:** Validate LLM output against Pydantic schema with automatic retry.

**Schema Definition:**

```python
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional

class NovelClassProposal(BaseModel):
    """Canonical schema for a novel class proposal."""
    
    class_name: str = Field(
        ...,
        min_length=2,
        max_length=100,
        description="Proposed class name (2-4 words preferred)",
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=500,
        description="Clear description of what this class represents",
    )
    rationale: str = Field(
        ...,
        min_length=20,
        max_length=1000,
        description="Why this class differs from existing ones",
    )
    nearest_known: List[str] = Field(
        default_factory=list,
        description="List of nearest known classes",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model confidence in this proposal",
    )
    is_valid_cluster: bool = Field(
        default=True,
        description="Whether this cluster is valid (not noise)",
    )
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific attributes (healthcare codes, etc.)",
    )
    cluster_id: Optional[int] = Field(
        default=None,
        description="Source cluster ID",
    )
    sample_count: int = Field(
        default=0,
        ge=0,
        description="Number of samples in this cluster",
    )
```

**Parser with Retry:**

```python
class PydanticOutputParser(PipelineStage):
    """Validate and parse LLM output with retry logic."""
    
    stage_id = "schema"
    stage_name = "Schema Enforcer"
    
    def __init__(
        self,
        schema: type[BaseModel] = NovelClassProposal,
        max_retries: int = 2,
        strict: bool = True,
        auto_reject_invalid: bool = False,
    ):
        self.schema = schema
        self.max_retries = max_retries
        self.strict = strict
        self.auto_reject_invalid = auto_reject_invalid
    
    def parse_with_retry(
        self,
        response: str,
        llm_call_fn: Callable[[str], str],
    ) -> Tuple[Optional[BaseModel], List[str]]:
        """Parse response with automatic retry on validation failure."""
        errors = []
        
        for attempt in range(self.max_retries + 1):
            try:
                json_str = self._extract_json(response)
                data = json.loads(json_str)
                proposal = self.schema(**data)
                return proposal, errors
                
            except ValidationError as e:
                errors.append(f"Attempt {attempt + 1}: {str(e)}")
                
                if attempt < self.max_retries:
                    retry_prompt = self._build_retry_prompt(response, str(e))
                    response = llm_call_fn(retry_prompt)
                elif self.auto_reject_invalid:
                    return None, errors
                else:
                    return self._create_partial(response, errors), errors
        
        return None, errors
```

---

## 07. Target Module Layout

```
src/novelentitymatcher/
├── __init__.py                    # Exports DiscoveryPipeline as primary API
│
├── pipeline/                      # NEW: Primary module
│   ├── __init__.py
│   ├── base.py                    # PipelineStage ABC, StageContext, StageResult
│   ├── orchestrator.py            # DiscoveryPipeline class (PRIMARY API)
│   ├── config.py                  # PipelineConfig Pydantic model
│   │
│   ├── stage1_embed/              # Vectorization
│   │   ├── __init__.py
│   │   ├── setfit_embedder.py     # SetFitEmbedder implementation
│   │   └── litellm_embedder.py    # LiteLLM embedder wrapper
│   │
│   ├── stage2_ood/                # OOD / NIL Filter
│   │   ├── __init__.py
│   │   ├── base_ood.py            # BaseOODFilter ABC
│   │   ├── mahalanobis.py         # MahalanobisOODFilter
│   │   └── lof.py                 # LocalOutlierFactorFilter
│   │
│   ├── stage3_cluster/            # Community Detection
│   │   ├── __init__.py
│   │   ├── base_cluster.py        # BaseClusterer ABC
│   │   ├── hdbscan_clusterer.py   # Wrapper for ScalableClusterer
│   │   └── leiden_clusterer.py    # LeidenClusterer
│   │
│   ├── stage4_extract/            # Statistical Extractor (optional)
│   │   ├── __init__.py
│   │   ├── base_extractor.py      # BaseExtractor ABC
│   │   ├── keybert_extractor.py   # KeyBERTExtractor
│   │   └── ctfidf_extractor.py    # CTFIDFExtractor
│   │
│   ├── stage5_judge/              # LLM Judge & Explainer
│   │   ├── __init__.py
│   │   ├── llm_judge.py           # LLMJudgeAgent
│   │   └── model_router.py        # Cascaded model routing logic
│   │
│   └── stage6_schema/             # Pydantic Output Parser
│       ├── __init__.py
│       ├── models.py              # NovelClassProposal schema
│       └── parser.py              # PydanticOutputParser with retry
│
├── hitl/                          # NEW: promotion & HITL API
│   ├── __init__.py
│   ├── staging.py                 # StagingTable class
│   ├── api.py                     # FastAPI endpoints (optional)
│   ├── promoter.py                # IndexPromoter + retrain trigger
│   └── normalizer.py              # Singlish/code-switch normalizer
│
├── core/                          # RETAINED: Core matching logic
│   ├── embedding_matcher.py       # Zero-shot matching
│   ├── entity_matcher.py          # SetFit/BERT training
│   ├── classifier.py              # SetFit classifier wrapper
│   └── ...
│
├── backends/                      # RETAINED: Provider integrations
│   └── ...
│
├── novelty/                       # DEPRECATED → MIGRATE TO pipeline/
│   └── ...                        # Keep during transition, delete after
│
└── data/
    └── pipeline_config.yaml       # Default pipeline configuration
```

### Key Module Changes

| Change Type | Module | Action |
|-------------|--------|--------|
| **NEW** | `pipeline/` | Primary module, all stages |
| **NEW** | `hitl/` | Human-in-the-loop workflow |
| **RETAIN** | `core/` | Matching logic (no API changes) |
| **RETAIN** | `backends/` | Provider integrations |
| **DEPRECATE** | `novelty/` | Migrate strategies to pipeline stages, delete after migration |
| **DELETE** | `NovelEntityMatcher` | Replace with `DiscoveryPipeline` |

---

## 08. Migration Phases & Timeline

### Phase 1: Abstract Pipeline Interfaces (Weeks 1-2)

**Goal:** Create the ABC layer for swappable stages.

**Tasks:**
- [ ] Create `pipeline/` module with `base.py` ABCs
- [ ] Define `PipelineStage`, `StageContext`, `StageResult`
- [ ] Define `BaseEmbedder`, `BaseOODFilter`, `BaseClusterer`, `BaseExtractor`, `BaseLLMJudge`, `BaseSchemaEnforcer`
- [ ] Create `PipelineConfig` Pydantic model
- [ ] Implement `DiscoveryPipeline` orchestrator
- [ ] Update `config.yaml` schema for stage injection
- [ ] Add unit tests for orchestrator

**Deliverables:**
- `pipeline/base.py` with all ABCs
- `pipeline/orchestrator.py` with `DiscoveryPipeline`
- `tests/test_pipeline/test_orchestrator.py`

---

### Phase 2: OOD Filter Implementation (Weeks 3-4)

**Goal:** Replace threshold-based NIL detection with statistical OOD methods.

**Tasks:**
- [ ] Implement `MahalanobisOODFilter` with per-class covariance
- [ ] Implement `LocalOutlierFactorFilter` as fallback
- [ ] Add `ood_threshold` and `ood_method` to config
- [ ] Create `stage2_ood/` module structure
- [ ] Benchmark OOD recall vs cosine threshold on existing fixtures
- [ ] **DELETE** old threshold-based NIL detection code
- [ ] Document minimum samples per class requirement

**Deliverables:**
- `pipeline/stage2_ood/mahalanobis.py`
- `pipeline/stage2_ood/lof.py`
- `tests/test_pipeline/test_ood.py`
- Benchmark results in `docs/benchmarks/ood_comparison.md`

---

### Phase 3: Community Detection + Extractor (Weeks 5-7)

**Goal:** Enable the "1,000 entities → 5 LLM calls" cost structure.

**Tasks:**
- [ ] Refactor `ScalableClusterer` to implement `BaseClusterer`
- [ ] Implement `LeidenClusterer` using `leidenalg` + `igraph`
- [ ] Add cluster metrics: silhouette, density, size histogram
- [ ] Implement `KeyBERTExtractor` with MMR diversity
- [ ] Implement `CTFIDFExtractor` as alternative
- [ ] Generate synthetic adversarial data for cold-start testing
- [ ] **DELETE** sample-level LLM call path

**Deliverables:**
- `pipeline/stage3_cluster/hdbscan_clusterer.py`
- `pipeline/stage3_cluster/leiden_clusterer.py`
- `pipeline/stage4_extract/keybert_extractor.py`
- `pipeline/stage4_extract/ctfidf_extractor.py`
- Synthetic test data in `data/synthetic/`

---

### Phase 4: LLM Judge + Schema Enforcer (Weeks 8-10)

**Goal:** Wire in the LLM verification/naming layer with proper output enforcement.

**Tasks:**
- [ ] Implement `LLMJudgeAgent` with cluster-level batching
- [ ] Add cascaded model routing (Haiku → Sonnet)
- [ ] Implement `PydanticOutputParser` with retry-on-validation-failure
- [ ] Write approved proposals to staging table
- [ ] Add confidence threshold for auto-promotion
- [ ] **DELETE** `LLMClassProposer` class
- [ ] **DELETE** `NovelEntityMatcher` class

**Deliverables:**
- `pipeline/stage5_judge/llm_judge.py`
- `pipeline/stage5_judge/model_router.py`
- `pipeline/stage6_schema/parser.py`
- `pipeline/stage6_schema/models.py` (NovelClassProposal)

---

### Phase 5: HITL Dashboard + Promotion (Weeks 11-14)

**Goal:** Enable human review and index promotion.

**Tasks:**
- [ ] Implement `StagingTable` with weighted confidence storage
- [ ] Create FastAPI endpoints: `GET /proposals`, `POST /proposals/{id}/approve`
- [ ] Implement `IndexPromoter` for index updates
- [ ] Add head-only SetFit retrain trigger on promotion
- [ ] Implement Singlish/code-switch normalizer
- [ ] Track taxonomy drift across runs
- [ ] **DELETE** unused novelty/ module code

**Deliverables:**
- `hitl/staging.py`
- `hitl/api.py`
- `hitl/promoter.py`
- `hitl/normalizer.py`

---

## 09. Breaking Changes Log

This section documents all planned breaking changes. Since this is a beta project, breaking changes are acceptable and will be implemented directly without deprecation periods.

### v0.4.0 Breaking Changes

| Change | Old API | New API | Migration |
|--------|---------|---------|-----------|
| Primary interface | `Matcher` | `DiscoveryPipeline` | Use `DiscoveryPipeline.from_config()` |
| Novelty detection | `NoveltyDetector` | Pipeline stages | Configure stages in YAML |
| NIL detection | Cosine threshold | Mahalanobis/LOF | Update config |

### v0.5.0 Breaking Changes

| Change | Old API | New API | Migration |
|--------|---------|---------|-----------|
| LLM proposals | `LLMClassProposer` | `LLMJudgeAgent` stage | Use cluster-level batching |
| Entity matcher | `NovelEntityMatcher` | `DiscoveryPipeline` | Use `pipeline.discover()` |
| Sample-level LLM | Per-entity calls | Per-cluster calls | No code change, automatic |

### v0.6.0 Breaking Changes

| Change | Old API | New API | Migration |
|--------|---------|---------|-----------|
| Output schema | Loose JSON | Strict Pydantic | Update response handling |
| Retry logic | Manual | Automatic | Remove retry code |

### v1.0.0 Breaking Changes

| Change | Old API | New API | Migration |
|--------|---------|---------|-----------|
| Module structure | `novelty/` | `pipeline/` | Update imports |
| Config format | Flat | Stage-based YAML | Reconfigure pipeline |
| HITL workflow | Manual file review | StagingTable + API | Use HITL module |

### Deleted Modules/Classes

The following will be **permanently deleted** (not deprecated):

| Module/Class | Deletion Phase | Replacement |
|--------------|----------------|-------------|
| `NovelEntityMatcher` | Phase 4 | `DiscoveryPipeline` |
| `LLMClassProposer` | Phase 4 | `LLMJudgeAgent` stage |
| `novelty/` module | Phase 5 | `pipeline/` stages |
| Threshold-based NIL | Phase 2 | OOD filter stages |
| Sample-level LLM calls | Phase 3 | Cluster-level batching |

---

## 10. Open Design Decisions

### Decision A: Auto-promotion Threshold

**Question:** What combined confidence threshold should trigger automatic index promotion?

**Options:**
1. **Conservative:** LLM confidence ≥ 0.95 AND cluster density ≥ 0.80
2. **Balanced:** LLM confidence ≥ 0.90 AND cluster density ≥ 0.70
3. **Aggressive:** LLM confidence ≥ 0.85 AND cluster density ≥ 0.60

**Recommendation:** Start with **Conservative**. Tune thresholds down as the system matures. Track precision of auto-promoted classes.

**Implementation:**
```python
# hitl/promoter.py
AUTO_PROMOTE_CONFIG = {
    "llm_confidence_threshold": 0.95,
    "cluster_density_threshold": 0.80,
    "min_cluster_size": 5,
}
```

### Decision B: HDBSCAN vs Leiden Default

**Question:** Which clustering algorithm should be the default?

**Analysis:**
| Factor | HDBSCAN | Leiden |
|--------|---------|--------|
| Setup complexity | Simple | Requires KNN graph |
| Hierarchical support | No | Yes |
| Noise handling | Explicit (-1 label) | Implicit |
| Healthcare domain fit | Flat taxonomies | Hierarchical (ICD-10 style) |

**Recommendation:** **HDBSCAN as default**. Include Leiden in registry for hierarchical domains. Document when to switch.

### Decision C: Cold-start Synthetic Data

**Question:** How to generate test data for benchmarking without real data?

**Options:**
1. **LLM-generated variants:** Prompt LLM to generate typos, acronyms, Singlish
2. **Rule-based perturbation:** Character swaps, phonetic variations
3. **Public healthcare datasets:** Use open medical taxonomies (ICD-10, SNOMED)

**Recommendation:** **Hybrid approach**:
1. Use ICD-10/SNOMED for base entities
2. Generate variants via LLM with controlled prompts
3. Store in `data/synthetic/` with seed for reproducibility

```python
# Example LLM prompt for synthetic generation
SYNTHETIC_PROMPT = """Generate 10 Singapore healthcare variations of "{entity_name}":
- Include typos
- Include acronyms (HBP, DM, HTN)
- Include Singlish/code-switch variations
- Include formal and informal forms

Format as JSON list."""
```

### Decision D: HITL Integration Method

**Question:** How should users interact with the HITL system?

**Options:**
1. **FastAPI REST API:** Full web service
2. **CLI tool:** Command-line review interface
3. **Jupyter notebook:** Interactive review widget
4. **File-based:** Review proposals in JSON/YAML files

**Recommendation:** **File-based as default**, FastAPI as optional `[hitl]` extra.

```bash
# File-based workflow
$ novelentitymatcher review ./proposals/staging.json
# Opens editor, user approves/rejects
# On approval: index updated, retrain triggered
```

---

## 11. Testing Strategy

### 11.1 Unit Tests

Each stage requires isolated unit tests:

```python
# tests/test_pipeline/test_ood.py
import pytest
import numpy as np
from novelentitymatcher.pipeline.stage2_ood import MahalanobisOODFilter, StageContext

class TestMahalanobisOODFilter:
    
    @pytest.fixture
    def filter(self):
        return MahalanobisOODFilter(threshold=0.95)
    
    @pytest.fixture
    def context(self):
        # Create synthetic reference data
        np.random.seed(42)
        reference_embeddings = np.vstack([
            np.random.randn(50, 128) + np.array([1] * 128),  # Class A
            np.random.randn(50, 128) + np.array([-1] * 128),  # Class B
        ])
        reference_labels = ["A"] * 50 + ["B"] * 50
        
        # Test embeddings: in-distribution + out-of-distribution
        test_embeddings = np.vstack([
            np.random.randn(10, 128) + np.array([1] * 128),  # Near class A
            np.random.randn(5, 128) * 3,  # OOD
        ])
        
        return StageContext(
            texts=[f"text_{i}" for i in range(15)],
            embeddings=test_embeddings,
            reference_embeddings=reference_embeddings,
            reference_labels=reference_labels,
            metadata={"predicted_classes": ["A"] * 10 + ["unknown"] * 5},
        )
    
    def test_initialization(self, filter, context):
        filter.initialize(context)
        assert len(filter._class_means) == 2
        assert len(filter._class_thresholds) == 2
    
    def test_detection(self, filter, context):
        filter.initialize(context)
        result = filter.process(context)
        
        # Should flag OOD samples
        assert result.metrics["n_ood"] >= 3
        assert len(result.passed_indices) >= 3
```

### 11.2 Integration Tests

Full pipeline integration tests:

```python
# tests/test_pipeline/test_integration.py
import pytest
from novelentitymatcher import DiscoveryPipeline

class TestDiscoveryPipeline:
    
    @pytest.fixture
    def entities(self):
        return [
            {"id": "cardio", "name": "Cardiology"},
            {"id": "neuro", "name": "Neurology"},
            {"id": "ortho", "name": "Orthopedics"},
        ]
    
    @pytest.fixture
    def pipeline(self):
        return DiscoveryPipeline.from_config("config.yaml")
    
    def test_end_to_end(self, entities, pipeline):
        pipeline.fit(entities=entities)
        
        queries = [
            "heart checkup",  # Near cardio
            "brain surgery",  # Near neuro
            "knee replacement",  # Near ortho
            "genetic therapy trial",  # OOD - novel
            "remote patient monitoring",  # OOD - novel
        ]
        
        report = pipeline.discover(queries)
        
        assert report.n_novel_samples >= 2
        assert len(report.proposals) >= 1
```

### 11.3 Benchmark Suite

Automated benchmarks for key metrics:

```yaml
# benchmarks/pipeline_benchmarks.yaml
benchmarks:
  - name: ood_detection_recall
    metric: recall@threshold
    baseline: 0.75  # Cosine threshold
    target: 0.90  # Mahalanobis
    
  - name: llm_call_efficiency
    metric: calls_per_1000_entities
    baseline: 1000  # One per entity
    target: 10  # Cluster-level
    
  - name: end_to_end_latency
    metric: seconds_per_100_entities
    baseline: 5.0
    target: 3.0
```

---

## 12. Appendix: Code Examples

### A. Basic Pipeline Usage

```python
from novelentitymatcher import DiscoveryPipeline

# Initialize from config
pipeline = DiscoveryPipeline.from_config("config.yaml")

# Fit with known entities
entities = [
    {"id": "cardio", "name": "Cardiology", "aliases": ["heart", "cardiac"]},
    {"id": "neuro", "name": "Neurology", "aliases": ["brain", "neural"]},
]
pipeline.fit(entities=entities)

# Run discovery
report = pipeline.discover(
    texts=["heart failure", "genetic testing", "brain tumor"],
    context="Healthcare provider taxonomy",
)

# Review results
for proposal in report.proposals:
    print(f"Proposed: {proposal.class_name}")
    print(f"  Confidence: {proposal.confidence:.2f}")
    print(f"  Rationale: {proposal.rationale}")
```

### B. Custom Pipeline Configuration

```python
from novelentitymatcher.pipeline import (
    DiscoveryPipeline,
    SetFitEmbedder,
    MahalanobisOODFilter,
    LeidenClusterer,
    KeyBERTExtractor,
    LLMJudgeAgent,
    PydanticOutputParser,
)

# Build custom pipeline
pipeline = DiscoveryPipeline(
    stages=[
        SetFitEmbedder(model="bge-m3", normalize=True),
        MahalanobisOODFilter(threshold=0.90, fallback_to_lof=True),
        LeidenClusterer(k_neighbors=20, resolution=1.5),
        KeyBERTExtractor(top_n=5, use_mmr=True),
        LLMJudgeAgent(model="medical", sample_size=15),
        PydanticOutputParser(strict=True),
    ],
    config={
        "hitl": {
            "auto_promote_threshold": 0.95,
            "staging_path": "./proposals/staging.json",
        }
    }
)

# Run pipeline
pipeline.fit(entities=entities)
report = pipeline.discover(texts=["query1", "query2", ...])
```

### C. Stage-Level Experimentation

```python
# Access individual stages for experiments
pipeline = DiscoveryPipeline.from_config("config.yaml")

# Get embeddings directly
embeddings = pipeline.stages["embed"].encode(["text"])

# Score OOD without full pipeline
ood_scores = pipeline.stages["ood"].score(embeddings)

# Cluster manually
labels = pipeline.stages["cluster"].fit_predict(embeddings)

# Extract keywords for a cluster
keywords = pipeline.stages["extract"].extract(
    ["sample1", "sample2", "sample3"]
)
```

### D. HITL Workflow

```python
from novelentitymatcher.hitl import StagingTable, IndexPromoter

# Load staging table
staging = StagingTable.load("./proposals/staging.json")

# Review proposals
for proposal in staging.pending:
    print(f"ID: {proposal.id}")
    print(f"Class: {proposal.class_name}")
    print(f"Confidence: {proposal.confidence}")
    
    # Approve or reject
    decision = input("Approve? (y/n/skip): ")
    if decision == "y":
        staging.approve(proposal.id)
    elif decision == "n":
        staging.reject(proposal.id)

# Promote approved classes to index
promoter = IndexPromoter(pipeline=pipeline)
promoter.promote(staging.approved)

# Head-only retrain (fast)
pipeline.fit(entities=promoter.updated_entities, mode="head-only")
```

### E. Singlish Normalization

```python
from novelentitymatcher.hitl import SinglishNormalizer

normalizer = SinglishNormalizer()

# Normalize code-switched input
normalized = normalizer.normalize("HBP patient needs follow-up")
# → "Hypertension patient needs follow-up"

normalized = normalizer.normalize("Gahmen hospital very far")
# → "Government hospital very far"

# Batch normalization
normalized_batch = normalizer.normalize_batch([
    "DM controlled",
    "TBT positive",
    "Poly referral needed",
])
# → ["Diabetes Mellitus controlled", "Tuberculosis positive", "Polyclinic referral needed"]
```

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-03-24 | minghao51 | Initial roadmap creation |
| 1.1 | 2026-03-24 | minghao51 | Updated for beta status - breaking changes allowed |

---

**Document Maintainers:** minghao51  
**Review Cycle:** Per milestone completion  
**Feedback:** Open an issue on GitHub for suggestions or questions

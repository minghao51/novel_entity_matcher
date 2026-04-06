# Clustering Strategy (HDBSCAN)

**Phase 2: Novelty Detection** | **Density-based clustering** | **Weight: 0.20 in ensemble**

---

## Mathematical Formulation

### HDBSCAN Clustering

HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) builds a hierarchy of clusters and extracts stable clusters:

```
Core distance:
  core_k(x) = distance(x, k-th nearest neighbor)

Mutual reachability distance:
  d_mreach(a, b) = max(core_k(a), core_k(b), d(a, b))

Minimum Spanning Tree (MST):
  Build MST using d_mreach as edge weights

Hierarchy:
  Extract clusters at all density levels by cutting MST at varying thresholds

Cluster stability:
  persistence(C) = Σ_{x ∈ C} (λ_max(x) - λ_min(C))
  
  where λ = 1/density (inverse density)
  λ_max(x) = density level at which x leaves the cluster
  λ_min(C) = density level at which cluster C forms
```

### Novelty Scoring

```
For each query point x:

If labeled as noise (cluster = -1):
  is_novel = True
  novelty_score = 1.0

If assigned to cluster C:
  cohesion = avg_pairwise_distance(members of C)
  support_score = 1 - cohesion
  
  is_novel = support_score < (1 - cohesion_threshold)
             OR cluster not validated (persistence < persistence_threshold)
  
  novelty_score = 1 - support_score
```

### Cluster Validation

```
A cluster C is valid if:
  persistence(C) ≥ persistence_threshold
  AND support(C) contains sufficient known-class samples
  AND cohesion ≤ cohesion_threshold

Known support:
  support_ratio = |C ∩ known_classes| / |C|
  
  Low support_ratio → cluster may represent novel class
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Reference embeddings: numpy array (n_ref × dim)           │
│  • Reference labels: List[str]                               │
│  • Query embeddings: numpy array (n_queries × dim)           │
│  • min_cluster_size: int (default: 5)                        │
│  • cohesion_threshold: float (default: 0.45)                 │
│  • persistence_threshold: float (default: 0.1)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: COMBINED EMBEDDING MATRIX              │
├─────────────────────────────────────────────────────────────┤
│  Technique: Concatenation                                    │
│  • X_combined = concat(reference_embeddings, query_embeddings)│
│  • Track origin of each point (reference vs query)           │
│  • Output: Combined embedding matrix + origin labels         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: HDBSCAN CLUSTERING                     │
├─────────────────────────────────────────────────────────────┤
│  Technique: HDBSCAN (hierarchical density-based clustering)  │
│  • Compute core distances for all points                     │
│  • Build mutual reachability distance matrix                 │
│  • Construct MST with mutual reachability distances          │
│  • Extract hierarchy of clusters                             │
│  • Select stable clusters (maximum persistence)              │
│  • Label noise points (cluster = -1)                         │
│  • Output: Cluster labels for all points                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: CLUSTER VALIDATION                     │
├─────────────────────────────────────────────────────────────┤
│  Technique: Cluster Quality Metrics                          │
│  • For each cluster C:                                       │
│    - Compute persistence (stability across density levels)   │
│    - Compute cohesion (avg pairwise distance within cluster) │
│    - Compute known support (fraction of known-class samples) │
│    - Validate: persistence ≥ threshold AND cohesion ≤ thresh │
│  • Output: {cluster_id: is_valid} mapping                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: NOVELTY SCORING                        │
├─────────────────────────────────────────────────────────────┤
│  Technique: Cluster-Based Novelty                            │
│  • For each query point:                                     │
│    If cluster = -1 (noise):                                  │
│      is_novel = True, novelty_score = 1.0                    │
│    Else if cluster not valid:                                │
│      is_novel = True, novelty_score = 1 - cohesion           │
│    Else:                                                     │
│      support_score = 1 - cohesion                            │
│      is_novel = support_score < (1 - cohesion_threshold)     │
│      novelty_score = 1 - support_score                       │
│  • Output: (novelty_score, is_novel, cluster_id, metrics)    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • cluster_id: int (-1 for noise)                            │
│  • persistence: float (cluster stability)                    │
│  • cohesion: float (within-cluster similarity)               │
│  • known_support: float (fraction of known-class samples)    │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Combine reference and query embeddings**: Cluster all points together
2. **Run HDBSCAN**: Find density-based clusters across all data
3. **Validate clusters**: Check persistence, cohesion, and known-class support
4. **Score novelty**: Noise points and invalid clusters flagged as novel

### Key Design Decisions

- **Combined clustering**: Reference and query points clustered together to detect novel clusters
- **Noise = novel**: Points not assigned to any cluster are automatically flagged
- **Cluster validation**: Not all clusters are valid — must meet quality thresholds
- **Batch-oriented**: Most effective when processing many queries together

### Implementation Details

- **Backend**: HDBSCAN library
- **Complexity**: O(n^2) for distance matrix, O(n log n) for MST
- **Best for batch**: Needs sufficient query samples to form clusters

---

## Findings

### Benchmark Performance

| Metric | Value |
|--------|-------|
| **Standalone AUROC** | Dataset-dependent |
| **Ensemble weight** | 0.20 |
| **Best use case** | Batch processing with novel clusters |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Detecting novel clusters (not just outliers) | **Use Clustering** |
| Batch processing with many samples | **Use Clustering** |
| Novel samples form coherent groups | **Use Clustering** |
| Complementary signal in ensemble | **Use Clustering** (weight: 0.20) |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Single query processing | Use **kNN Distance** or **SetFit Centroid Distance** |
| Few query samples (< min_cluster_size) | Use **kNN Distance** |
| Need fast inference | Use **SetFit Centroid Distance** |
| Standalone detection | Use **SetFit Centroid Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | O(n^2) for clustering |
| **Inference time** | O(n^2) for combined clustering |
| **Memory** | O(n^2) for distance matrix |
| **GPU required** | No |
| **Data requirement** | Batch of queries (≥ min_cluster_size) |
| **Best for** | Batch novelty detection |

### Strengths

- **Detects novel clusters**: Not just individual outliers, but groups of novel samples
- **No distributional assumptions**: Non-parametric, handles arbitrary cluster shapes
- **Automatic noise detection**: Points not in any cluster are flagged
- **Rich metadata**: Provides persistence, cohesion, and known support metrics
- **Good for batch analysis**: Can discover emerging novel classes

### Weaknesses

- **Computationally expensive**: O(n^2) complexity
- **Batch-oriented**: Needs many samples to be effective
- **Not suitable for single queries**: Cannot cluster a single point
- **Parameter sensitive**: min_cluster_size affects results significantly
- **Lower weight in ensemble**: 0.20 (least trusted among active strategies)

# kNN Distance Strategy

**Phase 2: Novelty Detection** | **Distance-based** | **Benchmark: 0.698 AUROC (raw), 0.866 AUROC (SetFit)**

---

## Mathematical Formulation

### k-Nearest Neighbor Search

For a query x with embedding e_x:

```
Find k nearest neighbors from reference set R:
  NN_k(x) = argmin_{e ∈ R} distance(e_x, e)  [k smallest distances]

For each neighbor n_i ∈ NN_k(x):
  d_i = distance(e_x, e_{n_i})
```

### Novelty Scoring

```
Using cosine distance:
  d_i = 1 - cosine_similarity(e_x, e_{n_i})
       = 1 - (e_x · e_{n_i}) / (||e_x|| · ||e_{n_i}||)

novelty_score = mean(d_1, d_2, ..., d_k)

is_novel = novelty_score ≥ distance_threshold
```

### Alternative: Nearest Neighbor Distance

```
novelty_score = min(d_1, d_2, ..., d_k)  # Distance to closest neighbor
```

### ANN-Backed Search

For large reference sets, approximate nearest neighbor search is used:

```
HNSW Index:
  Build: O(n · log(n) · M) where M = max connections
  Query: O(log(n) · M)

FAISS Index:
  Build: O(n · d) for flat index
  Query: O(n · d) for flat, O(log(n) · d) for IVF
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Reference embeddings: numpy array (n_samples × dim)       │
│  • Reference labels: List[str]                               │
│  • Query texts: List[str]                                    │
│  • k: int (default: 5)                                       │
│  • distance_threshold: float (default: 0.55)                 │
│  • metric: str ("cosine", "euclidean", "l2")                 │
│  • ANN backend: "hnswlib" or "faiss" (optional)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: INDEX BUILDING                         │
├─────────────────────────────────────────────────────────────┤
│  Technique: ANN Index (HNSWlib / FAISS) or Exact Search      │
│  • For ANN: Build HNSW graph or IVF index                    │
│    - HNSW: Hierarchical navigable small world graph          │
│    - FAISS: Inverted file or flat index                      │
│  • For exact: Store embeddings in numpy array                │
│  • Output: Search index structure                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: QUERY ENCODING                         │
├─────────────────────────────────────────────────────────────┤
│  Technique: SentenceTransformer / SetFit Encoder             │
│  • Encode each query text to embedding vector                │
│  • If SetFit: use fine-tuned encoder                         │
│  • If raw: use pre-trained encoder                           │
│  • Output: Query embeddings (n_queries × dim)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: k-NEAREST NEIGHBOR SEARCH              │
├─────────────────────────────────────────────────────────────┤
│  Technique: ANN Search or Exact kNN                          │
│  • For each query embedding:                                 │
│    - Search index for k nearest reference embeddings         │
│    - Return (distances, indices) pairs                       │
│  • ANN: Approximate but O(log n)                             │
│  • Exact: Precise but O(n)                                   │
│  • Output: distances (n_queries × k), indices (n_queries × k)│
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: NOVELTY SCORING                        │
├─────────────────────────────────────────────────────────────┤
│  Technique: Mean Distance Aggregation                        │
│  • For each query:                                           │
│    novelty_score = mean(distances[k])                        │
│    is_novel = novelty_score ≥ distance_threshold             │
│  • Optionally:                                               │
│    - nearest_class = most common label among k neighbors     │
│    - class_support = fraction of neighbors from nearest class│
│  • Output: (novelty_score, is_novel, nearest_class, metrics) │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 2] (cosine distance range)     │
│  • is_novel: bool                                            │
│  • nearest_class: str (majority class among k neighbors)     │
│  • distances: List[float] (k nearest distances)              │
│  • class_support: float (fraction from nearest class)        │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Build index**: Store reference embeddings in ANN or exact search structure
2. **Encode queries**: Transform query texts to embedding space
3. **Search kNN**: Find k nearest reference embeddings for each query
4. **Score novelty**: Average distance to k neighbors as novelty signal

### Key Design Decisions

- **Cosine distance**: Default metric, works well with normalized embeddings
- **Mean aggregation**: More stable than min or max distance
- **ANN support**: HNSWlib/FAISS for O(log n) search on large datasets
- **k = 3-5**: Sweet spot for most datasets (benchmark-validated)

### Implementation Details

- **ANN backends**: HNSWlib (default for speed), FAISS (alternative)
- **Metric options**: cosine, euclidean, l2
- **Class information**: Can report nearest class and class support ratio

---

## Configuration Options

Options are set via `KNNConfig`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `k` | int | 5 | [1, 100] | Number of nearest neighbors to consider |
| `distance_threshold` | float | 0.55 | [0.0, 1.0] | Mean distance threshold. Samples above this are flagged |
| `strong_threshold` | float | 0.85 | [0.0, 1.0] | High-confidence novelty threshold (heuristic gate) |
| `metric` | str | "cosine" | — | Distance metric: `cosine`, `euclidean`, etc. |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import KNNConfig

config = DetectionConfig(
    strategies=["knn_distance"],
    knn_distance=KNNConfig(
        k=20,
        distance_threshold=0.45,
        strong_threshold=0.85,
        metric="cosine",
    ),
)
```

---

## Findings

### Benchmark Performance

**ag_news (raw embeddings, k=20):**

| Metric | Value |
|--------|-------|
| **Test AUROC** | 0.698 |
| **DR@1%** | 1.1% |
| **DR@5%** | 5.1% |

**ag_news (SetFit embeddings, k=20):**

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.866** |
| **DR@1%** | 1.4% |
| **DR@5%** | 7.2% |

**go_emotions (k=3):**

| Metric | Value |
|--------|-------|
| **Test AUROC** | 0.976 |
| **DR@1%** | 94.6% |

**Strategy benchmark (ag_news, parameter sweep):**

| Parameters | AUROC | AUPRC | DR@1% |
|------------|-------|-------|-------|
| k=3, threshold=0.3 | **0.986** | **0.990** | 94.8% |
| k=5, threshold=0.3 | 0.982 | 0.987 | 93.2% |
| k=20, threshold=0.55 | 0.866 | 0.870 | 1.4% |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Large-scale datasets (> 100K entities) | **Use kNN with ANN** |
| Complementary signal in ensemble | **Use kNN Distance** |
| SetFit embeddings available | **Use kNN with SetFit** (0.866 AUROC) |
| Need class attribution (nearest class) | **Use kNN Distance** |
| Production default in ensemble | **Use kNN Distance** (weight: 0.45) |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Standalone with raw embeddings | Use **SetFit Centroid Distance** (0.886 vs 0.698) |
| Very small reference sets (< 100) | Use **Mahalanobis** (if Gaussian) |
| Entity name matching | Use **Pattern Strategy** |
| Need to detect novel clusters | Use **Clustering (HDBSCAN)** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | Near-zero (index build only) |
| **Inference time** | O(log n) with ANN, O(n) exact |
| **Memory** | O(n · d) for reference embeddings |
| **GPU required** | No |
| **Data requirement** | Reference embeddings needed |
| **Scalability** | Excellent with ANN (millions of references) |

### Strengths

- **Scalable**: ANN-backed search handles millions of references
- **No training**: Just needs reference embeddings
- **Class attribution**: Reports nearest class and support
- **Strong with SetFit**: 0.866 AUROC when using fine-tuned embeddings
- **Configurable k**: Tune k for precision/recall tradeoff
- **Highest weight in ensemble**: 0.45 (most trusted signal)

### Weaknesses

- **Raw embeddings underperform**: 0.698 AUROC without SetFit
- **Memory intensive**: Must store all reference embeddings
- **k sensitivity**: Too small k = noisy, too large k = diluted signal
- **Distance threshold tuning**: Requires calibration on validation data
- **Curse of dimensionality**: Less effective in very high dimensions without good embeddings

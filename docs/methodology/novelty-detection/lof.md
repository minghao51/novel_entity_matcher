# Local Outlier Factor (LOF) Strategy

**Phase 2: Novelty Detection** | **Density-based** | **Benchmark: 0.648 AUROC (ag_news), 0.850 AUROC (go_emotions)**

---

## Mathematical Formulation

### k-Distance

For a point x, the k-distance is the distance to its k-th nearest neighbor:

```
k-distance(x) = distance(x, k-th nearest neighbor of x)
```

### Reachability Distance

```
reach-dist_k(x, y) = max(k-distance(y), distance(x, y))

This smooths the distance metric by using the larger of:
- The actual distance between x and y
- The k-distance of y (local density around y)
```

### Local Reachability Density

```
LRD(x) = |N_k(x)| / Σ_{y ∈ N_k(x)} reach-dist_k(x, y)

where:
  N_k(x) = k-nearest neighbors of x
  |N_k(x)| = k (number of neighbors)
```

### Local Outlier Factor

```
LOF(x) = (1/|N_k(x)|) · Σ_{y ∈ N_k(x)} LRD(y) / LRD(x)

Interpretation:
  LOF(x) ≈ 1  → x has similar density to neighbors (known)
  LOF(x) > 1  → x is in a sparser region than neighbors (potentially novel)
  LOF(x) < 1  → x is in a denser region than neighbors (very typical)
```

### Novelty Scoring

```
# sklearn's LOF returns negative scores for inliers, positive for outliers
novelty_score = -LOF(x)  # Negate so higher = more novel

is_novel = novelty_score < score_threshold  # threshold typically 0.0
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
│  • n_neighbors: int (default: 20)                            │
│  • contamination: float (default: 0.1)                       │
│  • metric: str ("cosine", "euclidean", "minkowski")          │
│  • score_threshold: float (default: 0.0)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: MODEL FITTING                          │
├─────────────────────────────────────────────────────────────┤
│  Technique: sklearn LocalOutlierFactor (novelty=True)        │
│  • Build k-NN graph on reference embeddings                  │
│  • For each reference point:                                 │
│    - Compute k-distance                                      │
│    - Compute reachability distances to neighbors             │
│    - Compute local reachability density (LRD)                │
│    - Compute LOF score                                       │
│  • Store reference data and precomputed distances            │
│  • Output: Fitted LOF model                                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: QUERY ENCODING                         │
├─────────────────────────────────────────────────────────────┤
│  Technique: SentenceTransformer / SetFit Encoder             │
│  • Encode query text to embedding vector                     │
│  • Output: Query embeddings (n_queries × dim)                │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: LOF SCORING                            │
├─────────────────────────────────────────────────────────────┤
│  Technique: Local Outlier Factor Prediction                  │
│  • For each query x:                                         │
│    - Find k nearest neighbors in reference set               │
│    - Compute reachability distances                          │
│    - Compute LRD(x)                                          │
│    - Compute LOF(x) = avg(LRD(y)/LRD(x)) for y in N_k(x)    │
│  • Output: LOF scores for all queries                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: NOVELTY DECISION                       │
├─────────────────────────────────────────────────────────────┤
│  Technique: Threshold-Based Classification                   │
│  • novelty_score = -LOF_score (negate for consistency)       │
│  • is_novel = novelty_score < score_threshold                │
│  • Output: (novelty_score, is_novel, lof_score)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float (negated LOF score)                  │
│  • is_novel: bool                                            │
│  • lof_score: float (raw LOF score from sklearn)             │
│  • n_neighbors: int (k used)                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Fit LOF model**: Build k-NN graph and compute local densities on reference data
2. **Encode queries**: Transform query texts to embedding space
3. **Score queries**: Compute LOF for each query relative to reference distribution
4. **Decide novelty**: Compare LOF score against threshold

### Key Design Decisions

- **novelty=True mode**: sklearn's novelty detection mode (not outlier detection)
- **Cosine distance**: Default metric for embedding spaces
- **contamination parameter**: Expected fraction of outliers in training data
- **n_neighbors = 20**: Default, balances local vs global density estimation

### Implementation Details

- **Backend**: sklearn's `LocalOutlierFactor` with `novelty=True`
- **Complexity**: O(n^2) for fitting, O(n) per query
- **Metric options**: cosine, euclidean, minkowski

---

## Configuration Options

Options are set via `LOFConfig`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `n_neighbors` | int | 20 | ≥ 2 | Number of neighbors for LOF computation |
| `contamination` | float | 0.1 | (0.0, 0.5] | Expected proportion of outliers in reference set |
| `metric` | str | "cosine" | "cosine", "euclidean", "manhattan", etc. | Distance metric |
| `score_threshold` | float | 0.0 | — | LOF score threshold. Samples below this are flagged as novel |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import LOFConfig

config = DetectionConfig(
    strategies=["lof"],
    lof=LOFConfig(
        n_neighbors=30,
        contamination=0.05,
        metric="cosine",
        score_threshold=-0.5,
    ),
)
```

---

## Findings

### Benchmark Performance

**ag_news (n=20):**

| Metric | Value |
|--------|-------|
| **Validation AUROC** | 0.677 |
| **Test AUROC** | **0.648** |
| **DR@1%** | 1.5% |
| **DR@5%** | 4.6% |

**go_emotions:**

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.850** |
| **AUPRC** | **0.828** |
| **DR@1%** | 12.8% |
| **DR@5%** | 33.6% |

**Strategy benchmark (ag_news, parameter sweep):**

| Parameters | AUROC | AUPRC | DR@1% |
|------------|-------|-------|-------|
| n=50, contamination=0.05 | **0.850** | **0.828** | 12.8% |
| n=20, contamination=0.1 | 0.648 | 0.650 | 1.5% |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Datasets with varying density across classes | **Use LOF** |
| Density-based anomaly detection needed | **Use LOF** |
| Complementary signal in ensemble | **Use LOF** (weight: 0.30) |
| go_emotions-like datasets | **Use LOF** (0.850 AUROC) |
| Need to detect local outliers (not global) | **Use LOF** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| ag_news-like datasets with raw embeddings | Use **SetFit Centroid Distance** (0.886 vs 0.648) |
| Very large reference sets (> 100K) | Use **kNN Distance** (faster) |
| Uniform density across classes | Use **Mahalanobis** |
| Standalone detection | Use **SetFit Centroid Distance** |
| Need fast inference | Use **kNN Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | O(n^2) for fitting |
| **Inference time** | O(n) per query |
| **Memory** | O(n · d) for reference data |
| **GPU required** | No |
| **Data requirement** | Moderate (100+ reference samples) |
| **Assumption** | Novel samples have lower local density than known samples |

### Strengths

- **Handles varying densities**: Detects local outliers, not just global ones
- **Good on go_emotions**: 0.850 AUROC
- **No distributional assumptions**: Non-parametric, works with any density shape
- **Complementary signal**: Captures different novelty patterns than distance-based methods
- **Well-established**: sklearn implementation, well-tested

### Weaknesses

- **Sensitive to n_neighbors**: Requires tuning for each dataset
- **Slower than kNN**: O(n^2) fitting, O(n) inference
- **Underperforms on ag_news**: 0.648 AUROC with default parameters
- **contamination sensitivity**: Expected outlier fraction must be estimated
- **Memory intensive**: Must store all reference data and k-NN graph
- **Not ideal as standalone**: Better as part of ensemble

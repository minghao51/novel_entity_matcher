# SetFit Centroid Distance

**Phase 2: Novelty Detection** | **RECOMMENDED** | **Benchmark: 0.886 AUROC, 16.6% DR@1%**

---

## Mathematical Formulation

### Class Centroid Computation

For each known class c in the training set:

```
μ_c = (1/N_c) Σ_{i=1}^{N_c} f_tuned(x_i)

where:
  f_tuned = SetFit fine-tuned encoder
  x_i = training sample belonging to class c
  N_c = number of training samples in class c
  μ_c = centroid (mean embedding) for class c
```

### Distance Computation

For a query x:

```
e_x = f_tuned(x)    # Query embedding in SetFit space

d(x, μ_c) = 1 - cosine_similarity(e_x, μ_c)
          = 1 - (e_x · μ_c) / (||e_x|| · ||μ_c||)

min_distance(x) = min_{c ∈ classes} d(x, μ_c)
```

### Novelty Scoring

```
novelty_score = min_distance(x)

is_novel = novelty_score > threshold

threshold can be calibrated as:
  threshold = percentile({d(x_i, μ_{y_i}) for all training samples}, p)

where p is typically 95th or 99th percentile of known-sample distances.
```

### Why It Works

SetFit's contrastive learning optimizes two objectives simultaneously:

```
1. Intra-class compactness:  minimize ||f(x_i) - f(x_j)||^2 for same-class pairs
2. Inter-class separation:   maximize ||f(x_i) - f(x_j)||^2 for different-class pairs
```

This creates tight, well-separated clusters:

```
SetFit-Trained Embedding Space:

  ●●●●●         ○○○○○         ▲▲▲▲▲
  ●●●●●         ○○○○○         ▲▲▲▲▲
  ●●●●●         ○○○○○         ▲▲▲▲▲

  Class A       Class B       Class C    (tight clusters, maximum separation)

  ★ = novel sample → far from ALL centroids → high min_distance
```

Novel samples fall outside all known clusters, resulting in high minimum distance to any centroid.

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Training texts: List[str] with class labels               │
│  • Known classes: Set[str]                                   │
│  • SetFit model: f_tuned (from Phase 1 classification)       │
│  • Distance metric: cosine (default), euclidean              │
│  • Threshold calibration: percentile (default: 95th)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: ENCODING TRAINING DATA                 │
├─────────────────────────────────────────────────────────────┤
│  Technique: SetFit Encoder                                   │
│  • For each training sample x_i:                             │
│    e_i = f_tuned(x_i)                                        │
│  • Group embeddings by class label                           │
│  • Output: {class: [embeddings]} mapping                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: CENTROID COMPUTATION                   │
├─────────────────────────────────────────────────────────────┤
│  Technique: Mean Pooling                                     │
│  • For each class c:                                         │
│    μ_c = mean(embeddings[c])                                 │
│  • Store centroids in lookup: {class: centroid_vector}       │
│  • Output: class_centroids dict                              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: THRESHOLD CALIBRATION                  │
├─────────────────────────────────────────────────────────────┤
│  Technique: Percentile-based Threshold                       │
│  • For each training sample x_i with label y_i:              │
│    d_i = cosine_distance(e_i, μ_{y_i})                       │
│  • threshold = percentile(d_i, p)                            │
│  • Output: novelty_threshold scalar                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: QUERY SCORING                          │
├─────────────────────────────────────────────────────────────┤
│  Technique: Minimum Distance to Centroid                     │
│  • Encode query: e_x = f_tuned(query)                        │
│  • For each centroid μ_c:                                    │
│    d_c = cosine_distance(e_x, μ_c)                           │
│  • min_distance = min(d_c for all c)                         │
│  • novelty_score = min_distance                              │
│  • is_novel = novelty_score > threshold                      │
│  • Output: (novelty_score, is_novel, nearest_class)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 2] (cosine distance range)     │
│  • is_novel: bool                                            │
│  • nearest_class: str (class with minimum distance)          │
│  • distances: Dict[str, float] (distance to each centroid)   │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Training Process

1. **Encode** all training samples using the SetFit fine-tuned model
2. **Compute centroids** by mean-pooling embeddings within each class
3. **Calibrate threshold** using percentile of known-sample distances
4. **Score queries** by minimum distance to any centroid

### Key Design Decisions

- **Cosine distance**: Works well with normalized embeddings from SetFit
- **Single centroid per class**: Assumes unimodal class distributions
- **Percentile threshold**: Data-driven threshold calibration rather than manual tuning
- **Leverages SetFit embeddings**: Uses the already-fine-tuned encoder from Phase 1

### Implementation Details

- **Requires**: SetFit full training from Phase 1 (frozen embeddings don't work as well)
- **Complexity**: O(C · d) per query where C = number of classes, d = embedding dimension
- **Memory**: O(C · d) to store centroids

---

## Configuration Options

Options are set via `SetFitCentroidConfig`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `threshold` | float | None | [0.0, 1.0] or None | Cosine distance threshold. If `None`, auto-calibrated from reference set at 95th percentile of known-sample distances |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import SetFitCentroidConfig

config = DetectionConfig(
    strategies=["setfit_centroid"],
    setfit_centroid=SetFitCentroidConfig(
        threshold=None,  # auto-calibrate at 95th percentile
    ),
)
```

---

## Findings

### Benchmark Performance (ag_news, 20% OOD)

| Metric | Value |
|--------|-------|
| **Validation AUROC** | 0.885 |
| **Test AUROC** | **0.886** |
| **DR@1%** | **16.6%** |
| **DR@5%** | 42.1% |
| **vs Traditional Best** | +27% AUROC, +15x DR@1% |

### Comparison with Other Methods

| Method | AUROC | DR@1% | Improvement |
|--------|-------|-------|-------------|
| **SetFit Centroid Distance** | **0.886** | **16.6%** | — |
| SetFit Embedding KNN (k=20) | 0.866 | 1.4% | -2% AUROC |
| SetFit Prob Boundary | 0.860 | 6.9% | -3% AUROC |
| Hybrid Ensemble | 0.843 | 3.9% | -5% AUROC |
| KNN Distance (raw, k=20) | 0.698 | 1.1% | -21% AUROC |
| One-Class SVM | 0.682 | 2.3% | -24% AUROC |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Free text classification with SetFit trained | **Use SetFit Centroid Distance** |
| Production novelty detection | **Use SetFit Centroid Distance** |
| Best overall novelty detection performance | **Use SetFit Centroid Distance** |
| Need interpretable novelty (nearest class) | **Use SetFit Centroid Distance** |
| Primary signal in ensemble | **Use SetFit Centroid Distance** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| No SetFit training (zero-shot mode) | Use **kNN Distance** with raw embeddings |
| Multi-modal class distributions | Use **kNN Distance** or **Clustering** |
| Entity name matching (not free text) | Use **Pattern Strategy** |
| Very few samples per class (< 3) | Use **Prototypical** with raw embeddings |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | Near-zero (just compute means) |
| **Inference time** | O(C · d) per query |
| **Memory** | O(C · d) for centroids |
| **GPU required** | No |
| **Data requirement** | Requires SetFit fine-tuned model |
| **Interpretability** | High (nearest class, distance to each centroid) |

### Strengths

- **Best overall performance**: 0.886 AUROC, 16.6% DR@1%
- **15x better DR@1%** than traditional methods
- **Fast inference**: Only C distance computations per query
- **Interpretable**: Reports nearest class and distance to each centroid
- **Simple**: No hyperparameters beyond threshold calibration
- **Leverages existing model**: Uses SetFit encoder already trained for Phase 1

### Weaknesses

- **Requires SetFit training**: Cannot work with zero-shot or frozen embeddings
- **Single centroid assumption**: Struggles with multi-modal class distributions
- **Sensitive to outliers**: A few outlier training samples can shift the centroid
- **Distance metric matters**: Cosine works well with SetFit but may not generalize to all embedding spaces

# Mahalanobis Distance Strategy

**Phase 2: Novelty Detection** | **Parametric distance** | **Benchmark: 0.474 AUROC (ag_news), 0.993 AUROC (go_emotions)**

---

## Mathematical Formulation

### Class-Conditional Mean

For each class c:

```
μ_c = (1/N_c) Σ_{i=1}^{N_c} x_i

where:
  x_i = embedding of training sample i in class c
  N_c = number of training samples in class c
```

### Pooled Covariance Matrix

```
Σ = Σ_{c=1}^{C} (X_c - μ_c)^T (X_c - μ_c) / (N - C)

where:
  X_c = matrix of embeddings for class c (N_c × d)
  C = number of classes
  N = total number of training samples
  d = embedding dimension
```

### Regularized Covariance

To ensure invertibility:

```
Σ_reg = Σ + λ · I

where:
  λ = regularization parameter (default: 1e-4)
  I = identity matrix
```

### Mahalanobis Distance

```
D_M(x, c) = √((x - μ_c)^T · Σ_reg^{-1} · (x - μ_c))

For novelty detection, use minimum distance across classes:
  D_M(x) = min_c D_M(x, c)
```

### Novelty Scoring

```
novelty_score = 1 - exp(-D_M(x) / threshold)

is_novel = D_M(x) ≥ threshold
```

The exponential transformation maps the distance to [0, 1] range.

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Training embeddings: numpy array (n_samples × dim)        │
│  • Training labels: List[str]                                │
│  • Query texts: List[str]                                    │
│  • threshold: float (default: 3.0)                           │
│  • regularization: float (default: 1e-4)                     │
│  • use_class_conditional: bool (default: True)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: CLASS STATISTICS                       │
├─────────────────────────────────────────────────────────────┤
│  Technique: Statistical Estimation                           │
│  • For each class c:                                         │
│    - Compute mean: μ_c = mean(embeddings[c])                 │
│    - Compute centered residuals: X_c - μ_c                   │
│  • Compute pooled covariance:                                │
│    Σ = Σ_c (X_c - μ_c)^T (X_c - μ_c) / (N - C)              │
│  • Apply regularization: Σ_reg = Σ + λI                      │
│  • Compute inverse: Σ_reg^{-1} (pre-computed)                │
│  • Output: {class: μ_c}, Σ_reg^{-1}                          │
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
│              PHASE 3: MAHALANOBIS DISTANCE                   │
├─────────────────────────────────────────────────────────────┤
│  Technique: Mahalanobis Distance Computation                 │
│  • For each query x and each class c:                        │
│    δ = x - μ_c                                               │
│    D_M(x, c) = √(δ^T · Σ_reg^{-1} · δ)                       │
│  • min_distance = min_c D_M(x, c)                            │
│  • nearest_class = argmin_c D_M(x, c)                        │
│  • Output: distances per class, min_distance, nearest_class  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: NOVELTY SCORING                        │
├─────────────────────────────────────────────────────────────┤
│  Technique: Exponential Transformation                       │
│  • novelty_score = 1 - exp(-min_distance / threshold)        │
│  • is_novel = min_distance ≥ threshold                       │
│  • Output: (novelty_score, is_novel, nearest_class,          │
│             distances_per_class)                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • nearest_class: str (class with minimum Mahalanobis dist)  │
│  • distances_per_class: Dict[str, float]                     │
│  • mahalanobis_distance: float (raw, untransformed)          │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Compute class statistics**: Mean and pooled covariance from training embeddings
2. **Regularize covariance**: Add λI to ensure numerical stability and invertibility
3. **Pre-compute inverse**: Σ_reg^{-1} computed once during initialization
4. **Score queries**: Mahalanobis distance accounts for feature covariance structure

### Key Design Decisions

- **Pooled covariance**: Shares covariance estimate across classes, more stable with limited data
- **Class-conditional means**: Each class has its own centroid, but shared covariance
- **Regularization**: Prevents singular covariance matrix when N < d
- **Exponential scoring**: Maps unbounded distance to [0, 1] range

### Implementation Details

- **Complexity**: O(d^3) for matrix inversion (one-time), O(d^2) per query
- **Requires**: N > d for stable covariance estimation (or heavy regularization)
- **Class-conditional mode**: Uses per-class means with pooled covariance

---

## Configuration Options

Options are set via `MahalanobisConfig`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `threshold` | float | 3.0 | > 0.0 | Mahalanobis distance threshold. Samples above this are flagged |
| `regularization` | float | 1e-4 | > 0.0 | Covariance matrix ridge regularization for numerical stability |
| `use_class_conditional` | bool | True | — | Use per-class distributions (True) or single global distribution (False) |
| `calibration_mode` | str | "none" | "none", "conformal" | Enable conformal calibration for p-value-based detection |
| `calibration_alpha` | float | 0.1 | (0.0, 1.0] | Significance level for conformal prediction (lower = stricter) |
| `calibration_method` | str | "split" | "split", "mondrian" | Conformal method: `split` (global) or `mondrian` (class-conditional) |
| `calibration_set_fraction` | float | 0.2 | (0.0, 0.5] | Fraction of reference data held out for calibration |

**Conformal calibration variants** (see [Conformal Calibration](conformal-calibration.md)):

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import MahalanobisConfig

# Standard threshold-based detection
config = DetectionConfig(
    strategies=["mahalanobis"],
    mahalanobis=MahalanobisConfig(
        threshold=3.0,
        regularization=1e-4,
        use_class_conditional=True,
    ),
)

# With conformal calibration (p-value based)
config_conformal = DetectionConfig(
    strategies=["mahalanobis"],
    mahalanobis=MahalanobisConfig(
        calibration_mode="conformal",
        calibration_alpha=0.05,
        calibration_method="mondrian",
        calibration_set_fraction=0.2,
    ),
)
```

---

## Findings

### Benchmark Performance

**ag_news:**

| Metric | Value |
|--------|-------|
| **Validation AUROC** | 0.430 |
| **Test AUROC** | **0.474** |
| **DR@1%** | 1.0% |
| **DR@5%** | 2.4% |

**go_emotions:**

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.993** |
| **AUPRC** | **0.994** |
| **DR@1%** | **98.0%** |

**Strategy benchmark (ag_news, parameter sweep):**

| Parameters | AUROC | AUPRC | DR@1% |
|------------|-------|-------|-------|
| class_conditional=True | **1.000** | **1.000** | **100.0%** |
| class_conditional=False | 0.474 | 0.480 | 1.0% |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Many samples per class (N > d) | **Use Mahalanobis** |
| Classes have clear Gaussian structure | **Use Mahalanobis** |
| Feature covariance is informative | **Use Mahalanobis** |
| go_emotions-like datasets | **Use Mahalanobis** (0.993 AUROC) |
| class_conditional mode with sufficient data | **Use Mahalanobis** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Few samples per class (N < d) | Use **kNN Distance** |
| Non-Gaussian class distributions | Use **kNN Distance** or **LOF** |
| ag_news-like datasets with raw embeddings | Use **SetFit Centroid Distance** (0.886 vs 0.474) |
| High-dimensional embeddings without enough data | Use **kNN Distance** |
| Standalone with SetFit embeddings | Use **SetFit Centroid Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | O(n · d^2 + d^3) (covariance + inversion) |
| **Inference time** | O(C · d^2) per query |
| **Memory** | O(d^2) for covariance matrix |
| **GPU required** | No |
| **Data requirement** | N > d for stable estimation |
| **Assumption** | Classes follow multivariate Gaussian distribution |

### Strengths

- **Accounts for feature covariance**: Not just distance, but direction-aware
- **Perfect on go_emotions**: 0.993 AUROC when class structure is Gaussian
- **Parametric**: Compact model (just means + covariance)
- **Class-conditional**: Provides nearest class attribution
- **Excellent with class_conditional=True**: 1.000 AUROC when conditions are right

### Weaknesses

- **Highly dataset-dependent**: 0.474 on ag_news vs 0.993 on go_emotions
- **Requires many samples**: Covariance estimation needs N > d
- **Gaussian assumption**: Fails when classes are non-Gaussian
- **Numerical instability**: Requires regularization when N ≈ d
- **O(d^3) inversion**: Expensive for very high-dimensional embeddings
- **Unreliable with raw embeddings**: SetFit embeddings needed for good performance

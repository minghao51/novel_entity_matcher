# One-Class SVM Strategy

**Phase 2: Novelty Detection** | **Boundary-based** | **Benchmark: 0.682 AUROC (ag_news)**

---

## Mathematical Formulation

### Decision Function

One-Class SVM finds a hyperplane that separates the data from the origin in feature space:

```
f(x) = Σ_{i=1}^{n} α_i · K(x_i, x) - ρ

where:
  α_i = dual coefficients (learned during training)
  x_i = support vectors (subset of training data)
  K(x_i, x) = kernel function
  ρ = offset (learned during training)
```

### RBF Kernel

```
K(x, y) = exp(-γ · ||x - y||^2)

where:
  γ = kernel coefficient (default: "scale" = 1 / (n_features · Var(X)))
  ||x - y||^2 = squared Euclidean distance
```

### Novelty Decision

```
Prediction:
  f(x) ≥ 0 → inlier (known class)
  f(x) < 0 → outlier (novel)

Confidence scoring:
  novelty_score = clip(-f(x), 0, 1)
  
  # Normalize decision function to [0, 1]
  novelty_score = clip(-decision_function(x) / scale, 0, 1)

is_novel = f(x) < 0  (predicted as -1)
```

### Optimization Objective

```
minimize: (1/2) ||w||^2 + (1/νn) Σ_i ξ_i - ρ

subject to:
  w · φ(x_i) ≥ ρ - ξ_i    for all i
  ξ_i ≥ 0                   for all i

where:
  ν = nu parameter (upper bound on fraction of outliers, default: 0.1)
  ξ_i = slack variables
  φ(x) = feature mapping (implicit via kernel)
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
│  • nu: float (default: 0.1)                                  │
│  • kernel: str ("rbf", "linear", "poly", "sigmoid")          │
│  • gamma: str/float ("scale", "auto", or value)              │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: MODEL FITTING                          │
├─────────────────────────────────────────────────────────────┤
│  Technique: sklearn OneClassSVM                              │
│  • Compute kernel matrix: K_ij = K(x_i, x_j)                │
│  • Solve quadratic optimization problem:                     │
│    - Find support vectors (subset of training data)          │
│    - Learn dual coefficients α_i                             │
│    - Learn offset ρ                                          │
│  • ν controls tradeoff: fraction of training data as outliers│
│  • Output: Fitted OneClassSVM model                          │
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
│              PHASE 3: DECISION FUNCTION                      │
├─────────────────────────────────────────────────────────────┤
│  Technique: Kernel Evaluation                                │
│  • For each query x:                                         │
│    f(x) = Σ_{i ∈ SV} α_i · K(x_i, x) - ρ                    │
│  • Only support vectors contribute (sparse solution)         │
│  • Output: decision_function values for all queries          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: NOVELTY SCORING                        │
├─────────────────────────────────────────────────────────────┤
│  Technique: Normalized Decision Function                     │
│  • novelty_score = clip(-f(x), 0, 1)                         │
│  • is_novel = f(x) < 0                                       │
│  • confidence = clip(f(x), 0, 1)  # For inlier confidence    │
│  • Output: (novelty_score, is_novel, decision_value)         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • decision_value: float (raw decision function output)      │
│  • n_support_vectors: int (model complexity indicator)       │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Fit One-Class SVM**: Learn boundary that encloses known data in kernel space
2. **Encode queries**: Transform query texts to embedding space
3. **Evaluate decision function**: Compute f(x) for each query
4. **Score novelty**: Negative decision function values indicate novelty

### Key Design Decisions

- **RBF kernel**: Default, captures non-linear boundaries
- **nu parameter**: Controls expected outlier fraction (tradeoff between sensitivity and specificity)
- **gamma="scale"**: Automatic kernel width based on data variance
- **No negative examples needed**: Only learns from known class data

### Implementation Details

- **Backend**: sklearn's `OneClassSVM`
- **Complexity**: O(n^2) to O(n^3) for training, O(n_sv · d) per query
- **Does not scale well**: Performance degrades with > 100K samples

---

## Configuration Options

Options are set via `OneClassConfig`:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `nu` | float | 0.1 | [0.0, 1.0] | Expected outlier fraction. Lower = stricter boundary |
| `kernel` | str | "rbf" | "rbf", "linear", "poly", "sigmoid" | SVM kernel type |
| `gamma` | str | "scale" | "scale", "auto", or float | Kernel coefficient |
| `model_name` | str | "sentence-transformers/all-MiniLM-L6-v2" | — | Sentence transformer model for embeddings |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import OneClassConfig

config = DetectionConfig(
    strategies=["oneclass"],
    oneclass=OneClassConfig(
        nu=0.05,
        kernel="rbf",
        gamma="scale",
    ),
)
```

---

## Findings

### Benchmark Performance

**ag_news (nu=0.1):**

| Metric | Value |
|--------|-------|
| **Validation AUROC** | 0.709 |
| **Test AUROC** | **0.682** |
| **DR@1%** | 2.3% |
| **DR@5%** | 5.8% |

**Strategy benchmark (ag_news, parameter sweep):**

| Parameters | AUROC | AUPRC | DR@1% |
|------------|-------|-------|-------|
| nu=0.05, rbf | **0.836** | **0.818** | 6.0% |
| nu=0.1, rbf | 0.682 | 0.690 | 2.3% |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Learning boundary of known class distribution | **Use One-Class SVM** |
| Small to medium datasets (100 - 10,000 entities) | **Use One-Class SVM** |
| No negative examples available | **Use One-Class SVM** |
| Non-linear boundary needed | **Use One-Class SVM** |
| Complementary signal in ensemble | **Use One-Class SVM** (weight: 0.10) |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Large datasets (> 100K samples) | Use **kNN Distance** with ANN |
| ag_news-like datasets | Use **SetFit Centroid Distance** (0.886 vs 0.682) |
| Need fast inference | Use **kNN Distance** |
| Linear boundary sufficient | Use **Mahalanobis** |
| GPU resources limited with large data | Use **kNN Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | O(n^2) to O(n^3) |
| **Inference time** | O(n_sv · d) per query |
| **Memory** | O(n_sv · d) for support vectors |
| **GPU required** | No |
| **Data requirement** | 100+ reference samples |
| **Assumption** | Known data can be separated from origin in kernel space |

### Strengths

- **Learns complex boundaries**: RBF kernel captures non-linear shapes
- **No negative examples**: Only needs known class data
- **Well-understood**: Decades of research, sklearn implementation
- **Fast training for small datasets**: Seconds to minutes
- **No GPU required**: Pure CPU computation
- **Tunable sensitivity**: nu parameter controls outlier fraction

### Weaknesses

- **Does not scale**: O(n^2) to O(n^3) training, poor beyond 100K samples
- **Sensitive to nu**: Requires careful tuning
- **Kernel selection matters**: RBF may not be optimal for all data
- **Less interpretable**: Hard to understand why a sample is classified as novel
- **Underperforms SetFit methods**: 0.682 vs 0.886 AUROC
- **Many support vectors**: Can be memory-intensive for large datasets

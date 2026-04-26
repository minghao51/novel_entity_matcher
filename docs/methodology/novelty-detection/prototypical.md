# Prototypical Strategy

**Phase 2: Novelty Detection** | **Prototype-based** | **Weight: 0.10 in ensemble**

---

## Mathematical Formulation

### Prototype (Centroid) Computation

For each known class c:

```
prototype_c = (1/N_c) Σ_{i=1}^{N_c} f(x_i)

where:
  f = encoder (pre-trained or SetFit fine-tuned)
  x_i = training sample from class c
  N_c = number of training samples in class c
```

### Distance to Nearest Prototype

```
For query x with embedding e_x:

d(x, prototype_c) = distance_metric(e_x, prototype_c)

min_distance(x) = min_c d(x, prototype_c)
nearest_class = argmin_c d(x, prototype_c)
```

### Distance Metrics

**Cosine Distance (default):**
```
d_cosine(a, b) = 1 - (a · b) / (||a|| · ||b||)
```

**Euclidean Distance:**
```
d_euclidean(a, b) = ||a - b||_2
```

**Mahalanobis Distance:**
```
d_mahalanobis(a, b) = √((a - b)^T · Σ^{-1} · (a - b))
```

### Novelty Scoring

```
novelty_score = min_distance(x)

is_novel = novelty_score > distance_threshold
```

### Multi-Prototype Extension

For multi-modal class distributions:

```
For each class c:
  Cluster embeddings into K_c sub-clusters
  prototype_c_k = centroid of sub-cluster k

d(x) = min_{c, k} distance(e_x, prototype_c_k)
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Training texts: List[str] with class labels               │
│  • Known classes: Set[str]                                   │
│  • Encoder: sentence-transformer (pre-trained or SetFit)     │
│  • distance_threshold: float (default: 0.5)                  │
│  • distance_metric: str ("cosine", "euclidean", "mahalanobis")│
│  • model_name: str (encoder model identifier)                │
│  • support_samples_per_class: int (default: 5)               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: ENCODING TRAINING DATA                 │
├─────────────────────────────────────────────────────────────┤
│  Technique: SentenceTransformer / SetFit Encoder             │
│  • For each training sample x_i:                             │
│    e_i = f(x_i)                                              │
│  • Group embeddings by class label                           │
│  • Output: {class: [embeddings]} mapping                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: PROTOTYPE COMPUTATION                  │
├─────────────────────────────────────────────────────────────┤
│  Technique: Mean Pooling (or Clustering for multi-prototype) │
│  • For each class c:                                         │
│    prototype_c = mean(embeddings[c])                         │
│  • Optionally: cluster within class for multi-prototype      │
│  • Store prototypes in lookup: {class: prototype_vector}     │
│  • Output: prototypes dict                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: THRESHOLD CALIBRATION                  │
├─────────────────────────────────────────────────────────────┤
│  Technique: Distance Distribution Analysis                   │
│  • For each training sample x_i with label y_i:              │
│    d_i = distance(e_i, prototype_{y_i})                      │
│  • threshold = percentile(d_i, p) or fixed value             │
│  • Output: distance_threshold scalar                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: QUERY SCORING                          │
├─────────────────────────────────────────────────────────────┤
│  Technique: Minimum Distance to Prototype                    │
│  • Encode query: e_x = f(query)                              │
│  • For each prototype_c:                                     │
│    d_c = distance_metric(e_x, prototype_c)                   │
│  • min_distance = min(d_c for all c)                         │
│  • nearest_class = argmin_c d_c                              │
│  • novelty_score = min_distance                              │
│  • is_novel = novelty_score > distance_threshold             │
│  • Output: (novelty_score, is_novel, nearest_class)          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float                                      │
│  • is_novel: bool                                            │
│  • nearest_class: str (class with minimum distance)          │
│  • distances: Dict[str, float] (distance to each prototype)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Encode** all training samples using the encoder
2. **Compute prototypes**: Mean embedding for each class
3. **Calibrate threshold**: Using distance distribution of known samples
4. **Score queries**: Minimum distance to any prototype

### Key Design Decisions

- **Simple and interpretable**: One prototype per class, easy to understand
- **Multiple distance metrics**: Cosine (default), Euclidean, Mahalanobis
- **Few-shot friendly**: Works with limited training data
- **Fast inference**: O(C) distance computations per query

### Implementation Details

- **Weight in ensemble**: 0.10
- **Complexity**: O(C · d) per query
- **Memory**: O(C · d) for prototypes

---

## Configuration Options

Options are set via `PrototypicalConfig`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `distance_threshold` | float | 0.5 | [0.0, 1.0] | Distance threshold for novelty detection |
| `model_name` | str | "sentence-transformers/all-MiniLM-L6-v2" | — | Sentence transformer model for embeddings |
| `support_samples_per_class` | int | 5 | ≥ 1 | Number of support samples per class for prototype computation |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import PrototypicalConfig

config = DetectionConfig(
    strategies=["prototypical"],
    prototypical=PrototypicalConfig(
        distance_threshold=0.6,
        support_samples_per_class=3,
    ),
)
```

---

## Findings

### Benchmark Performance

| Metric | Value |
|--------|-------|
| **Standalone AUROC** | Dataset-dependent |
| **Ensemble weight** | 0.10 |
| **Best use case** | Few-shot scenarios with clear class structure |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Interpretable novelty detection | **Use Prototypical** |
| When class information is valuable | **Use Prototypical** |
| Multi-class scenarios with clear structure | **Use Prototypical** |
| Few-shot (< 50 samples) | **Use Prototypical + kNN** |
| Fast prototyping and experimentation | **Use Prototypical** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Multi-modal class distributions | Use **kNN Distance** |
| Standalone with SetFit embeddings | Use **SetFit Centroid Distance** |
| High-dimensional embeddings without reduction | Use **kNN Distance** |
| Need best accuracy | Use **SetFit Centroid Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | Near-zero (just compute means) |
| **Inference time** | O(C · d) per query |
| **Memory** | O(C · d) for prototypes |
| **GPU required** | No |
| **Data requirement** | Labeled training data (class labels) |
| **Interpretability** | High (nearest class, distance to each prototype) |

### Strengths

- **Simple and interpretable**: Easy to understand and debug
- **Fast inference**: Only C distance computations per query
- **Few-shot friendly**: Works with limited training data
- **Provides class information**: Reports nearest known class
- **No GPU required**: Pure CPU computation
- **Fast "training"**: Just computing means

### Weaknesses

- **Single prototype assumption**: Struggles with multi-modal class distributions
- **Threshold selection requires calibration**: No automatic threshold
- **Requires labeled data**: Needs class labels for prototype computation
- **Sensitive to outliers**: A few outlier samples can shift the prototype
- **Lower weight in ensemble**: 0.10 (less trusted than kNN or confidence)

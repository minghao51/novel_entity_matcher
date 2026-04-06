# Methodology: Classification & Novelty Detection

This document provides the mathematical formulations, architectural DAGs, and methodological details for the two-phase pipeline used in Novel Entity Matcher.

**Related**: [Novelty Detection Benchmark](../novelty-detection-benchmark.md) | [Novel Class Detection](../novel-class-detection.md) | [Matcher Modes](../matcher-modes.md)

---

## Pipeline Overview

```
                    ┌─────────────────────────────┐
                    │       Query Input            │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   TEXT NORMALIZER            │
                    │   (optional preprocessing)   │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │   PHASE 1A       │ │   PHASE 1B       │ │   PHASE 1C       │
    │   CLASSIFICATION │ │   CLASSIFICATION │ │   CLASSIFICATION │
    │   (SetFit Full)  │ │   (SetFit Head)  │ │   (BERT)         │
    │   91.2% acc      │ │   54.7% acc      │ │   ~95% acc       │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   MatchResult              │
                    │   - predicted_id           │
                    │   - confidence scores      │
                    │   - embedding vector       │
                    │   - top-k candidates       │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   PHASE 2: NOVELTY          │
                    │   DETECTION                 │
                    │   SetFit Centroid Distance  │
                    │   AUROC: 0.886              │
                    │   DR@1%: 16.6%              │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   Signal Fusion             │
                    │   (weighted/union/voting)   │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   Decision                  │
                    │   KNOWN class assignment    │
                    │   or NOVEL class flag       │
                    └─────────────────────────────┘
```

---

## Phase 1: Classification

Classification assigns a query to one of the known entity classes. The system supports multiple modes that route to different underlying implementations.

### 1.1 SetFit Full Training (Recommended)

**Implementation**: `src/novelentitymatcher/core/classifier.py`  
**Benchmark Performance**: 91.2% test accuracy on ag_news (500 training samples)

#### Mathematical Formulation

**Stage 1: Contrastive Fine-Tuning**

SetFit generates pairs from labeled data and optimizes a contrastive loss:

```
For each pair (x_i, x_j):
  z_i = f(x_i)    # SentenceTransformer encoding
  z_j = f(x_j)

  If same class (positive pair):
    L_pos = ||z_i - z_j||^2

  If different class (negative pair):
    L_neg = max(0, margin - ||z_i - z_j||^2)

  L_contrastive = L_pos + L_neg
```

The SetFit library internally uses a temperature-scaled contrastive loss over multiple iterations:

```
L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))
```

where `τ` is temperature and `sim` is cosine similarity.

**Stage 2: Classification Head (Logistic Regression)**

After contrastive fine-tuning, a logistic regression head is trained on the tuned embeddings:

```
P(y=k|x) = softmax(W_k · φ(x) + b_k)

where:
  φ(x) = f_tuned(x)  # fine-tuned embedding
  W, b = learned via L2-regularized logistic regression
  Regularization: C = head_c (default 1.0)
  Solver: lbfgs
  Class weight: balanced
```

#### When to Use

| Condition | Recommendation |
|-----------|---------------|
| ≥ 3 examples per entity | **Use Full SetFit** |
| Need best accuracy with moderate data | **Use Full SetFit** |
| Production deployment | **Use Full SetFit** |

**Pros**: Best accuracy, learns domain-specific embeddings, robust to variations  
**Cons**: ~3 min training time, requires sufficient training data

---

### 1.2 SetFit Head-Only Training

**Implementation**: `src/novelentitymatcher/core/classifier.py`  
**Benchmark Performance**: 54.7% test accuracy on ag_news (500 training samples)

#### Mathematical Formulation

**Frozen Embeddings + PCA + Logistic Regression**

When `skip_body_training=True`, the SentenceTransformer body is frozen:

```
φ(x) = f_frozen(x)  # pre-trained embeddings (not fine-tuned)

# Optional PCA dimensionality reduction
X_pca = PCA(n_components = min(pca_dims, n_samples - 1, 384)) · X_embeddings

# Logistic Regression on reduced embeddings
P(y=k|x) = softmax(W_k · φ_pca(x) + b_k)
```

PCA is applied to reduce overfitting when training samples are limited.

#### When to Use

| Condition | Recommendation |
|-----------|---------------|
| < 3 examples per entity | **Use Head-Only** |
| Need fast training (~30s) | **Use Head-Only** |
| Quick iteration/exploration | **Use Head-Only** |

**Pros**: Fast training, works with minimal data, no GPU required  
**Cons**: Lower accuracy, embeddings not adapted to domain

---

### 1.3 Zero-Shot Classification

**Implementation**: `src/novelentitymatcher/core/matcher.py` → `EmbeddingMatcher`  
**Benchmark Performance**: 73.3% test accuracy on ag_news (no training)

#### Mathematical Formulation

**Pure Cosine Similarity**

```
For each known entity e with name/alias text:
  e_embedding = f_pretrained(e.text)
  q_embedding = f_pretrained(query)
  
  score(e) = cosine_similarity(q_embedding, e_embedding)
           = (q · e) / (||q|| · ||e||)

predicted_entity = argmax_e score(e)
confidence = max_e score(e)
```

#### When to Use

| Condition | Recommendation |
|-----------|---------------|
| No training data available | **Use Zero-Shot** |
| Entity names are semantically distinct | **Use Zero-Shot** |
| Prototyping/exploration | **Use Zero-Shot** |

**Pros**: No training required, instant setup, works out of the box  
**Cons**: Lower accuracy, cannot learn from data

---

### 1.4 BERT Classification

**Implementation**: `src/novelentitymatcher/core/bert_classifier.py`  
**Benchmark Performance**: Superior accuracy for complex patterns (100+ examples/entity)

#### Mathematical Formulation

**Transformer Fine-Tuning with Cross-Entropy**

```
h = BERT(x)  # transformer output
h_CLS = h[0]  # [CLS] token representation

P(y=k|x) = softmax(W · h_CLS + b)

L = -Σ_i log(P(y_i | x_i))  # cross-entropy loss
```

Supports mixed precision (fp16) for faster training on GPU.

#### When to Use

| Condition | Recommendation |
|-----------|---------------|
| ≥ 100 total examples, ≥ 8 per entity | **Use BERT** |
| High-stakes accuracy critical | **Use BERT** |
| Complex patterns (sarcasm, nuance) | **Use BERT** |
| GPU resources available | **Use BERT** |

**Pros**: Superior accuracy (often 3-5% better than SetFit), state-of-the-art architecture  
**Cons**: Slower training (~5 min), slower inference, higher compute cost

---

### 1.5 Hybrid Mode

**Implementation**: `src/novelentitymatcher/core/hybrid.py`  
**Use Case**: Large datasets (10k+ entities)

#### Pipeline Stages

```
Stage 1: BLOCKING (BM25/TF-IDF/Fuzzy)
  10,000 entities → 1,000 candidates

Stage 2: RETRIEVAL (Embedding Similarity)
  1,000 candidates → 50 candidates

Stage 3: RERANKING (Cross-Encoder)
  50 candidates → 5 final results
```

#### When to Use

| Condition | Recommendation |
|-----------|---------------|
| 10k+ entities | **Use Hybrid** |
| Need both speed and accuracy | **Use Hybrid** |
| Large-scale production | **Use Hybrid** |

---

### 1.6 Auto Mode Selection

The system automatically selects the optimal mode based on training data:

```
No training data → zero-shot
< 3 examples/entity → head-only
≥ 3 examples/entity, < 100 total → full
≥ 100 total, ≥ 8 examples/entity → bert
```

---

## Phase 2: Novelty Detection

Novelty detection determines whether a query belongs to an unknown (novel) class. The system uses a multi-signal approach where individual strategy scores are fused into a final novelty decision.

**Benchmark Reference**: See [novelty-detection-benchmark.md](../novelty-detection-benchmark.md) for comprehensive benchmark results.

### 2.1 Signal Fusion Architecture

```
Query Embedding + Predicted Class + Confidence
         │
    ┌────┴────────────────────────────────────────┐
    │                                             │
    ▼           ▼           ▼           ▼         ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│Confidence│ │ kNN    │ │ LOF    │ │Cluster │ │ ...    │
│Strategy  │ │Distance│ │Strategy│ │Strategy│ │        │
└────┬─────┘ └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
     │           │          │          │          │
     ▼           ▼          ▼          ▼          ▼
  s₁ ∈ [0,1]  s₂ ∈ [0,1]  s₃ ∈ [0,1]  s₄ ∈ [0,1]  ...
     │           │          │          │          │
     └───────────┴──────────┴──────────┴──────────┘
                              │
                    ┌─────────▼─────────┐
                    │   SignalCombiner   │
                    │   Method: weighted │
                    │   score = Σ(wᵢ·sᵢ)│
                    │        ─────────── │
                    │        Σ(wᵢ)       │
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Heuristic Gates  │
                    │   ALWAYS novel if: │
                    │   - uncertainty ≥ 0.85
                    │   - knn_score ≥ 0.85
                    │   - weighted ≥ 0.60
                    └─────────┬─────────┘
                              │
                    ┌─────────▼─────────┐
                    │   NOVEL / KNOWN    │
                    │   Decision         │
                    └───────────────────┘
```

### 2.2 Available Detection Strategies

#### SetFit Centroid Distance (RECOMMENDED)

**Benchmark**: AUROC 0.886, DR@1% 16.6% (ag_news)  
**Weight**: 0.45 (highest in ensemble)

**Mathematical Formulation**:

```
For each known class c:
  μ_c = (1/N_c) Σ_{x∈c} f_tuned(x)    # class centroid in SetFit embedding space

For query x:
  d(x) = min_c cosine_distance(f_tuned(x), μ_c)
  
  novelty_score = d(x)
  is_novel = d(x) > threshold
```

**Why it works**: SetFit's contrastive learning creates tight, well-separated class clusters. Novel samples fall outside these clusters and have high minimum distance to all centroids.

```
SetFit-Trained Embeddings:
  ●●●●●         ○○○○○
  ●●●●●         ○○○○○    ← Tight clusters, maximum separation
  ●●●●●         ○○○○○
  
  Class A       Class B
  
  ★ = novel sample (far from all centroids)
```

**When to Use**:
- Free text classification (best overall performance)
- When SetFit full training is used for Phase 1
- Production default for novelty detection

---

#### kNN Distance Strategy

**Benchmark**: AUROC 0.698 (raw embeddings), 0.866 (SetFit embeddings)  
**Weight**: 0.45 (highest in ensemble)

**Mathematical Formulation**:

```
For query x with embedding e_x:
  Find k nearest neighbors from reference set R:
    NN_k(x) = top-k most similar embeddings in R
  
  For each neighbor n_i ∈ NN_k(x):
    d_i = 1 - cosine_similarity(e_x, e_{n_i})
  
  novelty_score = mean(d_1, ..., d_k)
  is_novel = novelty_score ≥ distance_threshold
```

Uses ANN-backed search structures (HNSWlib/FAISS) for O(log n) search efficiency.

**When to Use**:
- Large-scale datasets (> 100K entities) with ANN indexing
- Complementary signal in ensemble
- When reference embeddings are available

---

#### Mahalanobis Distance Strategy

**Benchmark**: AUROC 0.474 (ag_news), 0.993 (go_emotions) — highly dataset dependent  
**Weight**: 0.35

**Mathematical Formulation**:

```
Pooled covariance matrix:
  Σ = Σ_c (X_c - μ_c)^T (X_c - μ_c) / (N - C)

Mahalanobis distance:
  D_M(x) = √((x - μ_c)^T Σ^{-1} (x - μ_c))

Novelty score:
  novelty_score = 1 - exp(-D_M(x) / threshold)
  is_novel = D_M(x) ≥ threshold
```

Regularization is applied to ensure invertibility: `Σ_reg = Σ + λI`

**When to Use**:
- Datasets with clear class-conditional Gaussian structure
- Many samples per class (needed for stable covariance estimation)
- When feature covariance is informative

**Avoid when**: Few samples per class (covariance matrix unstable)

---

#### Local Outlier Factor (LOF) Strategy

**Benchmark**: AUROC 0.648 (ag_news), 0.850 (go_emotions)  
**Weight**: 0.30

**Mathematical Formulation**:

```
Local reachability density:
  LRD(x) = |N_k(x)| / Σ_{y∈N_k(x)} reach-dist_k(x, y)

LOF score:
  LOF(x) = (1/|N_k(x)|) Σ_{y∈N_k(x)} LRD(y) / LRD(x)

novelty_score = LOF(x)
is_novel = LOF(x) < score_threshold
```

**When to Use**:
- Datasets with varying density across classes
- Density-based anomaly detection needed
- Complementary signal in ensemble

---

#### One-Class SVM Strategy

**Benchmark**: AUROC 0.682 (ag_news)  
**Weight**: 0.10

**Mathematical Formulation**:

```
Decision function:
  f(x) = Σ_i α_i K(x_i, x) - ρ

  where K is the RBF kernel: K(x, y) = exp(-γ ||x - y||^2)

novelty_score = clip(-f(x), 0, 1)
is_novel = f(x) < 0  (predicted as -1)
```

The `nu` parameter controls the expected fraction of outliers.

**When to Use**:
- Learning boundary of known class distribution
- Small to medium datasets (100 - 10,000 entities)
- No negative examples available

---

#### Confidence Strategy

**Benchmark**: AUROC 0.500 (essentially random)  
**Weight**: 0.35

**Mathematical Formulation**:

```
novelty_score = 1 - max(P(y|x))
is_novel = novelty_score > (1 - threshold)
```

**When to Use**:
- Always included as baseline signal in ensemble
- Quick heuristic check
- Not recommended as sole detection method

---

#### Uncertainty Strategy

**Weight**: 0.35

**Mathematical Formulation**:

```
Margin uncertainty:
  margin = P(y_1|x) - P(y_2|x)  # top-1 vs top-2 probability

Entropy uncertainty:
  H(x) = -Σ_k P(y_k|x) log P(y_k|x)
  H_normalized = H(x) / log(K)  # normalized to [0, 1]

is_novel = margin < margin_threshold OR H_normalized > entropy_threshold
```

**When to Use**:
- Capturing classifier confusion
- Complementary signal to distance-based methods

---

#### Clustering Strategy (HDBSCAN)

**Weight**: 0.20

**Mathematical Formulation**:

```
Apply HDBSCAN on combined reference + query embeddings:

For each query point:
  If labeled as noise (cluster = -1):
    is_novel = True
  
  If assigned to cluster C:
    cohesion = avg_pairwise_distance(members of C)
    support_score = 1 - cohesion
    is_novel = support_score < (1 - cohesion_threshold)
               OR cluster not validated
```

**When to Use**:
- Detecting novel clusters (not just outliers)
- Batch processing with many samples
- When novel samples form coherent groups

---

#### Pattern Strategy

**Benchmark**: AUROC 0.999 (entity-like text), 0.50 (free text)  
**Weight**: 0.20

**Mathematical Formulation**:

```
novelty_score = 0.25 × (1 - ngram_overlap)
              + 0.20 × (1 - 4gram_overlap)
              + 0.15 × length_deviation
              + 0.10 × capitalization_mismatch
              + 0.15 × prefix_rarity
              + 0.15 × suffix_rarity

is_novel = novelty_score > threshold
```

**When to Use**:
- Entity name matching (SKUs, product codes, structured entities)
- Orthographic/linguistic pattern violations
- NOT for free text classification

---

### 2.3 Signal Combination Methods

| Method | Formula | Use Case |
|--------|---------|----------|
| **weighted** | `score = Σ(wᵢ × sᵢ) / Σ(wᵢ)` | Default, best overall performance |
| **union** | `novel = any(flags)` | High recall needed (catch everything) |
| **intersection** | `novel = all(flags)` | High precision needed (few false positives) |
| **voting** | `novel = count(flags) > n/2` | Balanced, robust to individual strategy failures |

#### Heuristic Override Gates (Weighted Mode)

A sample is **ALWAYS** flagged as novel if ANY condition is met:

```
is_novel = True if:
  uncertainty_score ≥ 0.85    # Strong uncertainty
  OR knn_score ≥ 0.85         # Strong kNN distance
  OR knn_score ≥ 0.45         # kNN gate threshold
  OR weighted_score ≥ 0.60    # Final novelty threshold
```

---

## Decision Flow

```
                    ┌─────────────────────────────┐
                    │       Query Input            │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Classification (SetFit)   │
                    │   Mode: full                │
                    │   Accuracy: 91.2%           │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Predicted: "Sci/Tech"     │
                    │   Confidence: 0.85          │
                    │   Embedding: [384-dim]      │
                    └─────────────┬───────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              │                   │                   │
              ▼                   ▼                   ▼
    ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
    │ Centroid Dist   │ │   kNN Dist      │ │   Uncertainty   │
    │ Score: 0.72     │ │   Score: 0.68   │ │   Score: 0.15   │
    └────────┬────────┘ └────────┬────────┘ └────────┬────────┘
             │                   │                   │
             └───────────────────┼───────────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │   Weighted Fusion           │
                    │   score = 0.72 (HIGH)       │
                    │   Gate: knn ≥ 0.45 ✓        │
                    └────────────┬───────────────┘
                                 │
                    ┌────────────▼───────────────┐
                    │  Decision: NOVEL CLASS      │
                    │  - Clf confidence: 0.85     │
                    │  - Novelty score: 0.72      │
                    │  - Primary signal: centroid │
                    └─────────────────────────────┘
```

---

## Method Selection Guide

### Classification Mode Selection

```
Do you have training data?
│
├─ No → zero-shot (73.3% accuracy)
│
└─ Yes → How many examples per entity?
          │
          ├─ < 3 → head-only (54.7% accuracy, ~30s training)
          │
          ├─ ≥ 3, < 100 total → full (91.2% accuracy, ~3min training) ← RECOMMENDED
          │
          └─ ≥ 100 total, ≥ 8 per entity → bert (~95% accuracy, ~5min training)
```

**Special case**: 10k+ entities → hybrid mode (blocking → retrieval → reranking)

### Novelty Detection Strategy Selection

| Scenario | Recommended Strategy | Expected AUROC |
|----------|---------------------|----------------|
| Free text + SetFit trained | **SetFit Centroid Distance** | 0.886 |
| Entity name matching | **Pattern + kNN Distance** | 0.99+ |
| Large-scale (> 100K) | **kNN Distance (ANN)** | 0.87+ |
| High recall needed | **Union combination** | Catch all |
| High precision needed | **Intersection/Voting** | Few false positives |
| Production default | **Confidence + kNN + Clustering** | Balanced |
| Few-shot (< 50 samples) | **Prototypical + kNN** | Works with limited data |

---

## Benchmark Summary

### Classification Results (ag_news, 500 training samples)

| Mode | Train Accuracy | Test Accuracy | Training Time |
|------|---------------|---------------|---------------|
| Zero-shot | — | 73.3% | ~3s |
| Head-only | 53.1% | 54.7% | ~5s |
| **Full SetFit** | **89.8%** | **91.2%** | **~64s** |

### Novelty Detection Results (ag_news, 20% OOD)

#### SetFit-Based Methods (Using Contrastive Learning Embeddings)

| Strategy | Test AUROC | DR@1% | vs Traditional |
|----------|-----------|-------|----------------|
| **SetFit Centroid Distance** | **0.886** | **16.6%** | **+27% AUROC, +15x DR@1%** |
| SetFit Embedding KNN (k=20) | 0.866 | 1.4% | +24% AUROC |
| SetFit Prob Boundary | 0.860 | 6.9% | +23% AUROC |
| Hybrid Ensemble | 0.843 | 3.9% | +21% AUROC |

#### Traditional Methods (No Training Required)

| Strategy | Test AUROC | DR@1% |
|----------|-----------|-------|
| KNN Distance (k=20) | 0.698 | 1.1% |
| One-Class SVM (nu=0.1) | 0.682 | 2.3% |
| LOF (n=20) | 0.648 | 1.5% |
| Isolation Forest | 0.585 | 1.7% |
| Mahalanobis | 0.474 | 1.0% |

**Key Insight**: SetFit-tuned embeddings dramatically improve novelty detection because contrastive learning creates tight, well-separated class clusters. Novel samples fall outside these clusters, making them easier to detect.

---

## Implementation Reference

```python
from novelentitymatcher import Matcher, NovelEntityMatcher
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import ConfidenceConfig, KNNConfig

# Phase 1: Classification
matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data, num_epochs=4)

# Phase 2: Novelty Detection
novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    detection_config=DetectionConfig(
        strategies=["confidence", "knn_distance", "clustering"],
        confidence=ConfidenceConfig(threshold=0.65),
        knn_distance=KNNConfig(distance_threshold=0.45),
    ),
)

# Discover novel classes
report = await novel_matcher.discover_novel_classes(
    queries=["query1", "query2"],
    existing_classes=["class1", "class2"],
)
```

---

## Metric Definitions

### AUROC (Area Under ROC Curve)
- Measures overall discrimination ability between known and novel samples
- 1.0 = perfect discrimination, 0.5 = random guessing
- **SetFit Centroid achieves 0.886** vs traditional best of 0.721 (+23%)

### DR@1% (Detection Rate at 1% False Positive Rate)
- At 1% false positive rate (1% of known samples incorrectly flagged as novel), what percentage of novel samples are correctly detected?
- **SetFit Centroid achieves 16.6%** vs traditional best of 2.3% (+15x improvement)
- Critical metric for production systems where false alarms are costly

### AUPRC (Area Under Precision-Recall Curve)
- Better than AUROC for imbalanced datasets
- Measures precision-recall tradeoff across thresholds

---

*See [novelty-detection-benchmark.md](../novelty-detection-benchmark.md) for detailed benchmark methodology and configuration.*

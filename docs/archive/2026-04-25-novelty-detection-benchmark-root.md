# Novelty Detection Benchmark Results

**Date**: April 2026
**Model**: sentence-transformers/all-MiniLM-L6-v2
**Dataset**: ag_news (4 classes: World, Sports, Business, Sci/Tech)

---

## Executive Summary

This benchmark evaluates both **traditional novelty detection methods** and **novel SetFit-based methods** that leverage contrastive learning embeddings. The key finding is that using SetFit-trained embeddings for novelty detection significantly outperforms traditional approaches.

### Key Results

| Phase | Task | Best Method | Performance |
|-------|------|-------------|-------------|
| Phase 1 | Classification | Full SetFit | **91.2% accuracy** |
| Phase 2 | Novelty Detection | SetFit Centroid Distance | **0.886 AUROC, 16.6% DR@1%** |

---

## Phase 1: Classification Performance

**Task**: "Which KNOWN class does this text belong to?"

| Mode | Train Samples | Train Accuracy | Val Accuracy | Test Accuracy | Overfit Gap | Train Time |
|------|---------------|----------------|--------------|---------------|--------------|------------|
| Zero-shot | 0 | 75.8% | 72.0% | 73.3% | +2.5% | ~3s |
| Head-only | 500 | 59.4% | 53.1% | 54.7% | +4.7% | ~5s |
| **Full SetFit** | 500 | **98.0%** | **89.8%** | **91.2%** | +6.8% | ~64s |

### Findings

- **Zero-shot** works well when class names are semantically distinct (e.g., "Sports", "Business")
- **Head-only** underperforms due to limited training (only logistic regression head)
- **Full SetFit** achieves best accuracy by fine-tuning embeddings via contrastive learning

---

## Phase 2: Novelty Detection Performance

**Task**: "Does this text belong to an UNKNOWN (novel) class?"

### Traditional Methods (No Training Required)

| Strategy | Val AUROC | Test AUROC | DR@1% | DR@5% |
|----------|-----------|------------|--------|-------|
| KNN Distance (k=20) | 0.721 | 0.698 | 1.1% | 5.1% |
| One-Class SVM (nu=0.1) | 0.709 | 0.682 | **2.3%** | 5.8% |
| LOF (n=20) | 0.677 | 0.648 | 1.5% | 4.6% |
| Mahalanobis | 0.430 | 0.474 | 1.0% | 2.4% |
| Isolation Forest | 0.572 | 0.585 | 1.7% | 5.3% |

### NEW: SetFit-Based Methods (Using Contrastive Learning Embeddings)

| Strategy | Val AUROC | Test AUROC | DR@1% | DR@5% | vs Traditional |
|----------|-----------|------------|--------|-------|----------------|
| **SetFit Centroid Distance** | **0.885** | **0.886** | **16.6%** | 42.1% | **+27% AUROC, +15x DR@1%** |
| **SetFit Embedding KNN (k=20)** | 0.868 | 0.866 | 1.4% | 7.2% | +24% AUROC |
| **SetFit Prob Boundary** | 0.854 | 0.860 | **6.9%** | 24.8% | +23% AUROC, +6x DR@1% |
| Hybrid Ensemble | 0.837 | 0.843 | 3.9% | 15.2% | +21% AUROC |
| SetFit Mahalanobis | 0.209 | 0.217 | 0.1% | 0.4% | Failed |

---

## Why SetFit-Based Methods Work Better

### Traditional Embeddings (Frozen)

```
Pre-trained embeddings create loose, overlapping clusters:

    ● ● ●       ○ ○ ○
      ● ● ●   ○ ○
        ● ● ○ ○

    Class A       Class B    (overlapping, not well-separated)
```

### SetFit-Trained Embeddings (Contrastive Learning)

```
Contrastive learning creates tight, well-separated clusters:

    ●●●●●         ○○○○○
    ●●●●●         ○○○○○
    ●●●●●         ○○○○○

    Class A       Class B    (tight clusters, maximum separation)
```

**Novel samples fall OUTSIDE these tight clusters**, making them easier to detect.

---

## Metric Definitions

### AUROC (Area Under ROC Curve)
- Measures overall discrimination ability
- 1.0 = perfect, 0.5 = random
- **SetFit Centroid achieves 0.886** vs traditional best of 0.721 (+23%)

### DR@1% (Detection Rate at 1% False Positive)
- Measures detection capability when limiting false alarms
- At 1% of known samples incorrectly flagged as novel, what % of novel samples are caught?
- **SetFit Centroid achieves 16.6%** vs traditional best of 2.3% (+15x improvement)

---

## Method Descriptions

### SetFit Centroid Distance (RECOMMENDED)

```python
# Compute class centroids in trained embedding space
class_centroids = {}
for cls in known_classes:
    cls_embeddings = trained_model.encode(texts_of_class[cls])
    class_centroids[cls] = mean(cls_embeddings)

# Novelty = minimum distance to any class centroid
query_embedding = trained_model.encode(query_text)
novelty_score = min(distance(query_embedding, centroid) for centroid in class_centroids)
```

**Why it works**: Contrastive learning pushes samples of the same class together and different classes apart. Novel samples are far from all class centroids.

### SetFit Embedding KNN

```python
# Use trained embeddings for KNN novelty detection
query_embedding = trained_model.encode(query_text)
train_embeddings = trained_model.encode(known_texts)

# Novelty = 1 - average similarity to k-nearest known neighbors
similarities = cosine_similarity(query_embedding, train_embeddings)
novelty_score = 1 - mean(top_k_similarities)
```

**Why it works**: KNN in well-separated embedding space better identifies samples that don't belong to any known class cluster.

### SetFit Prob Boundary

```python
# Novelty = inverse of classification confidence
probabilities = trained_model.predict_proba(query_text)
max_probability = max(probabilities)
novelty_score = 1 - max_probability
```

**Why it works**: Novel samples receive low classification confidence because they don't match any known class well.

---

## Benchmark Configuration

```yaml
model: sentence-transformers/all-MiniLM-L6-v2
dataset: ag_news
ood_ratio: 0.2  # 20% of classes treated as unknown
train_samples: 500
val_samples: 1000
test_samples: 1000
known_classes: [Sports, Business, Sci/Tech]
ood_classes: [World]
```

---

## Production Recommendation

### Complete Workflow

```
Query Input
    │
    ├─► PHASE 1: CLASSIFICATION (SetFit Full Mode)
    │      │
    │      └─► Assign to known class with confidence
    │           Accuracy: 91.2%
    │
    └─► PHASE 2: NOVELTY DETECTION (SetFit Centroid Distance)
           │
           └─► Score novelty based on distance to class centroids
                AUROC: 0.886
                DR@1%: 16.6%
```

### Decision Flow

```
                    ┌─────────────────────────────┐
                    │       Query Input            │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Classification (SetFit)   │
                    │   Confidence: 0.85          │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │   Novelty Score (Centroid)    │
                    │   Score: 0.72 (HIGH)         │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Decision: NOVEL CLASS       │
                    │  - Low clf confidence        │
                    │  - High novelty score       │
                    └─────────────────────────────┘
```

### Implementation

```python
from novelentitymatcher.novelty import NovelEntityMatcher

# Initialize with both classification and novelty detection
matcher = NovelEntityMatcher(
    entities=[...],  # Known class definitions
    mode="full",    # Use SetFit for classification
    novelty_strategy="setfit_centroid",  # Use SetFit embeddings
)

# Fit on known data
matcher.fit(training_data)

# Match with novelty detection
result = matcher.match("Potential new class text")

# Check results
if result.is_novel and result.novel_score > 0.7:
    print(f"Potential novel class: {result.novel_score:.2f}")
```

---

## Benchmark Command

```bash
uv run scripts/benchmark_full_pipeline.py
```

Results are emitted as benchmark CSV artifacts in the project root (for example: `*_benchmark_results.csv` / `*_benchmark.csv`).

---

## Related Documentation

- [Novel Class Detection](./novel-class-detection.md) - Overview of novelty detection in the package
- [Architecture](./architecture.md) - System architecture
- [Matcher Modes](./matcher-modes.md) - Classification modes (zero-shot, head-only, full)
- [Quickstart](./quickstart.md) - Getting started guide

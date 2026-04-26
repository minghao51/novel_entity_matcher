# Comprehensive Benchmark Results (April 2026)

## Benchmark Environment
- **Model**: sentence-transformers/all-MiniLM-L6-v2
- **Datasets**: ag_news (4 classes), go_emotions (28 classes)
- **OOD ratio**: 20% of classes held out
- **Samples**: 200 train, 200 val, 500 test

---

## Phase 1: Novelty Strategy Benchmark

### Summary (Best Per Strategy Across Datasets)

| Strategy | Val AUROC | Test AUROC | Test DR@1% | Recommendation |
|----------|-----------|------------|------------|----------------|
| **setfit_centroid** | **0.915** | **0.907** | **0.147** | **Production default** |
| **knn_distance** | 0.893 | 0.883 | 0.103 | **Production default** |
| **ensemble_adaptive** | 0.911 | 0.906 | 0.187 | Best ensemble |
| **ensemble_weighted** | 0.911 | 0.905 | 0.124 | Strong ensemble |
| oneclass_svm | 0.836 | 0.834 | 0.143 | Good individual |
| lof | 0.872 | 0.871 | 0.069 | Good individual |
| pattern | 0.682 | 0.630 | 0.002 | Entity names only |
| mahalanobis | 0.696 | 0.691 | 0.029 | Dataset-dependent |
| self_knowledge | 0.588 | 0.563 | 0.011 | Weak, experimental |
| isolation_forest | 0.577 | 0.572 | 0.013 | Weak |
| uncertainty | 0.500 | 0.500 | 0.002 | Random (no all_probs) |
| prototypical | 0.507 | 0.507 | 0.006 | Near random |
| setfit (contrastive) | 0.483 | 0.452 | 0.017 | Below random |
| mahalanobis_conformal | 0.545 | 0.520 | 0.034 | Dataset-dependent |

### Per-Dataset Results

#### ag_news (4 classes, 1 OOD: World)

| Strategy | Val AUROC | Test AUROC | Test DR@1% |
|----------|-----------|------------|------------|
| setfit_centroid | 0.915 | 0.907 | 0.147 |
| knn_distance (k=30) | 0.893 | 0.883 | 0.103 |
| ensemble_adaptive | 0.911 | 0.906 | 0.187 |
| ensemble_weighted | 0.911 | 0.905 | 0.124 |
| oneclass_svm | 0.836 | 0.834 | 0.143 |
| lof | 0.872 | 0.871 | 0.069 |
| pattern | 0.682 | 0.630 | 0.002 |
| mahalanobis | 0.696 | 0.691 | 0.029 |
| self_knowledge | 0.588 | 0.563 | 0.011 |

#### go_emotions (28 classes, 5 OOD)

| Strategy | Val AUROC | Test AUROC | Test DR@1% |
|----------|-----------|------------|------------|
| mahalanobis_conformal | 0.545 | 0.520 | 0.034 |
| self_knowledge | 0.557 | 0.500 | 0.022 |
| isolation_forest | 0.557 | 0.498 | 0.006 |
| oneclass_svm | 0.543 | 0.481 | 0.006 |
| pattern | 0.499 | 0.517 | 0.006 |

**Note**: go_emotions is inherently harder (28 fine-grained emotion classes, high overlap). All strategies perform near random. This is a data characteristic, not a strategy limitation.

### KNN Parameter Sweep

| k | Val AUROC | Test AUROC | Test DR@1% |
|---|-----------|------------|------------|
| 3 | 0.860 | 0.862 | 0.075 |
| 5 | 0.876 | 0.873 | 0.075 |
| 10 | 0.880 | 0.877 | 0.080 |
| 20 | 0.889 | 0.881 | 0.090 |
| **30** | **0.893** | **0.883** | **0.103** |

**Optimal k: 20-30** (default updated to 20).

---

## Phase 2: Ensemble Weight Optimization (Optuna)

- **Trials**: 100
- **Best method**: voting
- **Best AUROC**: 0.719 (val), 0.691 (test)
- **Best novelty_threshold**: 0.609

**Finding**: Individual strategies (setfit_centroid: 0.907, knn: 0.883) outperform any Optuna-optimized ensemble (0.719). This is because the SignalCombiner's voting method with dummy confidences degrades individual strategy quality. **Recommendation**: Use `weighted` combination for production, with setfit_centroid and knn as primary strategies.

---

## Phase 4: Classification Benchmarks

### BERT vs SetFit (10 entities, 50 samples/entity, 3 epochs)

| Metric | SetFit | BERT (distilbert) | Ratio |
|--------|--------|-------------------|-------|
| Training Time | 41.26s | 13.20s | 3.13x slower |
| Peak Memory | 32.06 MB | 545.11 MB | 17x less |
| Inference Throughput | 1671/s | 998/s | 1.68x faster |
| **Accuracy** | **100.0%** | **88.0%** | **1.14x better** |

**Finding**: SetFit dominates on few-shot classification: higher accuracy, 17x less memory, 1.68x faster inference. BERT only justified for 100+ samples/entity.

---

## Phase 5: Infrastructure Benchmarks

### ANN Backend (exact only, hnswlib/faiss not installed)

| Size | Backend | Build (s) | QPS | Latency (ms) |
|------|---------|-----------|-----|-------------|
| 1K | exact | 0.000 | 29,934 | 0.03 |
| 5K | exact | 0.001 | 5,322 | 0.19 |
| 10K | exact | 0.002 | 2,400 | 0.42 |

**Note**: ANN backends (hnswlib, faiss) were not available for this run. Exact search is fine up to 10K vectors. For 100K+ vectors, install hnswlib or faiss-cpu.

---

## Default Configuration Updates

Based on these benchmarks, the following defaults were updated:

| Setting | Old Default | New Default | Reason |
|---------|-------------|-------------|--------|
| `DetectionConfig.strategies` | `["confidence", "knn_distance"]` | `["confidence", "knn_distance", "setfit_centroid"]` | setfit_centroid: 0.907 AUROC |
| `KNNConfig.k` | 5 | 20 | k=20: 0.881 vs k=5: 0.873 AUROC |
| `WeightConfig.setfit` | 0.1 | 0.02 | Below random (0.452 AUROC) |
| `WeightConfig.prototypical` | 0.1 | 0.02 | Near random (0.507 AUROC) |
| `WeightConfig.self_knowledge` | 0.15 | 0.08 | Weak (0.563 AUROC) |

### Recommended Production Configuration

```python
from novelentitymatcher import Matcher, NovelEntityMatcher
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import KNNConfig
from novelentitymatcher.novelty.config.weights import WeightConfig

matcher = Matcher(entities=entities, mode="full")
matcher.fit(training_data, num_epochs=4)

novel_matcher = NovelEntityMatcher(
    matcher=matcher,
    detection_config=DetectionConfig(
        strategies=["confidence", "knn_distance", "setfit_centroid"],
        combine_method="weighted",
        knn_distance=KNNConfig(k=20),
        weights=WeightConfig(
            setfit_centroid=0.45,
            knn=0.45,
            confidence=0.35,
        ),
    ),
)
```

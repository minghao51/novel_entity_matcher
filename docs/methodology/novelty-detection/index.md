# Novelty Detection Methods

Individual documentation for each novelty detection strategy in Phase 2 of the Novel Entity Matcher pipeline.

---

## Methods

| Strategy | Benchmark AUROC | DR@1% | Ensemble Weight | Best For |
|----------|----------------|-------|-----------------|----------|
| [SetFit Centroid Distance](setfit-centroid-distance.md) | **0.886** | **16.6%** | 0.45 | Production, free text |
| [kNN Distance](knn-distance.md) | 0.698 (raw) / 0.866 (SetFit) | 1.1% / 1.4% | 0.45 | Large-scale, ANN |
| [Mahalanobis Distance](mahalanobis-distance.md) | 0.474 (ag_news) / 0.993 (go_emotions) | 1.0% / 98.0% | 0.35 | Gaussian class structure |
| [LOF](lof.md) | 0.648 (ag_news) / 0.850 (go_emotions) | 1.5% / 12.8% | 0.15 | Varying density |
| [One-Class SVM](oneclass-svm.md) | 0.682 | 2.3% | 0.10 | Boundary learning |
| [Confidence](confidence.md) | 0.500 (random) | 0.2% | 0.35 | Baseline, always included |
| [Uncertainty](uncertainty.md) | ~0.500 | — | 0.35 | Classifier confusion |
| [Pattern](pattern.md) | 0.999 (entity) / 0.500 (free text) | 99.8% / — | 0.20 | Entity names |
| [Clustering (HDBSCAN)](clustering.md) | Dataset-dependent | — | 0.20 | Novel clusters, batch |
| [Prototypical](prototypical.md) | Dataset-dependent | — | 0.10 | Few-shot, interpretable |
| [SetFit Contrastive](setfit-contrastive.md) | Dataset-dependent | — | 0.10 | Few-shot, domain-specific |
| [Self-Knowledge](self-knowledge.md) | Dataset-dependent | — | 0.15 | Sparse autoencoder, internal |
| [Conformal Calibration](conformal-calibration.md) | — | — | — | Statistical calibration layer (wraps strategies) |

---

## Selection Guide

| Scenario | Recommended Strategy |
|----------|---------------------|
| Free text + SetFit trained | [SetFit Centroid Distance](setfit-centroid-distance.md) |
| Entity name matching | [Pattern](pattern.md) + [kNN Distance](knn-distance.md) |
| Large-scale (> 100K) | [kNN Distance](knn-distance.md) with ANN |
| High recall needed | Union combination method |
| High precision needed | Intersection/Voting combination |
| Production default | Confidence + kNN + Clustering |
| Few-shot (< 50 samples) | [Prototypical](prototypical.md) + [kNN Distance](knn-distance.md) |
| Statistical guarantees on false positive rate | [Mahalanobis](mahalanobis-distance.md) with [Conformal Calibration](conformal-calibration.md) |
| Learning embedding space structure | [Self-Knowledge](self-knowledge.md) (experimental) |

---

## Signal Fusion

Strategies are combined using one of five methods:

| Method | Formula | Use Case |
|--------|---------|----------|
| **weighted** | `score = Σ(wᵢ × sᵢ) / Σ(wᵢ)` | Default, best overall |
| **union** | `novel = any(flags)` | High recall |
| **intersection** | `novel = all(flags)` | High precision |
| **voting** | `novel = count(flags) > n/2` | Balanced |
| **meta_learner** | Logistic regression on strategy outputs | Learned combination (requires training data) |

---

## Each strategy page contains:

- **Mathematical Formulation**: Equations and formulas
- **DAG**: Inputs, components/techniques, outputs
- **Approach/Methodology**: Process and key design decisions
- **Configuration Options**: Parameters, defaults, and usage examples
- **Findings**: Benchmark performance, when to use, when not to use, characteristics, strengths, weaknesses

---

**Related**: [Classification Methods](../classification/) | [Methodology Overview](../overview.md)

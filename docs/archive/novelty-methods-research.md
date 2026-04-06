# Novel Entity Discovery Methods: Research & Implementation Guide

Archived note: this document is retained for historical research context. For the current supported novelty workflow, use [`../novel-class-detection.md`](../novel-class-detection.md).

## Overview

This document provides a comprehensive overview of novel entity discovery methods researched for integration into novel_entity_matcher's existing `NovelEntityMatcher` and `NoveltyDetector` system.

**Date**: 2026-03-19
**Status**: Research Complete, Proposal Only
**Scope**: Low and Medium Complexity Methods

This page is intentionally forward-looking. Unless a section explicitly says a method is already present in `novel_entity_matcher`, treat the implementation snippets and file layouts below as proposed work rather than current repository state.

---

## Table of Contents

1. [Current Implementation](#current-implementation)
2. [Research Findings](#research-findings)
3. [Method Comparisons](#method-comparisons)
4. [Implementation Details](#implementation-details)
5. [Performance Considerations](#performance-considerations)
6. [References](#references)

---

## Current Implementation

### Existing Components ✅

The novel_entity_matcher project already has a robust novelty detection system:

#### NovelEntityMatcher
- **Location**: `src/novelentitymatcher/novelty/entity_matcher.py`
- **Purpose**: Primary orchestration API for novelty-aware matching
- **Features**:
  - Multi-stage pipeline: matcher → detector → LLM proposer
  - Async/sync APIs for single and batch queries
  - Configurable detection strategies

#### NoveltyDetector
- **Location**: `src/novelentitymatcher/novelty/detector.py`
- **Purpose**: Multi-signal novelty detection
- **Current Strategies**:
  1. **CONFIDENCE**: Top-k uncertainty scoring (margin, entropy)
  2. **KNN_DISTANCE**: ANN-backed k-NN novelty scoring (HNSWlib/FAISS)
  3. **CLUSTERING**: HDBSCAN with cluster validation
- **Signal Fusion**: Weighted combination with configurable thresholds

#### Detection Strategies
- **Location**: `src/novelentitymatcher/novelty/schemas.py`
- **Available**:
  - `CONFIDENCE`: Low confidence detection
  - `KNN_DISTANCE`: Distance to known ANN neighbors
  - `CLUSTERING`: Density-based clustering
  - `CENTROID`: Legacy centroid distance (deprecated)

#### Current Detection Signals

**Uncertainty Metrics:**
- Margin score (top-1 vs top-2)
- Entropy score (normalized entropy)
- Confidence novelty (1 - top-1 confidence)
- Aggregate uncertainty score (weighted combination)

**kNN Novelty Metrics:**
- Mean distance to k-nearest known neighbors
- Nearest neighbor distance
- Predicted class neighbor ratio
- Predicted class support
- Aggregate kNN novelty score

**Cluster Validation Metrics:**
- Persistence (HDBSCAN cluster stability)
- Cohesion (within-cluster similarity)
- Separation (from known classes)
- Known support (overlap with predicted classes)

### What's Missing ❌

1. **No trainable discovery models**: Current system uses hand-crafted signals and thresholds
2. **No contrastive learning for novelty**: SetFit is used for classification, not novelty detection
3. **Limited few-shot learning**: No way to adapt to new domains with few examples
4. **No active learning**: Can't solicit feedback to improve novelty detection
5. **No temporal modeling**: Can't detect emerging trends over time
6. **Limited interpretability**: Hard to understand WHY something is novel

---

## Research Findings

### Low Complexity Methods

#### 1. Pattern-Based Novelty Detection

**Principle**: Novel entities often violate linguistic patterns of known entities.

**Approach**:
- Extract character-level and structural patterns from known entities
- Score novelty based on pattern violations
- No training required - purely rule-based

**Patterns to Extract**:
- Character n-grams (trigrams, 4-grams)
- Capitalization patterns (Title Case, UPPER, lower)
- Length ranges (min/max character count)
- Prefix/suffix distributions
- Special character presence (numbers, symbols, hyphens)
- Word boundaries for multi-word entities

**Scoring**:
```python
novelty_score = weighted_average([
    1 - ngram_overlap,           # Lower overlap = higher novelty
    length_violation,            # Outside normal length range
    capitalization_mismatch,     # Unusual capitalization
    prefix_rarity,               # Rare starting characters
    suffix_rarity,               # Rare ending characters
])
```

**Integration**:
- Add as `DetectionStrategy.PATTERN` to existing enum
- Implement `PatternBasedNoveltyStrategy` class
- Integrate into `NoveltyDetector._detect_by_pattern()`
- Add to weighted signal fusion

**Pros**:
- ✅ Fast, no model inference required
- ✅ Captures orthographic novelty that embeddings miss
- ✅ Works well for structured entity types (currencies, product codes)
- ✅ Easy to interpret and debug
- ✅ No training data required

**Cons**:
- ❌ May miss semantically novel but structurally similar entities
- ❌ Requires sufficient known entity samples for pattern extraction
- ❌ Language-dependent (patterns vary across languages)
- ❌ Less effective for free-form text

**Best For**:
- Structured entities (SKUs, IDs, codes)
- Multi-lingual scenarios where language patterns differ
- Complementary signal to embedding-based methods

**Implementation Effort**: 1-2 days

---

### Medium Complexity Methods

#### 2. One-Class Classification

**Principle**: Train a model to recognize only known entities; anything else is novel.

**Approaches**:
- **One-Class SVM**: Finds maximal margin boundary around known data
- **Isolation Forest**: Isolates anomalies by random splitting
- **Local Outlier Factor**: Density-based local anomaly detection

**One-Class SVM Details**:
- Uses RBF kernel to create non-linear boundary
- `nu` parameter controls expected outlier fraction
- Works directly with embedding features

**Training Process**:
```python
1. Encode known entities with sentence transformer
2. Fit OneClassSVM on embeddings
3. Calibrate threshold on validation set
4. Predict: -1 = novel, +1 = known
```

**Integration Strategy**:
- Add as `DetectionStrategy.ONE_CLASS_SVM`
- Implement `OneClassNoveltyDetector` class
- Use existing embeddings from matcher
- Add save/load functionality for trained models

**Pros**:
- ✅ Simple to implement
- ✅ Works with existing embeddings
- ✅ No negative examples required
- ✅ Fast training (seconds to minutes)
- ✅ sklearn-based, well-documented
- ✅ No GPU required

**Cons**:
- ❌ Sensitive to `nu` parameter (requires tuning)
- ❌ May not scale well to very large datasets (>100K samples)
- ❌ Kernel selection affects performance
- ❌ Less interpretable than distance-based methods

**Best For**:
- General-purpose novelty detection
- Small to medium datasets (100 - 10,000 entities)
- When training speed is important
- When GPU resources are limited

**Implementation Effort**: 2-3 days

---

#### 3. Prototypical Networks

**Principle**: Compute prototype (centroid) for each known class in embedding space. Novel entities are far from all prototypes.

**Approach**:
- Compute mean embedding for each class label
- Find nearest prototype for query
- Novel if distance to nearest prototype exceeds threshold

**Training Process**:
```python
1. Group training data by class label
2. Compute prototype (mean embedding) for each class
3. Compute pairwise distances to calibrate threshold
4. Use threshold for novelty detection
```

**Distance Metrics**:
- Cosine distance (default, works well with embeddings)
- Euclidean distance (alternative)
- Mahalanobis distance (accounts for covariance)

**Multi-Prototype Extension**:
- Use multiple prototypes per class for multi-modal distributions
- Cluster embeddings within each class
- Use cluster centroids as prototypes

**Integration Strategy**:
- Add as `DetectionStrategy.PROTOTYPICAL`
- Implement `PrototypicalNoveltyDetector` class
- Use existing embeddings from matcher
- Provide class information alongside novelty decision

**Pros**:
- ✅ Simple and interpretable
- ✅ Fast inference (O(num_classes))
- ✅ Works with existing embeddings
- ✅ Provides class information (nearest known class)
- ✅ No GPU required
- ✅ Fast "training" (just computing means)

**Cons**:
- ❌ Single prototype per class may not capture multi-modal distributions
- ❌ Threshold selection requires calibration
- ❌ Requires labeled training data (class labels)
- ❌ Less effective for high-dimensional embeddings without dimensionality reduction

**Best For**:
- Interpretable novelty detection
- When class information is valuable
- Multi-class scenarios with clear class structure
- Fast prototyping and experimentation

**Implementation Effort**: 2-3 days

---

#### 4. SetFit Contrastive Learning for Novelty

**Principle**: Learn embedding space where known entities cluster together and novel entities are pushed away. Similar to SetFit but optimized for novelty detection rather than classification.

**Approach**:
- Use SetFit's sentence transformer backbone
- Train with contrastive loss instead of classification loss
- Positive pairs: same entity with augmentations
- Negative pairs: different entities or known vs novel

**Training Process**:
```python
1. Create training pairs:
   - Positive: entity + augmented version
   - Negative: entity + different entity
   - Novel-negative: entity + synthetic novel
2. Train SetFit model with contrastive loss
3. Calibrate novelty threshold on known entities
4. Detect novelty by distance to known cluster
```

**Data Augmentations**:
- Lowercase/uppercase conversion
- Adding/removing spaces
- Typos (character substitutions)
- Synonym replacement (if applicable)
- Back-translation (for multi-lingual)

**Synthetic Novel Generation**:
- Random character substitutions
- Cross-language translations
- Adversarial examples
- Out-of-vocabulary words
- Mixed/mashed entities

**Integration Strategy**:
- Add as `DetectionStrategy.SETFIT_CONTRASTIVE`
- Implement `SetFitNoveltyDetector` class
- Leverage existing SetFit infrastructure
- Add synthetic data generation utilities
- Provide training/validation pipeline

**Pros**:
- ✅ Leverages existing SetFit infrastructure
- ✅ Few-shot learning (works with limited examples)
- ✅ Fast inference with sentence transformers
- ✅ Can be quickly trained and validated
- ✅ No GPU required for inference
- ✅ State-of-the-art performance for few-shot scenarios

**Cons**:
- ❌ Requires synthetic novel examples for best performance
- ❌ Need to calibrate threshold carefully
- ❌ Training slower than one-class methods
- ❌ Hyperparameter sensitive (margin, learning rate)
- ❌ May overfit to training augmentations

**Best For**:
- Few-shot scenarios (limited known entities)
- When SetFit is already in the stack
- Rapid prototyping and experimentation
- Domains with clear entity boundaries

**Implementation Effort**: 3-4 days

---

## Method Comparisons

### Complexity Matrix

| Method | Training Time | Inference Speed | Memory | GPU Required | Data Needed | Implementation Effort |
|--------|--------------|-----------------|--------|--------------|-------------|----------------------|
| **Pattern-Based** | None | Very Fast | Low | ❌ No | Low (50+ entities) | 1-2 days |
| **One-Class SVM** | Fast (seconds) | Fast | Medium | ❌ No | Medium (100+ entities) | 2-3 days |
| **Prototypical** | Very Fast | Very Fast | Low | ❌ No | Medium (100+ entities, labeled) | 2-3 days |
| **SetFit Contrastive** | Medium (minutes) | Fast | Medium | ❌ No | Low (few-shot, 10+ entities) | 3-4 days |

### Accuracy vs Speed Trade-off

```
High Accuracy, Slow:     SetFit Contrastive
High Accuracy, Fast:     Prototypical, One-Class SVM
Medium Accuracy, Fast:   Pattern-Based
```

### Use Case Recommendations

**For Quick Prototyping:**
1. Pattern-Based (no training)
2. Prototypical (fast, interpretable)

**For Production:**
1. One-Class SVM (robust, well-understood)
2. SetFit Contrastive (best accuracy, few-shot)

**For Resource-Constrained:**
1. Pattern-Based (no inference cost)
2. Prototypical (minimal computation)

**For Interpretable Results:**
1. Pattern-Based (rule-based)
2. Prototypical (class-based)

---

## Implementation Details

### File Structure

```
src/novelentitymatcher/novelty/
├── strategies/
│   ├── __init__.py
│   ├── base.py                    # Base strategy class
│   ├── pattern_strategy.py        # Pattern-based
│   ├── oneclass_strategy.py       # One-Class SVM
│   ├── prototypical_strategy.py   # Prototypical networks
│   └── setfit_novelty.py          # SetFit Contrastive
├── training.py                     # Training utilities
└── benchmark.py                    # Benchmarking tools

tests/
├── test_pattern_strategy.py
├── test_oneclass_strategy.py
├── test_prototypical_strategy.py
├── test_setfit_novelty.py
└── test_novelty_benchmark.py

examples/
├── pattern_strategy_example.py
├── oneclass_training_example.py
├── prototypical_training_example.py
├── setfit_novelty_training_example.py
└── novelty_benchmark_example.py
```

### Integration Points

#### 1. Update DetectionStrategy Enum

**File**: `src/novelentitymatcher/novelty/schemas.py`

```python
class DetectionStrategy(str, Enum):
    """Available detection strategies."""
    CONFIDENCE = "confidence"
    CENTROID = "centroid"
    KNN_DISTANCE = "knn_distance"
    CLUSTERING = "clustering"
    PATTERN = "pattern"              # New
    ONE_CLASS_SVM = "oneclass_svm"   # New
    PROTOTYPICAL = "prototypical"    # New
    SETFIT_CONTRASTIVE = "setfit"    # New
```

#### 2. Extend NoveltyDetector

**File**: `src/novelentitymatcher/novelty/detector.py`

```python
class NoveltyDetector:
    def __init__(self, config: DetectionConfig, embedding_dim: int = 768):
        # ... existing code ...

        # Initialize new strategies
        self.pattern_strategy = None
        self.oneclass_strategy = None
        self.prototypical_strategy = None
        self.setfit_strategy = None

    def detect_novel_samples(self, ...):
        # ... existing detection logic ...

        # Add new strategy detections
        if DetectionStrategy.PATTERN in self.config.strategies:
            pattern_flags = self._detect_by_pattern(texts)
            # Update signals

        if DetectionStrategy.ONE_CLASS_SVM in self.config.strategies:
            oneclass_flags = self._detect_by_oneclass(texts, embeddings)
            # Update signals

        if DetectionStrategy.PROTOTYPICAL in self.config.strategies:
            proto_flags = self._detect_by_prototypical(texts, embeddings, predicted_classes)
            # Update signals

        if DetectionStrategy.SETFIT_CONTRASTIVE in self.config.strategies:
            setfit_flags = self._detect_by_setfit(texts, embeddings)
            # Update signals

        # ... existing signal fusion ...
```

#### 3. Unified Training Interface

**File**: `src/novelentitymatcher/novelty/training.py`

```python
class UnifiedNoveltyTrainer:
    """Unified training interface for all trainable strategies."""

    def __init__(
        self,
        strategy: DetectionStrategy,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        **kwargs,
    ):
        self.strategy = strategy
        self.model_name = model_name
        self.kwargs = kwargs
        self.detector = None

    def train(
        self,
        training_data: List[dict],
        validation_data: Optional[List[dict]] = None,
        show_progress: bool = True,
    ) -> dict:
        """Train the selected novelty detector."""

        if self.strategy == DetectionStrategy.ONE_CLASS_SVM:
            self.detector = OneClassNoveltyDetector(
                model_name=self.model_name,
                **self.kwargs,
            )
            # ... training logic ...

        elif self.strategy == DetectionStrategy.PROTOTYPICAL:
            self.detector = PrototypicalNoveltyDetector(
                model_name=self.model_name,
                **self.kwargs,
            )
            # ... training logic ...

        elif self.strategy == DetectionStrategy.SETFIT_CONTRASTIVE:
            self.detector = SetFitNoveltyDetector(
                model_name=self.model_name,
                **self.kwargs,
            )
            # ... training logic ...

        # Validate and return metrics
        if validation_data:
            return self.evaluate(validation_data)
        return {}

    def evaluate(self, test_data: List[dict]) -> dict:
        """Evaluate on test data."""
        # ... evaluation logic ...
```

---

## Performance Considerations

### Training Performance

| Method | Training Time (1000 entities) | Scaling | Memory |
|--------|------------------------------|---------|--------|
| Pattern-Based | 0 ms (no training) | O(1) | Negligible |
| One-Class SVM | 1-5 seconds | O(n²) | Low |
| Prototypical | < 1 second | O(n) | Low |
| SetFit Contrastive | 1-5 minutes | O(n) | Medium |

### Inference Performance

| Method | Inference Time (per query) | Batch Processing | GPU Impact |
|--------|---------------------------|------------------|------------|
| Pattern-Based | < 1 ms | Excellent | None |
| One-Class SVM | 1-5 ms | Good | None |
| Prototypical | 1-3 ms | Excellent | None |
| SetFit Contrastive | 10-50 ms | Good | Optional |

### Memory Requirements

| Method | Model Size | Memory During Training | Memory During Inference |
|--------|-----------|----------------------|------------------------|
| Pattern-Based | < 1 MB | Negligible | < 1 MB |
| One-Class SVM | < 10 MB | Low | < 10 MB |
| Prototypical | < 5 MB | Low | < 5 MB |
| SetFit Contrastive | 100-500 MB | Medium | 100-500 MB |

### Scalability

**Pattern-Based**: Scales to millions of entities
**One-Class SVM**: Best for < 100K entities
**Prototypical**: Scales well, O(num_classes)
**SetFit Contrastive**: Scales well, batch processing recommended

---

## References

### Academic Papers

1. **SetFit**: "SetFit: Few-shot Learning with Sentence Embeddings" (Wang et al., 2022)
   - https://arxiv.org/abs/2209.11055

2. **One-Class SVM**: "Estimating the Support of a High-Dimensional Distribution" (Schölkopf et al., 2001)
   - https://doi.org/10.1162/089976601750399415

3. **Prototypical Networks**: "Prototypical Networks for Few-shot Learning" (Snell et al., 2017)
   - https://arxiv.org/abs/1703.05175

4. **Contrastive Learning**: "A Simple Framework for Contrastive Learning of Visual Representations" (Chen et al., 2020)
   - https://arxiv.org/abs/2002.05709

### Libraries & Tools

- **SetFit**: https://github.com/huggingface/setfit
- **scikit-learn**: https://scikit-learn.org/stable/modules/svm.html#one-class-svm
- **sentence-transformers**: https://www.sbert.net
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers

### Related Work in novel_entity_matcher

- `docs/novel-class-detection.md` - Current novelty detection system
- `src/novelentitymatcher/novelty/` - Existing implementation
- `examples/novel_discovery_example.py` - Usage examples

---

## Next Steps

### Immediate (Week 1)
1. Implement Pattern-Based Strategy
2. Create benchmark dataset
3. Document integration process

### Short-Term (Weeks 2-3)
4. Implement One-Class SVM
5. Implement Prototypical Networks
6. Implement SetFit Contrastive
7. Create unified training interface

### Medium-Term (Weeks 4-5)
8. Performance benchmarking
9. A/B testing framework
10. Production integration
11. Documentation updates

---

## Appendix: Discussion Summary

### Key Decisions

1. **Scope Limitation**: Focus on low and medium complexity methods only
   - Rationale: Faster implementation, easier maintenance, sufficient for most use cases
   - Excluded: Deep learning methods (autoencoders, deep SVDD, OpenMax)

2. **Leverage Existing Infrastructure**:
   - Use existing sentence transformers and SetFit
   - Integrate with current NoveltyDetector architecture
   - Reuse existing embedding computation

3. **Progressive Enhancement**:
   - Start with pattern-based (no training)
   - Add trainable methods incrementally
   - Allow users to choose based on their needs

### Trade-offs Considered

**Accuracy vs Complexity**:
- Chose methods with good accuracy and low complexity
- Deep learning methods excluded due to complexity and resource requirements

**Training Speed vs Performance**:
- Prioritized fast training for rapid iteration
- SetFit offers best performance with acceptable training time

**Interpretability vs Automation**:
- Pattern-based: Most interpretable
- One-Class SVM: Less interpretable but robust
- Prototypical: Good balance (class-based interpretation)

### Alternative Approaches Not Taken

1. **Deep Autoencoders**: High complexity, requires GPU, longer training
2. **Deep SVDD**: Complex implementation, requires careful tuning
3. **OpenMax**: Theoretically sound but complex, unstable Weibull fitting
4. **Isolation Forest**: Good alternative to One-Class SVM, but less common
5. **Local Outlier Factor**: Computationally expensive for large datasets

### Future Enhancements

Potential future additions if needed:
1. Ensemble methods combining multiple strategies
2. Active learning for continuous improvement
3. Temporal modeling for trend detection
4. Multi-modal detection (text + metadata)
5. Hierarchical novelty detection (category-level novelty)

---

*Last Updated: 2026-03-19*
*Author: Novel Entity Matcher Research Notes*
*Version: 1.0*

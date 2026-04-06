# Confidence Strategy

**Phase 2: Novelty Detection** | **Classifier-based** | **Benchmark: 0.500 AUROC (random)**

---

## Mathematical Formulation

### Top-1 Confidence

```
P(y|x) = classifier.predict_proba(x)  # Probability distribution over classes

confidence = max_k P(y=k|x)  # Top-1 probability

novelty_score = 1 - confidence

is_novel = novelty_score > (1 - threshold)
         = confidence < threshold
```

### Default Threshold

```
threshold = 0.7  (default)

is_novel = confidence < 0.7
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Query texts: List[str]                                    │
│  • Classifier: SetFitClassifier / BERTClassifier             │
│  • threshold: float (default: 0.7)                           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: CLASSIFICATION                         │
├─────────────────────────────────────────────────────────────┤
│  Technique: Classifier predict_proba                         │
│  • For each query x:                                         │
│    probs = classifier.predict_proba(x)                       │
│    confidence = max(probs)                                   │
│    predicted_class = argmax(probs)                           │
│  • Output: (confidence, predicted_class, probs) per query    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: NOVELTY SCORING                        │
├─────────────────────────────────────────────────────────────┤
│  Technique: Inverse Confidence                               │
│  • novelty_score = 1 - confidence                            │
│  • is_novel = confidence < threshold                         │
│  • Output: (novelty_score, is_novel, confidence)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • confidence: float (classifier top-1 probability)          │
│  • predicted_class: str                                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Classify query**: Get probability distribution from trained classifier
2. **Extract confidence**: Take maximum probability as confidence score
3. **Score novelty**: Inverse of confidence (low confidence = high novelty)
4. **Threshold**: Flag as novel if confidence falls below threshold

### Key Design Decisions

- **Zero overhead**: Uses classifier output that's already computed
- **Always available**: Works with any classification mode
- **Simple**: Single threshold, no additional model or computation
- **Baseline signal**: Included in all ensemble configurations by default

### Implementation Details

- **Weight in ensemble**: 0.35
- **No additional model**: Reuses classifier's predict_proba output
- **Always included**: Part of default strategy set

---

## Findings

### Benchmark Performance

**ag_news:**

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.500** (random) |
| **DR@1%** | 0.2% |
| **DR@5%** | 0.2% |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Baseline signal in ensemble | **Always include** |
| Quick heuristic check | **Use Confidence** |
| Zero-overhead novelty signal | **Use Confidence** |
| Part of default strategy set | **Use Confidence** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Standalone novelty detection | Use **SetFit Centroid Distance** (0.886 vs 0.500) |
| Sole detection method | **Never use alone** — essentially random |
| High-accuracy novelty detection needed | Use **kNN Distance** or **SetFit Centroid Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | None (uses existing classifier) |
| **Inference time** | O(1) (just read classifier output) |
| **Memory** | None additional |
| **GPU required** | No |
| **Data requirement** | None (uses trained classifier) |
| **Standalone performance** | Random (AUROC 0.500) |

### Strengths

- **Zero overhead**: No additional computation beyond classification
- **Always available**: Works with any classifier
- **Simple to interpret**: Low confidence = uncertain prediction
- **Good ensemble member**: Contributes signal when combined with other methods
- **Default inclusion**: Part of every ensemble configuration

### Weaknesses

- **Random performance standalone**: AUROC 0.500 — no better than coin flip
- **Classifier overconfidence**: Modern classifiers can be confidently wrong
- **Not calibrated**: Softmax probabilities are often poorly calibrated
- **Misses embedding-space novelty**: A sample can be far from all known classes but still receive high classifier confidence
- **Never use alone**: Must be combined with distance-based or density-based methods

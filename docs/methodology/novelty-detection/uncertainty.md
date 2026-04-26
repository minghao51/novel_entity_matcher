# Uncertainty Strategy

**Phase 2: Novelty Detection** | **Classifier uncertainty** | **Weight: 0.35 in ensemble**

---

## Mathematical Formulation

### Margin Uncertainty

```
Sort class probabilities in descending order:
  P_1 ≥ P_2 ≥ ... ≥ P_K

margin = P_1 - P_2    # Gap between top-1 and top-2

margin_novelty = 1 - margin    # Invert: low margin = high uncertainty

is_novel_by_margin = margin < margin_threshold
```

### Entropy Uncertainty

```
Shannon entropy:
  H(x) = -Σ_{k=1}^{K} P_k · log(P_k)

Maximum entropy (uniform distribution):
  H_max = log(K)

Normalized entropy:
  H_norm(x) = H(x) / H_max    # Normalized to [0, 1]

is_novel_by_entropy = H_norm(x) > entropy_threshold
```

### Combined Uncertainty

```
is_novel = is_novel_by_margin OR is_novel_by_entropy

novelty_score = weighted_average([
  margin_novelty,
  H_norm
])
```

### Default Thresholds

```
margin_threshold = 0.3      # Flag if top-1 and top-2 are within 0.3
entropy_threshold = 1.5     # Flag if normalized entropy > 1.5
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Query texts: List[str]                                    │
│  • Classifier: SetFitClassifier / BERTClassifier             │
│  • margin_threshold: float (default: 0.3)                    │
│  • entropy_threshold: float (default: 1.5)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: PROBABILITY EXTRACTION                 │
├─────────────────────────────────────────────────────────────┤
│  Technique: Classifier predict_proba                         │
│  • For each query x:                                         │
│    probs = classifier.predict_proba(x)                       │
│    P = [P_1, P_2, ..., P_K] sorted descending               │
│  • Output: Probability distributions per query               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: MARGIN COMPUTATION                     │
├─────────────────────────────────────────────────────────────┤
│  Technique: Top-1 vs Top-2 Margin                            │
│  • margin = P[0] - P[1]                                     │
│  • margin_novelty = 1 - margin                               │
│  • is_novel_by_margin = margin < margin_threshold            │
│  • Output: margin, margin_novelty, flag                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: ENTROPY COMPUTATION                    │
├─────────────────────────────────────────────────────────────┤
│  Technique: Shannon Entropy                                  │
│  • H(x) = -Σ P_k · log(P_k)                                 │
│  • H_norm = H(x) / log(K)                                   │
│  • is_novel_by_entropy = H_norm > entropy_threshold          │
│  • Output: entropy, normalized_entropy, flag                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: COMBINED DECISION                      │
├─────────────────────────────────────────────────────────────┤
│  Technique: OR Logic                                         │
│  • is_novel = is_novel_by_margin OR is_novel_by_entropy      │
│  • novelty_score = weighted_average of both signals          │
│  • Output: (novelty_score, is_novel, margin, entropy)        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • margin: float (P_1 - P_2)                                 │
│  • entropy: float (normalized Shannon entropy)               │
│  • margin_flag: bool                                         │
│  • entropy_flag: bool                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Extract probabilities**: Get full probability distribution from classifier
2. **Compute margin**: Gap between top-1 and top-2 predictions
3. **Compute entropy**: Measure of probability distribution uniformity
4. **Combine signals**: Flag as novel if either margin or entropy indicates uncertainty

### Key Design Decisions

- **Two complementary signals**: Margin captures ambiguity between top classes, entropy captures overall confusion
- **OR logic**: Either signal can trigger novelty flag (high recall)
- **Normalized entropy**: Scales to [0, 1] regardless of number of classes

### Implementation Details

- **Weight in ensemble**: 0.35
- **Requires**: Full probability distributions (not just top-1 prediction)
- **Heuristic gate**: uncertainty_score ≥ 0.85 → ALWAYS novel

---

## Configuration Options

Options are set via `UncertaintyConfig`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `margin_threshold` | float | 0.3 | [0.0, 1.0] | Margin between top-1 and top-2 predictions. Small margin = high uncertainty → novel |
| `entropy_threshold` | float | 1.5 | ≥ 0.0 | Entropy threshold. High entropy = classifier confusion → novel |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import UncertaintyConfig

config = DetectionConfig(
    strategies=["uncertainty"],
    uncertainty=UncertaintyConfig(
        margin_threshold=0.25,
        entropy_threshold=1.2,
    ),
)
```

---

## Findings

### Benchmark Performance

| Metric | Value |
|--------|-------|
| **Standalone AUROC** | ~0.500 (similar to confidence) |
| **Ensemble weight** | 0.35 |
| **Heuristic gate** | uncertainty_score ≥ 0.85 → always novel |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Capturing classifier confusion | **Use Uncertainty** |
| Complementary signal to distance-based methods | **Use Uncertainty** |
| Detecting ambiguous queries (close between classes) | **Use Uncertainty** |
| Part of ensemble | **Use Uncertainty** (weight: 0.35) |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Standalone detection | Use **SetFit Centroid Distance** |
| Well-separated classes | Margin will always be high, low signal |
| Single-class scenarios | Entropy not meaningful |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | None (uses existing classifier) |
| **Inference time** | O(K) per query (K = number of classes) |
| **Memory** | None additional |
| **GPU required** | No |
| **Data requirement** | None (uses trained classifier) |

### Strengths

- **Captures classifier confusion**: Detects when classifier is uncertain between classes
- **Two complementary signals**: Margin and entropy capture different types of uncertainty
- **Zero additional model cost**: Uses classifier output already computed
- **Good ensemble member**: Contributes different signal than distance-based methods
- **Heuristic gate**: Strong uncertainty (≥ 0.85) always triggers novel flag

### Weaknesses

- **Poor standalone performance**: Similar to confidence (~0.500 AUROC)
- **Classifier-dependent**: Only as good as the underlying classifier's calibration
- **Misses embedding-space novelty**: Cannot detect samples far from all classes if classifier is confident
- **Sensitive to number of classes**: Entropy threshold needs adjustment for different K

# Conformal Calibration

**Phase 2: Novelty Detection** | **Statistical calibration layer** | **Wraps existing strategies with p-values**

---

## Mathematical Formulation

### Nonconformity Scores

For a calibration set of known samples with raw OOD scores:

```
nonconformity_scores = sort(raw_scores)  # Ascending order

Higher raw scores → more anomalous → more novel
```

### p-Value Computation

For a test sample with raw score s:

```
p-value(s) = (|{i : calibration_score_i ≥ s}| + 1) / (n_calibration + 1)

Interpretation:
  p-value ≈ 0  → very novel (no calibration sample scored this high)
  p-value ≈ 1  → not novel (many calibration samples scored higher)
  is_novel = p-value < α  (significance level)
```

The `+1` terms provide finite-sample coverage guarantee: P(p-value < α) ≤ α for any distribution.

### Split Conformal Method

```
1. Split reference data into training (80%) and calibration (20%) sets
2. Compute raw OOD scores for calibration set
3. Store sorted calibration scores
4. For each test sample:
     p-value = fraction of calibration scores ≥ test score
```

### Mondrian (Class-Conditional) Conformal Method

```
1. Group calibration scores by class label
2. For each class c, store sorted calibration scores: S_c
3. For test sample with predicted class ĉ:
     p-value = fraction of S_{ĉ} scores ≥ test score
4. Fallback: if class has no calibration data, use global calibration scores

Advantage: Maintains coverage per class, not just globally
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Raw OOD scores from any strategy (e.g., Mahalanobis)      │
│  • Calibration labels: List[str] (for Mondrian method)       │
│  • alpha: float (default: 0.1) — significance level          │
│  • method: "split" or "mondrian"                             │
│  • calibration_set_fraction: float (default: 0.2)           │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: CALIBRATION                             │
├─────────────────────────────────────────────────────────────┤
│  Technique: Conformal Score Distribution                      │
│  • Collect raw OOD scores for calibration samples             │
│  • Split method: sort all calibration scores                  │
│  • Mondrian method: group and sort by class                   │
│  • Output: calibration score distributions                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: P-VALUE COMPUTATION                     │
├─────────────────────────────────────────────────────────────┤
│  Technique: Rank-based p-value                                │
│  • For each test score s:                                     │
│    - Find position in sorted calibration scores               │
│    - p-value = (n_above + 1) / (n_total + 1)                 │
│  • Mondrian: use class-specific calibration set               │
│  • Output: p-values for all test samples                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • p-values: float array ∈ [0, 1]                            │
│  • is_novel: bool array (p-value < alpha)                    │
│  • calibration_metadata: dict (alpha, method, n_calibration)  │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Calibrate**: Compute nonconformity scores from a held-out calibration set
2. **Predict**: Convert raw strategy scores to statistically grounded p-values
3. **Decide**: Flag samples with p-values below significance level α as novel

### Key Design Decisions

- **Distribution-free**: No assumptions about the underlying score distribution
- **Finite-sample guarantee**: P(false positive) ≤ α for any sample size
- **Two methods**: Split (simpler) and Mondrian (per-class, more precise)
- **Wraps existing strategies**: Can be applied to any strategy that produces raw OOD scores

### Implementation Details

- **Currently integrated with**: Mahalanobis distance strategy via `MahalanobisConfig.calibration_mode`
- **ConformalCalibrator class**: `src/novelentitymatcher/novelty/strategies/conformal.py`
- **Can be used standalone**: Accepts any array of raw scores and labels

---

## Configuration Options

Conformal calibration is configured as part of the Mahalanobis strategy:

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `calibration_mode` | str | "none" | "none", "conformal" | Enable/disable conformal calibration |
| `calibration_alpha` | float | 0.1 | (0.0, 1.0] | Significance level. Lower = stricter (fewer false positives) |
| `calibration_method` | str | "split" | "split", "mondrian" | Calibration method. Mondrian provides per-class guarantees |
| `calibration_set_fraction` | float | 0.2 | (0.0, 0.5] | Fraction of reference data held out for calibration |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import MahalanobisConfig

config = DetectionConfig(
    strategies=["mahalanobis"],
    mahalanobis=MahalanobisConfig(
        threshold=3.0,
        calibration_mode="conformal",
        calibration_alpha=0.05,
        calibration_method="mondrian",
        calibration_set_fraction=0.2,
    ),
)
```

---

## Findings

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Need statistical guarantees on false positive rate | **Use Conformal Calibration** |
| Mahalanobis distance with well-calibrated thresholds | **Use Mondrian conformal** |
| Per-class coverage guarantees | **Use Mondrian method** |
| Simple global calibration | **Use Split method** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Small calibration set (< 20 samples) | Raw threshold (too few scores for reliable p-values) |
| Strategies with well-known thresholds | Use native thresholding |
| Need maximum detection power | Raw scores may be more discriminative than p-values |

### Characteristics

| Property | Value |
|----------|-------|
| **Calibration cost** | O(n log n) for sorting calibration scores |
| **Inference cost** | O(log n) per sample (binary search in sorted scores) |
| **Memory** | O(n) for calibration score storage |
| **Assumptions** | Distribution-free (exchangeability) |
| **Coverage guarantee** | P(false positive) ≤ α |

### Strengths

- **Statistical guarantees**: Finite-sample coverage, no distributional assumptions
- **Interpretable p-values**: Directly interpretable as probability of being in-distribution
- **Flexible**: Wraps any strategy producing raw scores
- **Two methods**: Split (simple) and Mondrian (per-class)

### Weaknesses

- **Reduces reference data**: Calibration set held out from training
- **Score discretization**: With few calibration samples, p-values are coarse
- **Exchangeability assumption**: Requires calibration and test data to be exchangeable
- **Power loss**: Converting to p-values may lose some discriminative power

---

**Related**: [Mahalanobis Distance](mahalanobis-distance.md) | [Novelty Detection Methods](index.md) | [Methodology Overview](../overview.md)

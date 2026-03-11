# Country Code Classifier Scripts

Related docs: [`index.md`](./index.md) | [`../quickstart.md`](../quickstart.md) | [`./index.md`](./index.md)

This document explains the different Python scripts in `experiments/country_classifier/` and their differences.

## Overview

| Script | Purpose | Complexity |
|--------|---------|------------|
| `experiments/country_classifier/country_classifier.py` | Baseline comparison (A/B/C scenarios) | Basic |
| `experiments/country_classifier/country_classifier_quick.py` | Quick optimization tests | Intermediate |
| `experiments/country_classifier/country_classifier_advanced.py` | Full optimization exploration | Advanced |

---

## Scripts Comparison

### 1. country_classifier.py (Baseline)

**Purpose:** Compare three fundamental training approaches.

**Scenarios:**
- **Scenario A - Zero-shot:** No training. Uses embedding similarity between input text and label names.
- **Scenario B - Head-only:** Freezes embeddings, trains only the classification head (2 epochs).
- **Scenario C - Full training:** Fine-tunes both embeddings and classifier head (4 epochs).

**Key Features:**
- Uses `SetFitClassifier` from `semanticmatcher`
- Simple, readable code
- Good for understanding the fundamentals

**Usage:**
```bash
uv run python experiments/country_classifier/country_classifier.py
```

---

### 2. country_classifier_quick.py (Quick Optimization)

**Purpose:** Quick experiments to improve beyond baseline 90.91% accuracy.

**Tests:**
- Baseline (4 epochs, mpnet)
- 6 epochs with mpnet
- 4 epochs with all-mpnet (stronger embeddings)
- LinearSVC classifier head
- all-mpnet + 6 epochs

**Key Features:**
- Simplified, streamlined code
- Faster iteration
- Tests most impactful optimizations

**Usage:**
```bash
uv run python experiments/country_classifier/country_classifier_quick.py
```

**Findings:**
- 6 epochs actually decreased accuracy (90.91% → 87.88%)
- Baseline configuration is already optimal

---

### 3. country_classifier_advanced.py (Advanced Optimization)

**Purpose:** Comprehensive optimization exploration.

**Scenarios:**
- **D) More epochs:** Tests 6, 8 epochs with tuned learning rates
- **E) Better embeddings:** Tests BGE-small, BGE-base, all-mpnet-base-v2
- **F) Different classifier heads:** Tests LinearSVC, LogisticRegression with various C values
- **G) Ensemble:** Majority voting from top models

**Key Features:**
- Most comprehensive
- Custom classifier testing head support
- Ensemble methodology
- Longer runtime (~15 min budget)

**Usage:**
```bash
uv run python experiments/country_classifier/country_classifier_advanced.py
```

---

## Pipeline Overview

```
                    ┌─────────────────────┐
                    │  Training Data CSV   │
                    │ (country_training_  │
                    │    data.csv)        │
                    └──────────┬──────────┘
                               │
                               ▼
              ┌────────────────────────────────┐
              │     Train/Test Split (80/20)   │
              └───────────────┬────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │Scenario│          │Scenario│          │Scenario│
   │   A    │          │   B    │          │   C    │
   │Zero-   │          │Head-   │          │Full    │
   │shot    │          │only    │          │Training│
   └────┬────┘          └────┬────┘          └────┬────┘
        │                     │                     │
        ▼                     ▼                     ▼
   Embedding             Fine-tuned          Fine-tuned
   similarity           classifier head      embeddings +
   only                 only (2 epochs)     classifier (4 epochs)
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌─────────────────────┐
                    │   Test Evaluation   │
                    │   (33 samples)      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Accuracy Results  │
                    └─────────────────────┘
```

---

## Key Differences

| Aspect | A | B | C | Quick | Advanced |
|--------|---|---|---|-------|----------|
| Training time | ~0s | ~30s | ~3min | ~15min | ~15min |
| GPU required | No | Yes | Yes | Yes | Yes |
| Embedding fine-tuning | No | No | Yes | Yes | Yes |
| Custom heads | N/A | No | No | Yes | Yes |
| Epochs tested | N/A | 2 | 4 | 4, 6 | 4, 6, 8 |
| Embeddings tested | N/A | mpnet | mpnet | mpnet, all-mpnet | mpnet, BGE |

---

## Results Summary

| Scenario | Accuracy |
|----------|----------|
| A) Zero-shot | 21.21% |
| B) Head-only (2 epochs) | 87.88% |
| C) Full training (4 epochs) | **90.91%** |

**Best Configuration:** Full training with `paraphrase-mpnet-base-v2` for 4 epochs = **90.91%**

---

## Data Format

Training data is stored in `data/country_training_data.csv`:

```csv
text,label
United States,US
USA,US
United Kingdom,GB
UK,GB
...
```

- **121 samples** total
- **33 unique country codes**
- **88 training / 33 test** (80/20 split)

---

## Recommendation

For new users, start with `country_classifier.py` to understand the fundamentals, then use `country_classifier_quick.py` for quick optimizations.

For the current experiment inventory, see [`index.md`](./index.md).

# Benchmark Results

Related docs: [`benchmarking.md`](./benchmarking.md) | [`speed-benchmark-results.md`](./speed-benchmark-results.md) | [`novelty-detection-benchmark.md`](./novelty-detection-benchmark.md)

## Latest Results: Zero-Shot vs Head-Only vs Full SetFit (April 2, 2026)

**Model:** sentence-transformers/all-MiniLM-L6-v2
**Key fix:** Probability reordering bug + PCA dimensionality reduction + stratified sampling

### ag_news (4 classes)

| Mode | Train Samples | Train Acc | Test Acc | Gap | vs Zero-Shot |
|------|--------------|-----------|----------|-----|--------------|
| zero-shot | - | 67.2% | **64.2%** | +3.0% | baseline |
| head-only-100 | 100 | 81.0% | 78.7% | +2.3% | **+14.5%** |
| head-only-200 | 200 | 81.5% | 79.6% | +1.9% | **+15.4%** |
| head-only-500 | 375 | 84.8% | 83.1% | +1.7% | **+18.9%** |
| full-100 | 100 | 93.0% | 85.2% | +7.8% | **+21.0%** |
| full-200 | 200 | 93.0% | 86.1% | +6.9% | **+21.9%** |
| full-500 | 375 | 94.1% | **87.2%** | +6.9% | **+23.0%** |

### yahoo_answers_topics (10 classes)

| Mode | Train Samples | Train Acc | Test Acc | Gap | vs Zero-Shot |
|------|--------------|-----------|----------|-----|--------------|
| zero-shot | - | 37.3% | **37.3%** | +0.0% | baseline |
| head-only-100 | 100 | 97.0% | 46.5% | +50.5% | **+9.2%** |
| head-only-200 | 200 | 54.5% | 37.6% | +16.9% | +0.3% |
| head-only-500 | 375 | 58.9% | 49.3% | +9.6% | **+12.0%** |
| full-100 | 100 | 94.0% | 56.0% | +38.0% | **+18.7%** |
| full-200 | 200 | 82.5% | 56.4% | +26.1% | **+19.1%** |
| full-500 | 375 | 77.6% | **60.3%** | +17.3% | **+23.0%** |

### Key Findings

1. **Full SetFit (contrastive + head) consistently beats head-only** on both datasets
2. **Full SetFit with 500 samples** achieves the best results:
   - ag_news: 87.2% test accuracy (+23.0% over zero-shot)
   - yahoo_answers: 60.3% test accuracy (+23.0% over zero-shot)
3. **Head-only with PCA** works well for ag_news (83.1% with 375 samples) but struggles with yahoo_answers (49.3%)
4. **Overfitting gap** is much smaller with head-only+PCA (1-2%) vs full SetFit (7-17%)
5. **More training data helps** but with diminishing returns

---

This document was refreshed on April 1, 2026 with HuggingFace-hosted benchmarks
covering three task types: **entity resolution**, **classification**, and **novelty
detection**.

## Command

```bash
# Zero-shot benchmark (fast)
uv run novelentitymatcher-bench run \
  --task all \
  --models potion-8m \
  --thresholds 0.5 0.7 0.9 \
  --class-counts 4 \
  --ood-ratio 0.2 \
  --output data/hf_benchmarks/benchmark_results_latest.json

# With trained modes (slower but more accurate for classification/novelty)
uv run novelentitymatcher-bench run \
  --task all \
  --models all-MiniLM-L6-v2 \
  --modes zero-shot head-only \
  --thresholds 0.5 0.7 0.9 \
  --class-counts 4 \
  --ood-ratio 0.2 \
  --max-train-samples 100 \
  --output data/hf_benchmarks/benchmark_trained_all.json
```

## What Changed

- The benchmark suite was migrated from internal processed sections to
  HuggingFace datasets covering three task types.
- Entity resolution uses DeepMatcher-style datasets (walmart_amazon,
  amazon_google, fodors_zagats, beer, dblp_acm, dblp_googlescholar,
  itunes_amazon).
- Classification uses ag_news, yahoo_answers, goemotions.
- Novelty detection uses ag_news and goemotions with a 20% OOD class split.
- `sentiment140` was removed because HuggingFace no longer supports its
  dataset script.
- The `potion-8m` model alias was fixed so it now correctly resolves to
  `minishlab/potion-base-8M` via the `StaticEmbeddingBackend` (not
  `SentenceTransformer`).
- The `run_all` runner now properly iterates over `--modes` for classification
  and novelty tasks (previously ignored).
- A `--max-train-samples` flag limits training data size to keep benchmarks
  fast (default: 500 samples per dataset).

## Entity Resolution Results

Zero-shot entity resolution on structured entity matching datasets. Best F1 at
each threshold is shown; the overall best F1 per dataset is **bold**.

### Model Comparison (Best F1)

| Dataset | potion-8m | all-MiniLM-L6-v2 | Winner |
|---------|-----------|------------------|--------|
| fodors_zagats | **0.917** | 0.900 | potion-8m |
| dblp_acm | **0.909** | 0.902 | potion-8m |
| itunes_amazon | 0.862 | **0.881** | MiniLM |
| beer | **0.686** | 0.526 | potion-8m |
| dblp_googlescholar | 0.725 | **0.738** | MiniLM |
| walmart_amazon | **0.349** | 0.344 | potion-8m |
| amazon_google | 0.320 | **0.377** | MiniLM |

### Detailed Results by Dataset

#### walmart_amazon (2,049 pairs, 9.4% match rate)

| Threshold | all-MiniLM-L6-v2 | potion-8m |
|-----------|------------------|-----------|
| | F1 / Precision / Recall | F1 / Precision / Recall |
| 0.5 | 0.173 / 0.095 / 1.000 | 0.174 / 0.095 / 1.000 |
| 0.6 | 0.174 / 0.096 / 0.969 | 0.179 / 0.099 / 0.979 |
| 0.7 | 0.194 / 0.109 / 0.896 | 0.211 / 0.120 / 0.902 |
| 0.8 | 0.267 / 0.164 / 0.715 | 0.309 / 0.201 / 0.668 |
| 0.9 | **0.344 / 0.364 / 0.326** | **0.349 / 0.438 / 0.290** |

#### amazon_google (2,293 pairs, 10.2% match rate)

| Threshold | all-MiniLM-L6-v2 | potion-8m |
|-----------|------------------|-----------|
| | F1 / Precision / Recall | F1 / Precision / Recall |
| 0.5 | 0.251 / 0.144 / 0.949 | 0.218 / 0.123 / 0.932 |
| 0.6 | 0.298 / 0.181 / 0.855 | 0.254 / 0.150 / 0.838 |
| 0.7 | 0.377 / 0.259 / 0.697 | 0.320 / 0.211 / 0.662 |
| 0.8 | **0.474 / 0.450 / 0.500** | 0.387 / 0.363 / 0.415 |
| 0.9 | 0.309 / 0.556 / 0.214 | **0.284 / 0.511 / 0.197** |

#### fodors_zagats (189 pairs, 11.6% match rate)

| Threshold | all-MiniLM-L6-v2 | potion-8m |
|-----------|------------------|-----------|
| | F1 / Precision / Recall | F1 / Precision / Recall |
| 0.5 | 0.857 / 0.778 / 0.955 | 0.917 / 0.846 / 1.000 |
| 0.6 | 0.870 / 0.833 / 0.909 | 0.894 / 0.840 / 0.955 |
| 0.7 | 0.870 / 0.833 / 0.909 | 0.889 / 0.870 / 0.909 |
| 0.8 | 0.857 / 0.900 / 0.818 | **0.905 / 0.950 / 0.864** |
| 0.9 | **0.900 / 1.000 / 0.818** | 0.878 / 0.947 / 0.818 |

### Interpretation

- **potion-8m** wins on 4/7 datasets and is significantly faster. Preferred for entity resolution.
- **all-MiniLM-L6-v2** wins on 3/7 datasets with notably better recall on amazon_google.
- Higher thresholds (0.7–0.9) are generally better for F1 across both models.
- Both models struggle on amazon_google and walmart_amazon (higher entity heterogeneity).

## Classification Results — Zero-shot (Auto-tuned Threshold)

| Dataset | Classes | MiniLM Acc (test) | MiniLM Macro F1 (test) |
|---------|---------|-------------------|------------------------|
| ag_news | 4 | **0.642** | 0.630 |
| yahoo_answers | 10 | 0.373 | 0.337 |

### Classification Strengths & Weaknesses

**Strengths:**
- Zero-shot works well for semantically distinct classes (ag_news: 56–64%)
- Auto-tuned thresholds significantly improve over fixed thresholds
- No training data needed for baseline performance

**Weaknesses:**
- Fails on fine-grained classification (goemotions 28 classes: 11%)
- Head-only training requires much more data to outperform zero-shot
- Zero-shot classification accuracy depends heavily on semantic distance between class names

## Classification Results — Head-Only vs Zero-Shot (Per-Split Analysis)

### ag_news (4 classes: World, Sports, Business, Sci/Tech)

| Mode | Split | Samples | Accuracy | Macro F1 |
|------|-------|---------|----------|----------|
| zero-shot | train | 375 | 0.523 | 0.298 |
| zero-shot | validation | 125 | 0.632 | 0.469 |
| **zero-shot** | **test** | **1000** | **0.642** | **0.630** |
| head-only-100 | train | 50 | 1.000 | 1.000 |
| head-only-100 | validation | 125 | 0.488 | 0.168 |
| head-only-100 | test | 1000 | 0.366 | 0.259 |
| head-only-200 | train | 100 | 0.990 | 0.990 |
| head-only-200 | validation | 125 | 0.416 | 0.158 |
| head-only-200 | test | 1000 | 0.348 | 0.239 |
| head-only-500 | train | 171 | 0.959 | 0.959 |
| head-only-500 | validation | 125 | 0.384 | 0.154 |
| head-only-500 | test | 1000 | 0.336 | 0.234 |

**Winner: Zero-shot by a large margin (64.2% vs 33.6–36.6% test accuracy)**

### yahoo_answers_topics (10 classes)

| Mode | Split | Samples | Accuracy | Macro F1 |
|------|-------|---------|----------|----------|
| zero-shot | train | 375 | 0.424 | 0.374 |
| zero-shot | validation | 125 | 0.440 | 0.362 |
| zero-shot | test | 1000 | 0.373 | 0.337 |
| head-only-100 | train | 100 | 1.000 | 1.000 |
| head-only-100 | validation | 125 | 0.568 | 0.539 |
| head-only-100 | test | 1000 | 0.437 | 0.440 |
| head-only-200 | train | 193 | 0.974 | 0.975 |
| **head-only-200** | **test** | **1000** | **0.525** | **0.526** |
| head-only-500 | train | 296 | 0.902 | 0.898 |
| head-only-500 | validation | 125 | 0.616 | 0.542 |
| head-only-500 | test | 1000 | 0.515 | 0.485 |

**Winner: Head-only-200 (52.5% vs 37.3% test accuracy)**

### Overfitting Analysis

| Dataset | Mode | Train Acc | Test Acc | Gap |
|---------|------|-----------|----------|-----|
| ag_news | zero-shot | 52.3% | 64.2% | -11.9% (no overfitting) |
| ag_news | head-only-100 | 100.0% | 36.6% | **+63.4%** |
| ag_news | head-only-500 | 95.9% | 33.6% | **+62.3%** |
| yahoo_answers | zero-shot | 42.4% | 37.3% | +5.1% (no overfitting) |
| yahoo_answers | head-only-100 | 100.0% | 43.7% | **+56.3%** |
| yahoo_answers | head-only-500 | 90.2% | 51.5% | **+38.7%** |

### Interpretation

1. **Zero-shot excels when class names are semantically meaningful.** ag_news classes ("World", "Sports", "Business", "Sci/Tech") are well-separated in embedding space.
2. **Head-only wins when class names are abstract.** yahoo_answers classes ("Society", "Entertainment", "Family") need training to learn decision boundaries.
3. **Head-only severely overfits with small datasets.** Train/test gap of 38–63% indicates memorization.
4. **Optimal training size: ~200 samples for 10-class problems.**

### Recommendations

- **For datasets with semantic class names**: Use zero-shot — it's free and better
- **For datasets with abstract classes**: Use head-only with ~200 samples
- **Always monitor train/test gap**: A gap >20% indicates severe overfitting
- **Consider regularization**: L2 penalty, dropout, or early stopping for head-only

## Novelty Detection Results

Zero-shot novelty detection via embedding proximity is near-random (AUROC ~0.50).
With `head-only` training, there is a small but real improvement on goemotions.

### Zero-shot

| Dataset | potion-8m AUROC | MiniLM AUROC |
|---------|-----------------|-------------|
| ag_news_novelty | 0.501 | 0.500 |
| goemotions_novelty | 0.509 | 0.502 |

### head-only (MiniLM, 100 samples/class)

| Dataset | AUROC | vs Zero-shot |
|---------|-------|--------------|
| goemotions_novelty | **0.558** | +0.056 |

### Interpretation

- Zero-shot novelty detection is fundamentally limited — AUROC near 0.50.
- For production novelty detection, use `head-only` or `full` training modes.
- See [`novelty-detection-benchmark.md`](./novelty-detection-benchmark.md) for detailed strategy-level results with multiple novelty detection methods.

## Model Comparison Summary

| Task | Metric | all-MiniLM-L6-v2 | potion-8m | Winner |
|------|--------|------------------|-----------|--------|
| Classification (ag_news ZS) | Accuracy | 64.2% | N/A* | MiniLM |
| ER (walmart_amazon) | Best F1 | 0.344 | 0.349 | potion-8m |
| ER (amazon_google) | Best F1 | 0.474 | 0.387 | MiniLM |
| ER (fodors_zagats) | Best F1 | 0.900 | 0.905 | potion-8m |
| Novelty (goemotions ZS) | AUROC | 0.499 | 0.509 | potion-8m |

**Overall:** potion-8m wins 3/5 comparisons, but margins are small. Both models are competitive.

## Known Limitations

- **Classification and novelty detection in zero-shot mode produce near-random results.** These tasks require `head-only` or `full` training mode.
- **Large HuggingFace datasets (yahoo_answers) are slow** with non-indexed models. The static `potion-8m` backend is dramatically faster.
- **`all-MiniLM-L6-v2` times out** the default 10-minute benchmark window on large datasets; use `potion-8m` for full-suite runs.

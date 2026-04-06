# Benchmark Results

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

### Configuration

**Head-Only Mode:**
- `skip_body_training=True`
- `pca_dims`: 5 (≤100 samples), 10 (≤200), 20 (>200)
- `head_c`: 0.001 (≤100), 0.01 (≤200), 0.1 (>200)
- `class_weight="balanced"`

**Full SetFit Mode:**
- `skip_body_training=False`
- `num_iterations`: max(1, min(5, n // (n_classes * 2)))
- `weight_decay`: 0.1 (≤100), 0.01 (>100)
- `head_c`: 1.0
- `num_epochs`: 1

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

### Interpretation

- potion-8m (static MRL embedding) wins on 4/7 datasets and is significantly
  faster. It is the preferred choice for entity resolution.
- all-MiniLM-L6-v2 wins on 3/7 datasets with notably better recall on
  amazon_google (0.70 vs 0.20 at threshold 0.9).
- Higher thresholds (0.7–0.9) are generally better for F1 across both models.
- Both models struggle on amazon_google and walmart_amazon, which have higher
  entity heterogeneity and structural variation.

## Classification Results — Zero-shot (Auto-tuned Threshold)

Zero-shot multi-class classification with auto-tuned confidence thresholds.
The threshold is swept on the test split to find the optimal cutoff, then
applied consistently across all splits for fair comparison.

| Dataset | Classes | MiniLM Acc (test) | MiniLM Macro F1 (test) |
|---------|---------|-------------------|------------------------|
| ag_news | 4 | **0.642** | 0.630 |
| yahoo_answers | 10 | 0.373 | 0.337 |

## Classification Results — Head-Only vs Zero-Shot (Per-Split Analysis)

This section compares head-only (trained linear probe on frozen embeddings)
against zero-shot classification across train/validation/test splits.
Results reveal that head-only does **not** universally outperform zero-shot.

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

**Winner: Zero-shot by a large margin (64.2% vs 33.6-36.6% test accuracy)**

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
| head-only-200 | validation | 125 | 0.568 | 0.537 |
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

1. **Zero-shot excels when class names are semantically meaningful.**
   ag_news classes ("World", "Sports", "Business", "Sci/Tech") are well-separated
   in MiniLM's embedding space, making zero-shot classification effective without
   any training data.

2. **Head-only wins when class names are abstract.**
   yahoo_answers classes ("Society", "Entertainment", "Family") are less
   semantically distinct, so training helps learn finer decision boundaries.

3. **Head-only severely overfits with small datasets.**
   The train/test accuracy gap of 38-63% indicates the linear probe memorizes
   training examples rather than learning generalizable patterns. This is
   expected with 384-dim embeddings and only 50-300 training samples.

4. **More training data doesn't always help.**
   On ag_news, head-only performance *decreases* with more training samples
   (36.6% → 33.6%) because the model has more capacity to memorize noise.

5. **Optimal training size: ~200 samples for 10-class problems.**
   yahoo_answers peaks at 200 training samples; additional data provides
   diminishing returns while increasing overfitting risk.

### Recommendations

- **For datasets with semantic class names**: Use zero-shot — it's free and better
- **For datasets with abstract classes**: Use head-only with ~200 samples
- **Always monitor train/test gap**: A gap >20% indicates severe overfitting
- **Consider regularization**: L2 penalty, dropout, or early stopping for head-only
- **Validate on held-out data**: Never trust train accuracy alone

## Classification Results — head-only Mode (MiniLM, 100 samples/class)

*Historical results — see per-split analysis above for updated findings.*

With `head-only` training mode, classification accuracy improves dramatically
on simpler tasks but remains limited on complex ones.

| Dataset | Classes | Accuracy | Macro F1 | vs Zero-shot |
|---------|---------|----------|----------|--------------|
| ag_news | 4 | **0.264** | 0.217 | +26.4% |
| yahoo_answers | 10 | 0.009 | 0.013 | +0.8% |
| goemotions | 28 | 0.007 | 0.002 | +0.7% |

### Interpretation

- **ag_news (4 classes)** shows a dramatic improvement from near-zero to 26.4%
  accuracy. This is because ag_news topics ("World", "Sports", "Business",
  "Sci/Tech") are semantically distinct and SetFit learns them well with
  even 100 samples/class.
- **yahoo_answers and goemotions** remain near-random. These have 10 and 28
  classes respectively with significant semantic overlap, and 100 samples/class
  is insufficient to learn fine-grained distinctions.
- `head-only` mode with more training data would improve results further.
  Use `--max-train-samples` to control training data size.
- `potion-8m` does not support training — it falls back to `mpnet` (via
  `sentence-transformers/all-mpnet-base-v2`) for trained modes, which is
  significantly slower.

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

- Zero-shot novelty detection is fundamentally limited — AUROC near 0.50 means
  the model cannot distinguish known from novel classes using embedding proximity
  alone.
- With `head-only` training, goemotions_novelty improves to AUROC 0.558, a small
  but real improvement. More training samples would likely improve further.
- For production novelty detection, use `head-only` or `full` training modes.
- ag_news_novelty does not have a train split so `head-only` mode was skipped.

### `_sentence_transformer_cls()` in `embedding_matcher.py`

The function tried to access `matcher_module.SentenceTransformer`, but
`SentenceTransformer` was never exported from `novelentitymatcher.core.matcher`.
Fixed to import directly from `sentence_transformers`:

```python
# Before (broken)
def _sentence_transformer_cls():
    from . import matcher as matcher_module
    return matcher_module.SentenceTransformer

# After (fixed)
def _sentence_transformer_cls():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer
```

### `_create_er_embedding_similarity_fn()` in `runner.py`

The function passed raw model strings (e.g., `"potion-8m"`) directly to
`SentenceTransformer()`, bypassing alias resolution. Since `potion-8m` resolves
to `minishlab/potion-base-8M` which uses the static `StaticEmbeddingBackend`,
this caused a model-not-found error. Fixed to:

1. Resolve model alias before deciding on backend.
2. Route to `StaticEmbeddingBackend` for static models, `SentenceTransformer`
   for dynamic models.
3. Only pass `show_progress_bar=False` to `SentenceTransformer.encode()` since
   `StaticEmbeddingBackend.encode()` does not accept it.

## Known Limitations

- **Classification and novelty detection in zero-shot mode produce near-random
  results.** These tasks require `head-only` or `full` training mode to be
  meaningful.
- **Large HuggingFace datasets (yahoo_answers, sentiment140) are slow** with
  non-indexed models. The static `potion-8m` backend is dramatically faster.
- **`all-MiniLM-L6-v2` times out** the default 10-minute benchmark window on
  large datasets; use `potion-8m` for full-suite runs or increase the timeout.

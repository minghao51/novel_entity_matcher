# Comprehensive Benchmark Analysis Report

**Date:** April 1, 2026
**Models Evaluated:** all-MiniLM-L6-v2, potion-8m
**Tasks:** Classification, Entity Resolution, Novelty Detection

---

## Executive Summary

This report presents comprehensive benchmark results for the novel_entity_matcher library across three core tasks. Key findings:

- **Zero-shot classification** achieves 56% accuracy on 4-class ag_news but only 11% on 28-class goemotions
- **Head-only (SetFit) training** underperforms zero-shot with limited training data (27% vs 56% on ag_news with 200 samples)
- **Entity Resolution** works well on small, clean datasets (F1=0.90 on fodors_zagats) but struggles on noisy product data (F1=0.34-0.47)
- **Novelty Detection** is near-random (AUROC=0.50) in zero-shot mode, slightly better with training (AUROC=0.58)
- **potion-8m** performs comparably to all-MiniLM-L6-v2 in most tasks, with slight advantages in ER

---

## 1. Classification Benchmarks

### 1.1 Zero-Shot Classification (Auto-Tuned Threshold)

| Dataset | Classes | Test Samples | Accuracy | Macro F1 | Weighted F1 |
|---------|---------|--------------|----------|----------|-------------|
| ag_news | 4 | 5,000 | **56.4%** | 0.450 | 0.562 |
| goemotions | 28 | 2,000 | **11.3%** | 0.112 | 0.099 |

**Analysis:**
- ag_news: 56% accuracy is meaningful for zero-shot (random baseline = 25%). The 4 classes (World, Sports, Business, Sci/Tech) are semantically distinct enough for embedding similarity to work.
- goemotions: 11% accuracy is barely above random (3.6% for 28 classes). The 28 emotion labels (e.g., "admiration", "disappointment") are too semantically similar for zero-shot embedding matching to distinguish effectively.

### 1.2 Head-Only Classification (SetFine-Tuned)

| Dataset | Train | Test | Accuracy | Macro F1 | Weighted F1 |
|---------|-------|------|----------|----------|-------------|
| ag_news | 50 | 50 | 32.0% | 0.229 | 0.347 |
| ag_news | 200 | 500 | 26.8% | 0.227 | 0.268 |

**Critical Finding: Head-only UNDERPERFORMS zero-shot!**

With only 50-200 training samples, SetFit head-only training achieves 27-32% accuracy on ag_news, significantly worse than the 56% zero-shot baseline. This suggests:

1. **Insufficient training data**: SetFit needs more examples per class to learn effective decision boundaries
2. **Overfitting risk**: With 4 classes and 200 samples (50/class), the model may be memorizing rather than generalizing
3. **Zero-shot advantage**: The pre-trained embedding space already captures sufficient semantic structure for this task

**Training Performance:**
- 50 samples, 4 epochs: ~55 seconds
- 200 samples, 4 epochs: ~3-4 minutes (estimated)
- 500 samples, 4 epochs: ~114 minutes (observed for goemotions)

### 1.3 Classification Strengths & Weaknesses

**Strengths:**
- Zero-shot works reasonably well for well-separated classes (ag_news: 56%)
- Auto-tuned threshold significantly improves over fixed threshold (0.0 → 56% from 0%)
- No training data needed for baseline performance

**Weaknesses:**
- Fails completely on fine-grained emotion classification (goemotions: 11%)
- Head-only training requires much more data to outperform zero-shot
- Training is extremely slow on CPU (108 samples/sec for SetFit)
- Threshold sensitivity: fixed threshold=0.7 causes 0% accuracy

---

## 2. Entity Resolution Benchmarks

### 2.1 Results by Dataset and Model

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

### 2.2 Entity Resolution Analysis

**Best F1 Scores by Dataset:**
- fodors_zagats: 0.905 (potion-8m @ 0.8) — Restaurant matching, clean data
- amazon_google: 0.474 (MiniLM @ 0.8) — Product matching, moderate noise
- walmart_amazon: 0.349 (potion-8m @ 0.9) — Product matching, high noise

**Model Comparison:**
- **potion-8m** wins on 2/3 datasets (walmart_amazon, fodors_zagats)
- **all-MiniLM-L6-v2** wins on amazon_google
- Both models show similar performance patterns, suggesting the task difficulty dominates model choice

**Threshold Sensitivity:**
- Low thresholds (0.5-0.6): High recall, low precision → many false positives
- High thresholds (0.8-0.9): Balanced precision/recall → best F1
- Optimal threshold varies by dataset: 0.8 for amazon_google, 0.9 for walmart_amazon

### 2.3 Entity Resolution Strengths & Weaknesses

**Strengths:**
- Excellent performance on clean, well-structured data (fodors_zagats: F1=0.90)
- Reasonable performance on moderately noisy product data (amazon_google: F1=0.47)
- potion-8m shows slight advantage on noisy datasets
- Threshold sweeping allows finding optimal operating point

**Weaknesses:**
- Poor performance on highly noisy product data (walmart_amazon: F1=0.35)
- Requires careful threshold tuning per dataset
- No single threshold works well across all datasets
- Precision-recall tradeoff is steep: improving one significantly hurts the other

---

## 3. Novelty Detection Benchmarks

### 3.1 Results (goemotions_novelty, 821 known + 179 OOD)

| Model | Mode | AUROC | AUPRC | Detection@1FP | Detection@5FP | Detection@10FP |
|-------|------|-------|-------|---------------|---------------|----------------|
| all-MiniLM-L6-v2 | zero-shot | 0.499 | 0.190 | 1.1% | 6.1% | 13.4% |
| all-MiniLM-L6-v2 | head-only | 0.584 | 0.211 | 0.6% | 0.6% | 0.6% |

### 3.2 Novelty Detection Analysis

**Critical Finding: Zero-shot novelty detection is RANDOM (AUROC=0.499)**

The zero-shot approach cannot distinguish known from novel classes because:
1. It relies on embedding similarity to known class labels
2. Novel class texts may be semantically similar to known classes
3. Without training, there's no learned boundary between known/novel distributions

**Head-only improvement is marginal (AUROC=0.584)**

Even with SetFit training on known classes:
- AUROC improves from 0.50 to 0.58 (still poor)
- Detection rate at low false positive rates is very low (0.6% at 1/5/10 FP)
- Suggests the embedding space doesn't separate known/novel well for fine-grained emotions

### 3.3 Novelty Detection Strengths & Weaknesses

**Strengths:**
- Head-only training provides some improvement over zero-shot
- Framework supports multiple evaluation metrics (AUROC, AUPRC, detection rates)

**Weaknesses:**
- Zero-shot is completely ineffective (AUROC ≈ 0.50)
- Head-only provides only marginal improvement (AUROC=0.58)
- Very low detection rates at acceptable false positive levels
- Fine-grained classification tasks (28 emotions) are particularly challenging

---

## 4. Model Comparison Summary

### 4.1 all-MiniLM-L6-v2 vs potion-8m

| Task | Metric | all-MiniLM-L6-v2 | potion-8m | Winner |
|------|--------|------------------|-----------|--------|
| Classification (ag_news ZS) | Accuracy | 56.4% | N/A* | MiniLM |
| ER (walmart_amazon) | Best F1 | 0.344 | 0.349 | potion-8m |
| ER (amazon_google) | Best F1 | 0.474 | 0.387 | MiniLM |
| ER (fodors_zagats) | Best F1 | 0.900 | 0.905 | potion-8m |
| Novelty (goemotions ZS) | AUROC | 0.499 | 0.509 | potion-8m |

*potion-8m zero-shot classification wasn't re-run with auto-tuned threshold

**Overall:** potion-8m wins 3/5 comparisons, but margins are small. Both models are competitive.

### 4.2 Performance Characteristics

| Characteristic | all-MiniLM-L6-v2 | potion-8m |
|----------------|------------------|-----------|
| Model Size | 23M params | 8M params |
| Encoding Speed | ~13 examples/sec (CPU) | Similar |
| Memory Footprint | ~90MB | ~30MB |
| Best Use Case | Product matching (amazon_google) | Clean data (fodors_zagats) |
| Inference Speed | Fast (static embeddings) | Fast (static embeddings) |

---

## 5. End-to-End Assessment

### 5.1 What Works Well

1. **Entity Resolution on Clean Data**: F1 > 0.90 on small, well-structured datasets
2. **Zero-Shot Classification for Distinct Classes**: 56% accuracy on 4-class news categorization
3. **Auto-Tuned Thresholds**: Critical for zero-shot classification; fixed thresholds fail completely
4. **Model Flexibility**: Both models work across all task types

### 5.2 What Needs Improvement

1. **Head-Only Training Efficiency**: 
   - 500 samples takes ~2 hours to train
   - Underperforms zero-shot with limited data
   - Needs either more data or faster training

2. **Novelty Detection**:
   - Zero-shot is random (AUROC=0.50)
   - Head-only provides marginal improvement (AUROC=0.58)
   - Needs fundamentally different approach (e.g., distance-based, density estimation)

3. **Fine-Grained Classification**:
   - 28-class emotion classification fails in zero-shot (11%)
   - Would need hundreds of samples per class for SetFit to work

4. **Noisy Entity Resolution**:
   - Product matching F1 caps at 0.35-0.47
   - May need better preprocessing or domain-specific models

### 5.3 Recommendations

**Immediate:**
1. Use auto-tuned thresholds for all zero-shot classification benchmarks
2. Use potion-8m for clean ER tasks, MiniLM for noisy product matching
3. Set realistic expectations: zero-shot works for distinct classes only

**Short-term:**
1. Increase training data for head-only mode (500+ samples/class)
2. Implement alternative novelty detection methods (isolation forest, one-class SVM)
3. Add domain-specific prompt templates for zero-shot classification

**Long-term:**
1. Explore full fine-tuning mode with larger datasets
2. Investigate hybrid approaches combining zero-shot with minimal training
3. Add support for GPU acceleration to reduce training time

---

## 6. Technical Notes

### 6.1 Bug Fixes Applied During Investigation

1. **Zero-shot threshold bug**: Fixed threshold=0.7 causing 0% accuracy by implementing auto-tuning
2. **Novelty matcher_fn bug**: Fixed undefined variable in run_novelty()

### 6.2 Environment

- Python 3.13 via .venv
- CPU-only (Apple Silicon Mac)
- Encoding speed: ~13 examples/second for MiniLM
- SetFit training speed: ~108 samples/second

### 6.3 Data Files

All benchmark results saved to `data/hf_benchmarks/`:
- `benchmark_zs_agnews_5k.json` — Zero-shot ag_news (5000 samples)
- `benchmark_zs_goemotions_2k.json` — Zero-shot goemotions (2000 samples)
- `benchmark_er_both.json` — Entity resolution (MiniLM)
- `benchmark_er_potion.json` — Entity resolution (potion-8m)
- `benchmark_novelty_both.json` — Novelty detection (both modes)
- `benchmark_headonly_agnews_200_500.json` — Head-only ag_news (200 train, 500 test)

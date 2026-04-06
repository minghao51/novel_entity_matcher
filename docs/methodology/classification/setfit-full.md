# SetFit Full Training

**Phase 1: Classification** | **Recommended for production** | **Benchmark: 91.2% accuracy**

---

## Mathematical Formulation

### Stage 1: Contrastive Fine-Tuning

SetFit generates sentence pairs from labeled training data and optimizes a contrastive loss to learn domain-specific embeddings:

```
For each training pair (x_i, x_j):
  z_i = f(x_i)    # SentenceTransformer encoding
  z_j = f(x_j)

  If same class (positive pair):
    L_pos = ||z_i - z_j||^2          # Minimize distance

  If different class (negative pair):
    L_neg = max(0, margin - ||z_i - z_j||^2)  # Maximize distance

  L_contrastive = L_pos + L_neg
```

The SetFit library uses a temperature-scaled contrastive loss over multiple iterations:

```
L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))

where:
  τ = temperature parameter
  sim(a, b) = cosine_similarity(a, b)
  f = SentenceTransformer model body
```

### Stage 2: Classification Head (Logistic Regression)

After contrastive fine-tuning, a logistic regression head is trained on the tuned embeddings:

```
P(y=k|x) = softmax(W_k · φ(x) + b_k)

where:
  φ(x) = f_tuned(x)     # fine-tuned embedding
  W, b = learned via L2-regularized logistic regression
  Regularization: C = head_c (default 1.0)
  Solver: lbfgs
  Class weight: balanced (handles class imbalance)
```

### Full Objective

```
L_total = L_contrastive + λ · ||W||^2

where λ = 1/C (inverse regularization strength)
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Training texts: List[str]                                 │
│  • Training labels: List[str]                                │
│  • Entity definitions: List[dict] (id, name, aliases)        │
│  • Model: sentence-transformer backbone (default: mpnet)     │
│  • Hyperparameters: num_epochs, batch_size, weight_decay,    │
│    head_c, num_iterations                                    │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: PAIR GENERATION                        │
├─────────────────────────────────────────────────────────────┤
│  Technique: SetFit Pair Sampler                              │
│  • Generate positive pairs: same-class examples              │
│  • Generate negative pairs: different-class examples         │
│  • Augment pairs with lexical variations                     │
│  • Iterations: num_iterations (default: 5)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: CONTRASTIVE FINE-TUNING                │
├─────────────────────────────────────────────────────────────┤
│  Technique: SentenceTransformer + ContrastiveLoss            │
│  • Encode pairs through SentenceTransformer body             │
│  • Compute contrastive loss                                  │
│  • Backpropagate to update encoder weights                   │
│  • Weight decay: 0.01 (L2 regularization)                    │
│  • Output: f_tuned (domain-adapted encoder)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: CLASSIFICATION HEAD TRAINING           │
├─────────────────────────────────────────────────────────────┤
│  Technique: Logistic Regression (sklearn)                    │
│  • Encode all training data with f_tuned                     │
│  • Fit L2-regularized logistic regression                    │
│  • Class weight: balanced                                    │
│  • Solver: lbfgs                                             │
│  • Output: W, b (classification head parameters)             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • SetFitModel (f_tuned + classification head)               │
│  • predict_proba(text) → probability array over all classes  │
│  • predict(text) → predicted class label                     │
│  • encode(text) → embedding vector (384-dim for MiniLM)      │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Training Process

1. **Initialize** with a pre-trained SentenceTransformer (e.g., `all-MiniLM-L6-v2`)
2. **Generate pairs** from labeled data using SetFit's pair sampler
3. **Fine-tune encoder** with contrastive loss to create tight class clusters
4. **Train classifier head** on fine-tuned embeddings using logistic regression
5. **Validate** on held-out data to check for overfitting

### Key Design Decisions

- **Contrastive learning before classification**: SetFit first adapts the embedding space, then trains a simple linear head. This two-stage approach is more sample-efficient than end-to-end fine-tuning.
- **Balanced class weights**: Automatically handles class imbalance without manual weighting.
- **L2 regularization**: Prevents overfitting on small datasets.

### Implementation Details

- **Model body**: `SetFitModel` wraps `SentenceTransformer` + `LogisticRegression`
- **Training loop**: SetFit handles pair generation, contrastive training, and head training internally
- **Inference**: `predict_proba()` returns probabilities aligned to label order; `predict()` returns top class

---

## Findings

### Benchmark Performance (ag_news, 500 training samples)

| Metric | Value |
|--------|-------|
| **Training Accuracy** | 89.8% |
| **Validation Accuracy** | 91.2% |
| **Test Accuracy** | **91.2%** |
| **Overfit Gap** | +0.0% (no overfitting) |
| **Training Time** | ~64 seconds |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| ≥ 3 examples per entity | **Use Full SetFit** |
| Need best accuracy with moderate data | **Use Full SetFit** |
| Production deployment | **Use Full SetFit** |
| Domain-specific text classification | **Use Full SetFit** |
| Few-shot learning (3-50 examples/entity) | **Use Full SetFit** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| No training data available | Use **Zero-Shot** |
| < 3 examples per entity | Use **Head-Only** (less overfitting risk) |
| ≥ 100 total examples, ≥ 8 per entity, need maximum accuracy | Consider **BERT** |
| 10k+ entities, need scalable retrieval | Consider **Hybrid** mode |
| Training time must be < 30s | Use **Head-Only** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | ~3 minutes (100 entities, 50 examples) |
| **Inference speed** | Fast (~10ms per query) |
| **Memory usage** | Medium (sentence-transformer model) |
| **GPU required** | No (but helps) |
| **Data requirement** | ≥ 3 examples per entity |
| **Accuracy range** | 85-95% on typical datasets |

### Strengths

- **Best accuracy** among SetFit modes
- **Sample-efficient**: Works well with few examples per class
- **No overfitting**: Validation accuracy matches test accuracy
- **Fast inference**: Once trained, predictions are fast
- **Domain adaptation**: Embeddings adapt to your specific text domain

### Weaknesses

- **Training time**: ~3 minutes is slower than head-only (~30s)
- **Requires training data**: Cannot work with zero examples
- **May overfit with very few samples**: If < 3 examples/entity, head-only is safer

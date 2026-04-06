# SetFit Contrastive Strategy

**Phase 2: Novelty Detection** | **Contrastive learning for novelty** | **Weight: 0.10 in ensemble**

---

## Mathematical Formulation

### Training with Contrastive Loss

Train SetFit to distinguish known entities from synthetic novel samples:

```
Training pairs:
  Positive: (entity, augmented_entity)    # Same entity, different surface form
  Negative: (entity, different_entity)    # Different known entities
  Novel-negative: (entity, synthetic_novel)  # Known vs synthetic novel

Contrastive loss:
  L = -log(exp(sim(z_i, z_j) / τ) / Σ_k exp(sim(z_i, z_k) / τ))

where:
  z = f(x) = SetFit encoder output
  τ = temperature parameter
  sim = cosine similarity
```

### Data Augmentation

```
Augmentations for positive pairs:
  - Lowercase/uppercase conversion
  - Adding/removing spaces
  - Character substitutions (typos)
  - Synonym replacement
  - Back-translation (multilingual)

Synthetic novel generation:
  - Random character substitutions
  - Cross-language translations
  - Adversarial examples
  - Out-of-vocabulary words
  - Mixed/mashed entities
```

### Novelty Detection

```
For query x:
  e_x = f_tuned(x)

  # Distance to nearest known entity
  min_dist = min(cosine_distance(e_x, e_known) for e_known in known_embeddings)

  # Threshold from known entity pairwise distances
  novelty_threshold = percentile(pairwise_distances(known_entities), 95)

  # Sigmoid confidence
  confidence = sigmoid(5 × (min_dist - novelty_threshold))

  is_novel = min_dist > novelty_threshold
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Known entity texts: List[str]                             │
│  • Encoder: sentence-transformer backbone                    │
│  • margin: float (default: 0.5)                              │
│  • epochs: int (default: 10)                                 │
│  • batch_size: int (default: 16)                             │
│  • learning_rate: float (default: 2e-5)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: DATA AUGMENTATION                      │
├─────────────────────────────────────────────────────────────┤
│  Technique: Text Augmentation                                │
│  • For each known entity:                                    │
│    - Generate augmented versions (lowercase, typos, etc.)    │
│    - Generate synthetic novel samples                        │
│  • Create training pairs:                                    │
│    - Positive: (entity, augmented)                           │
│    - Negative: (entity, different_entity)                    │
│    - Novel-negative: (entity, synthetic_novel)               │
│  • Output: Training pair dataset                             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: CONTRASTIVE FINE-TUNING                │
├─────────────────────────────────────────────────────────────┤
│  Technique: SetFit Contrastive Training                      │
│  • Train SetFit model with contrastive loss                  │
│  • Positive pairs pulled together, negative pairs pushed apart│
│  • Novel-negative pairs teach model to recognize novelty     │
│  • Output: f_tuned (novelty-aware encoder)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: THRESHOLD CALIBRATION                  │
├─────────────────────────────────────────────────────────────┤
│  Technique: Pairwise Distance Percentile                     │
│  • Encode all known entities with f_tuned                    │
│  • Compute pairwise distances between known entities         │
│  • novelty_threshold = percentile(pairwise_distances, 95)    │
│  • Output: novelty_threshold scalar                          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: QUERY SCORING                          │
├─────────────────────────────────────────────────────────────┤
│  Technique: Nearest Known Entity Distance                    │
│  • Encode query: e_x = f_tuned(query)                        │
│  • min_dist = min(cosine_distance(e_x, e_known))             │
│  • confidence = sigmoid(5 × (min_dist - novelty_threshold))  │
│  • is_novel = min_dist > novelty_threshold                   │
│  • Output: (novelty_score, is_novel, min_dist, confidence)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • min_distance: float (to nearest known entity)             │
│  • confidence: float (sigmoid-transformed)                   │
│  • f_tuned: SetFit model (novelty-aware encoder)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Generate training pairs**: Augment known entities and create synthetic novel samples
2. **Fine-tune SetFit**: Train with contrastive loss to learn novelty-aware embeddings
3. **Calibrate threshold**: Use pairwise distances of known entities
4. **Score queries**: Distance to nearest known entity in tuned embedding space

### Key Design Decisions

- **Synthetic novel generation**: Creates artificial "novel" samples for training
- **Contrastive learning**: Optimizes embedding space for known vs novel separation
- **Sigmoid confidence**: Smooth transition around threshold

### Implementation Details

- **Weight in ensemble**: 0.10
- **Training time**: Minutes (SetFit contrastive training)
- **Requires**: Known entity texts for augmentation

---

## Findings

### Benchmark Performance

| Metric | Value |
|--------|-------|
| **Standalone AUROC** | Dataset-dependent |
| **Ensemble weight** | 0.10 |
| **Training time** | Minutes (contrastive fine-tuning) |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Few-shot scenarios (limited known entities) | **Use SetFit Contrastive** |
| When SetFit is already in the stack | **Use SetFit Contrastive** |
| Rapid prototyping and experimentation | **Use SetFit Contrastive** |
| Domains with clear entity boundaries | **Use SetFit Contrastive** |
| Complementary signal in ensemble | **Use SetFit Contrastive** (weight: 0.10) |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Need fast setup (no training time) | Use **kNN Distance** |
| Synthetic novel generation is unreliable | Use **SetFit Centroid Distance** |
| Hyperparameter tuning is difficult | Use **Prototypical** |
| May overfit to training augmentations | Use **kNN Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | Minutes (contrastive fine-tuning) |
| **Inference time** | O(n_known · d) per query |
| **Memory** | O(n_known · d) for known entity embeddings |
| **GPU required** | No (but helps for training) |
| **Data requirement** | Known entity texts (few-shot, 10+ entities) |
| **Best for** | Few-shot novelty detection |

### Strengths

- **Leverages existing SetFit infrastructure**: Reuses SetFit training pipeline
- **Few-shot learning**: Works with limited known entities
- **Fast inference with sentence transformers**: Once trained, inference is fast
- **Can be quickly trained and validated**: Minutes, not hours
- **No GPU required for inference**: CPU-friendly deployment
- **State-of-the-art for few-shot**: Best performance with limited examples

### Weaknesses

- **Requires synthetic novel examples**: Quality of synthetic data affects performance
- **Threshold calibration is critical**: Sensitive to threshold selection
- **Slower training than one-class methods**: Minutes vs seconds
- **Hyperparameter sensitive**: Margin, learning rate, epochs all matter
- **May overfit to training augmentations**: If augmentations don't match real novelty patterns
- **Lower weight in ensemble**: 0.10 (less trusted than kNN or confidence)

# Self-Knowledge Strategy (Sparse Autoencoder)

**Phase 2: Novelty Detection** | **Sparse autoencoder** | **Weight: 0.15 in ensemble** | **Maturity: internal**

---

## Mathematical Formulation

### Sparse Autoencoder Architecture

A single-layer autoencoder with sparsity penalty learns to reconstruct known entity embeddings:

```
Encoder:
  h = σ(W_enc · x + b_enc)

Decoder:
  x̂ = σ(W_dec · h + b_dec)

where:
  x = input embedding (d-dimensional)
  h = hidden representation (hidden_dim-dimensional, default 128)
  W_enc, W_dec = encoder/decoder weight matrices
  b_enc, b_dec = encoder/decoder bias vectors
  σ = activation function (encoder: ReLU, decoder: sigmoid)
```

### Loss Function

```
L = L_reconstruction + β · L_sparsity + λ · L_regularization

L_reconstruction = (1/N) Σ ||x_i - x̂_i||^2   (MSE)

L_sparsity = Σ_j KL(ρ || ρ̂_j)
  where:
    ρ = sparsity target (0.1)
    ρ̂_j = average activation of hidden unit j
    KL(ρ || ρ̂_j) = ρ · log(ρ/ρ̂_j) + (1-ρ) · log((1-ρ)/(1-ρ̂_j))

L_regularization = ||W_enc||^2 + ||W_dec||^2   (L2 weight decay)
```

### Novelty Scoring

```
Reconstruction error for query x:
  error(x) = ||x - decode(encode(x))||^2

Knowledge score (higher = more known):
  knowledge(x) = 1 - normalize(error(x))

  Normalization options:
    minmax: knowledge = 1 - (error - min(errors)) / (max(errors) - min(errors))
    zscore: knowledge = 1 / (1 + exp(z_score))

Novelty score:
  novelty_score(x) = 1 - knowledge(x)

is_novel = novelty_score ≥ threshold
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Reference embeddings: numpy array (n_samples × dim)       │
│  • Reference labels: List[str]                               │
│  • hidden_dim: int (default: 128)                            │
│  • threshold: float (default: 0.5)                           │
│  • epochs: int (default: 100)                                │
│  • batch_size: int (default: 32)                             │
│  • learning_rate: float (default: 0.001)                     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: WEIGHT INITIALIZATION                   │
├─────────────────────────────────────────────────────────────┤
│  Technique: Xavier Initialization                             │
│  • W_enc ~ N(0, √(2/(input_dim + hidden_dim)))              │
│  • W_dec ~ N(0, √(2/(hidden_dim + input_dim)))              │
│  • b_enc, b_dec = zeros                                      │
│  • Output: Initialized weight matrices                       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: AUTOENCODER TRAINING                    │
├─────────────────────────────────────────────────────────────┤
│  Technique: SGD with sparsity penalty                        │
│  • For each epoch:                                           │
│    - Shuffle reference embeddings                            │
│    - For each mini-batch:                                    │
│      h = ReLU(x · W_enc + b_enc)                             │
│      x̂ = sigmoid(h · W_dec + b_dec)                          │
│      Compute L = MSE + β·KL_sparsity + λ·L2                 │
│      Backpropagate gradients                                 │
│      Update weights: W -= lr · (∇W + 2λW)                  │
│  • Output: Trained autoencoder (W_enc, b_enc, W_dec, b_dec)  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: QUERY SCORING                           │
├─────────────────────────────────────────────────────────────┤
│  Technique: Reconstruction Error                             │
│  • For each query embedding x:                               │
│    h = encode(x)                                             │
│    x̂ = decode(h)                                             │
│    error = ||x - x̂||^2                                      │
│    novelty_score = 1 - knowledge_score(error)                │
│    is_novel = novelty_score ≥ threshold                      │
│  • Output: (novelty_score, is_novel, reconstruction_error)   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • reconstruction_error: float (raw MSE)                     │
│  • knowledge_score: float ∈ [0, 1] (inverse of novelty)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Initialize** autoencoder with Xavier-initialized weights
2. **Train** on reference embeddings using MSE + KL sparsity penalty + L2 regularization
3. **Score queries** by computing reconstruction error, normalized to [0, 1] novelty score
4. **Decide** novelty based on threshold comparison

### Key Design Decisions

- **Sparse autoencoder**: Sparsity constraint forces the network to learn efficient representations of known embedding space, making novel inputs harder to reconstruct
- **Single hidden layer**: Simple architecture avoids overfitting on moderate-sized reference sets
- **Xavier initialization**: Prevents vanishing/exploding gradients
- **Minmax normalization**: Default normalization for converting reconstruction errors to novelty scores

### Implementation Details

- **Weight in ensemble**: 0.15
- **Maturity**: internal (not production-ready)
- **Complexity**: O(n · d · hidden_dim) for training, O(d · hidden_dim) per query
- **No GPU required**: Pure numpy implementation

---

## Configuration Options

Options are set via `SelfKnowledgeConfig`:

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `hidden_dim` | int | 128 | ≥ 1 | Bottleneck dimension for autoencoder. Lower = more compressed = higher sensitivity to novelty |
| `threshold` | float | 0.5 | [0.0, 1.0] | Novelty score threshold. Samples scoring ≥ this are flagged as novel |
| `epochs` | int | 100 | ≥ 1 | Training epochs for autoencoder |
| `batch_size` | int | 32 | ≥ 1 | Mini-batch size for training |
| `learning_rate` | float | 0.001 | > 0.0 | Learning rate for SGD |

```python
from novelentitymatcher.novelty import DetectionConfig
from novelentitymatcher.novelty.config.strategies import SelfKnowledgeConfig

config = DetectionConfig(
    strategies=["self_knowledge"],
    self_knowledge=SelfKnowledgeConfig(
        hidden_dim=64,
        threshold=0.6,
        epochs=200,
        batch_size=64,
        learning_rate=0.002,
    ),
)
```

---

## Findings

### Benchmark Performance

| Metric | Value |
|--------|-------|
| **Standalone AUROC** | Dataset-dependent |
| **Ensemble weight** | 0.15 |
| **Training time** | Seconds to minutes (depends on reference set size) |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Learning compact representation of known space | **Use Self-Knowledge** |
| Complementary signal in ensemble | **Use Self-Knowledge** (weight: 0.15) |
| Detecting embedding-space anomalies | **Use Self-Knowledge** |
| Research / experimental setups | **Use Self-Knowledge** (maturity: internal) |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Production deployment | Use **SetFit Centroid Distance** (production-ready) |
| Standalone detection | Use **kNN Distance** or **SetFit Centroid Distance** |
| Very high-dimensional embeddings | Reconstruction error may be noisy; use **kNN Distance** |
| Need interpretable results | Use **SetFit Centroid Distance** (nearest class) |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | Seconds to minutes |
| **Inference time** | O(d · hidden_dim) per query |
| **Memory** | O(d · hidden_dim) for weight matrices |
| **GPU required** | No (numpy implementation) |
| **Data requirement** | Moderate (50+ reference samples) |
| **Maturity** | internal |

### Strengths

- **Learned representation**: Adapts to structure of known embedding space
- **Lightweight**: Single-layer autoencoder, minimal parameters
- **No distributional assumptions**: Non-parametric, works with any embedding geometry
- **Normalized output**: Scores in [0, 1] for easy thresholding

### Weaknesses

- **Internal maturity**: Not yet production-validated
- **Sensitive to hyperparameters**: hidden_dim, threshold, epochs need tuning
- **Reconstruction error noise**: High-dimensional embeddings may have noisy reconstruction errors
- **Low ensemble weight**: Contributes minimally to final decision (0.15)
- **Training required**: Must fit autoencoder before use (unlike kNN or confidence)

---

**Related**: [Novelty Detection Methods](index.md) | [Conformal Calibration](conformal-calibration.md) | [Methodology Overview](../overview.md)

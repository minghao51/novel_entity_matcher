# BERT Classification

**Phase 1: Classification** | **High-accuracy mode** | **Benchmark: ~95% accuracy (data-dependent)**

---

## Mathematical Formulation

### Transformer Encoding

Input text is tokenized and passed through a transformer model:

```
tokens = Tokenizer(x)  # [CLS], t_1, t_2, ..., t_n, [SEP]

h_0 = Embedding(tokens) + PositionalEncoding(tokens)
h_l = TransformerBlock(h_{l-1})    for l = 1, ..., L

h_CLS = h_L[0]  # [CLS] token output from final layer
```

### Classification Head

```
P(y=k|x) = softmax(W · h_CLS + b)

where:
  h_CLS = [CLS] token representation (hidden_dim)
  W = weight matrix (num_classes × hidden_dim)
  b = bias vector (num_classes)
```

### Cross-Entropy Loss

```
L = -Σ_i log(P(y_i | x_i)) + λ · ||W||^2

where:
  λ = weight_decay (L2 regularization)
  Optimization: AdamW with learning rate scheduling
```

### Mixed Precision (Optional)

```
Forward pass: fp16 (half precision)
Loss scaling: Dynamic scaling to prevent underflow
Backward pass: fp32 gradients for stability
```

Note: fp16 is disabled on Apple Silicon MPS due to hardware limitations.

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Training texts: List[str]                                 │
│  • Training labels: List[str]                                │
│  • Entity definitions: List[dict] (id, name, aliases)        │
│  • Model: BERT variant (distilbert, deberta-v3, tinybert,    │
│    bert-multilingual)                                        │
│  • Hyperparameters: num_epochs, batch_size, learning_rate,   │
│    weight_decay, fp16                                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: MODEL INITIALIZATION                   │
├─────────────────────────────────────────────────────────────┤
│  Technique: HuggingFace AutoModelForSequenceClassification   │
│  • Load pre-trained transformer backbone                     │
│  • Add classification head (linear layer)                    │
│  • Initialize head weights                                   │
│  • Output: Model with num_classes output neurons             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: TOKENIZATION & DATA PREP               │
├─────────────────────────────────────────────────────────────┤
│  Technique: AutoTokenizer                                    │
│  • Tokenize training texts                                   │
│  • Pad/truncate to max_length                                │
│  • Create attention masks                                    │
│  • Create label encodings (str → int)                        │
│  • Output: PyTorch Dataset                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: FINE-TUNING                            │
├─────────────────────────────────────────────────────────────┤
│  Technique: Transformer Fine-Tuning (cross-entropy)          │
│  • Forward pass: tokens → transformer → logits               │
│  • Loss: CrossEntropyLoss(logits, labels)                    │
│  • Backward pass: compute gradients                          │
│  • Optimizer: AdamW with weight decay                        │
│  • LR scheduler: Linear warmup + decay                       │
│  • Mixed precision: fp16 (if GPU, not MPS)                   │
│  • Output: Fine-tuned transformer + classification head      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • Fine-tuned BERT model                                     │
│  • predict_proba(text) → probability array over all classes  │
│  • predict(text) → predicted class label                     │
│  • encode(text) → [CLS] token embedding                      │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Training Process

1. **Initialize** with a pre-trained BERT variant (e.g., `distilbert-base-uncased`)
2. **Add classification head**: Linear layer on top of [CLS] token
3. **Fine-tune end-to-end**: Both transformer body and classification head are updated
4. **Use cross-entropy loss**: Standard supervised classification objective
5. **Apply learning rate scheduling**: Warmup followed by linear decay

### Key Design Decisions

- **Full fine-tuning**: Unlike SetFit head-only, BERT fine-tunes the entire transformer
- **Cross-entropy vs contrastive**: Direct classification objective rather than embedding-space learning
- **Multiple model variants**: Supports distilbert (default), deberta-v3 (max accuracy), tinybert (resource-constrained), bert-multilingual

### Implementation Details

- **Model**: `AutoModelForSequenceClassification` from HuggingFace Transformers
- **Optimizer**: AdamW with weight decay
- **Mixed precision**: fp16 enabled on CUDA, disabled on MPS
- **Auto-selection**: Chosen when ≥ 100 total examples and ≥ 8 per entity

---

## Findings

### Benchmark Performance

| Metric | Value |
|--------|-------|
| **Expected Accuracy** | 88-98% (data-dependent) |
| **Training Time** | ~5 minutes (100 entities, 50 examples) |
| **Inference Speed** | Medium (~50ms per query) |
| **Overfit Gap** | Variable (depends on data volume) |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| ≥ 100 total examples, ≥ 8 per entity | **Use BERT** |
| High-stakes accuracy critical (legal, medical, financial) | **Use BERT** |
| Complex pattern recognition (sarcasm, nuanced sentiment) | **Use BERT** |
| GPU resources available | **Use BERT** |
| Inference speed is not critical | **Use BERT** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| < 100 total examples | Use **Full SetFit** (better few-shot performance) |
| < 8 examples per entity | Use **Full SetFit** |
| Need fast inference | Use **Full SetFit** (~10ms vs ~50ms) |
| Resource-constrained environment | Use **Full SetFit** or **Head-Only** |
| No GPU available | Use **Full SetFit** (CPU-friendly) |
| Training time must be < 3 min | Use **Full SetFit** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | ~5 minutes (100 entities, 50 examples) |
| **Inference speed** | Medium (~50ms per query) |
| **Memory usage** | High (full transformer model) |
| **GPU required** | Recommended (not required) |
| **Data requirement** | ≥ 8 examples per entity, ≥ 100 total |
| **Accuracy range** | 88-98% on typical datasets |

### Available BERT Models

| Model | Use Case | Accuracy | Speed |
|-------|----------|----------|-------|
| `distilbert` (default) | General purpose | High | Medium |
| `deberta-v3` | Maximum accuracy | Highest | Slow |
| `tinybert` | Resource-constrained | Medium | Fast |
| `bert-multilingual` | Multilingual text | High | Medium |

### Strengths

- **Superior accuracy**: Often 3-5% better than SetFit on complex tasks
- **Works with smaller datasets**: Effective with 8-16 examples per class
- **State-of-the-art architecture**: Full transformer with self-attention
- **Complex pattern recognition**: Better at sarcasm, nuance, contextual cues

### Weaknesses

- **Slower training**: ~5 minutes vs ~3 minutes for SetFit full
- **Slower inference**: ~50ms vs ~10ms for SetFit
- **Higher compute cost**: Full transformer pass required per query
- **Larger model files**: Hundreds of MB on disk
- **Overfitting risk**: With insufficient data, may memorize training examples

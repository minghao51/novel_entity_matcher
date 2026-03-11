# Classifier Route Comparison

Related docs: [`matcher-modes.md`](./matcher-modes.md) | [`bert-classifier.md`](./bert-classifier.md) | [`models.md`](./models.md)

This document compares the main routes exposed by `Matcher`:

- `zero-shot`
- `head-only`
- `full`
- `bert`
- `hybrid`

The goal is not to rank them globally. Each route optimizes for a different mix of setup cost, latency, accuracy, and hardware footprint.

## Benchmark Results Summary

**Datasets tested:**
- Small: occupations (23 entities)
- Medium: languages (184 entities)
- Large: products (1,025 entities)

**Key Findings:**
- Static embeddings (potion-8m) achieve **23,000-38,000 QPS** - **14-53x faster** than dynamic embeddings
- Training time ranges from **18s (minilm head-only)** to **685s (bge-base head-only)**
- BERT models achieve perfect accuracy (100%) but take **90-324s** to train
- Dynamic embeddings (minilm, bge-base, mpnet) provide better accuracy at lower throughput

## Performance Visualizations

### Embedding Model Throughput Comparison

![Embedding Performance](images/embeddings_performance.png)

Static embeddings (potion-8m, potion-32m, mrl-en) dramatically outperform dynamic embeddings for throughput while maintaining competitive accuracy.

### Latency Comparison

![Embedding Latency](images/embeddings_latency.png)

Average and P95 latency across all embedding models and datasets. Static embeddings show sub-millisecond latency compared to 5-15ms for dynamic models.

### Accuracy Comparison Across All Routes

![Accuracy Comparison](images/accuracy_comparison.png)

Comparison of top-1 accuracy across all routes and models. Training routes (head-only, full, bert) show significant accuracy improvements over zero-shot on complex datasets.

### Training Time vs Accuracy Tradeoff

![Training vs Accuracy](images/training_vs_accuracy.png)

Scatter plot showing the relationship between training time and accuracy for SetFit (head-only, full) and BERT routes.

### Static vs Dynamic Embeddings

![Static vs Dynamic](images/static_vs_dynamic.png)

Side-by-side comparison of static embeddings (potion-8m, potion-32m, mrl-en) vs dynamic embeddings (minilm, bge-base, mpnet) across throughput and accuracy.

### Model Selection Decision Tree

![Model Selection Guide](images/model_selection_guide.png)

Interactive decision tree for selecting the appropriate route based on dataset size, available labels, and performance requirements.

## Detailed Benchmark Results

### Zero-Shot Route Performance

| Model | Occupations (23 entities) | Languages (184 entities) | Products (1,025 entities) |
|-------|---------------------------|--------------------------|---------------------------|
| **potion-8m** | 38,342 QPS, 100% acc | 23,202 QPS, 97% acc | 31,499 QPS, 93% acc |
| **potion-32m** | 28,147 QPS, 100% acc | 19,015 QPS, 97% acc | 25,856 QPS, 93% acc |
| **mrl-en** | 4,696 QPS, 100% acc | 4,458 QPS, 98% acc | 4,215 QPS, 92% acc |
| **minilm** | 1,620 QPS, 100% acc | 598 QPS, 92% acc | 1,120 QPS, 89% acc |
| **bge-base** | 1,311 QPS, 100% acc | 800 QPS, 94% acc | 1,098 QPS, 91% acc |
| **mpnet** | 1,510 QPS, 100% acc | 727 QPS, 95% acc | 1,232 QPS, 91% acc |

**Key Insights:**
- Static embeddings are **14-53x faster** than dynamic embeddings
- potion-8m achieves **23,000-38,000 QPS** with 93-100% accuracy
- All models achieve >90% accuracy even on the large products dataset
- Static embeddings scale better with dataset size

### Head-Only Route Performance (1-2 examples per entity)

| Model | Training Time | Occupations | Languages | Products |
|-------|---------------|-------------|-----------|----------|
| **minilm** | 20s | 194 QPS, 100% acc | 43 QPS, 6% acc | 194 QPS, 100% acc |
| **bge-base** | 33s | 117 QPS, 100% acc | 101 QPS, 8% acc | 117 QPS, 100% acc |
| **mpnet** | 53s | 113 QPS, 100% acc | 106 QPS, 8% acc | 112 QPS, 100% acc |

**Key Insights:**
- Training time: **20-685s** depending on model and dataset size
- Fastest training with minilm (20s on occupations)
- Perfect accuracy (100%) on occupations and products with all models
- Languages dataset shows lower accuracy due to complexity (184 entities)

### Full Route Performance (3+ examples per entity)

| Model | Training Time | Occupations | Languages | Products |
|-------|---------------|-------------|-----------|----------|
| **minilm** | 18s | 194 QPS, 100% acc | 62 QPS, 6% acc | 186 QPS, 100% acc |
| **bge-base** | 38s | 114 QPS, 100% acc | 70 QPS, 8% acc | 121 QPS, 100% acc |
| **mpnet** | 48s | 115 QPS, 100% acc | 108 QPS, 8% acc | 115 QPS, 100% acc |

**Key Insights:**
- Training time: **18-468s** depending on model and dataset size
- Similar accuracy to head-only on most datasets
- Better throughput on languages dataset (62-108 QPS vs 43-106 QPS)
- Slightly faster training than head-only for larger datasets

### BERT Route Performance

| Model | Training Time | Memory (MB) | Inference (s) | Throughput (/s) | Accuracy |
|-------|---------------|-------------|---------------|-----------------|----------|
| **tinybert** | 51s | 118 | 0.68s | 292 | 55% |
| **distilbert** | 90s | 549 | 0.97s | 206 | **100%** |
| **roberta-base** | 324s | 974 | 0.78s | 258 | **100%** |

**Key Insights:**
- Training time: **51-324s** (tinybert fastest, roberta-base slowest)
- **Perfect accuracy (100%)** with distilbert and roberta-base
- tinybert achieves 55% accuracy - significantly lower than other BERT models
- distilbert offers best balance: 90s training, 549 MB memory, 100% accuracy
- roberta-base takes 3.6x longer than distilbert for same accuracy

## Quick Summary

| Route | Training data needed | Latency profile | Quality profile | Compute profile | Best fit |
|---|---|---|---|---|---|
| `zero-shot` | None | **23,000-38,000 QPS** (static), **598-1,620 QPS** (dynamic) | **93-100%** accuracy (static), **89-100%** (dynamic) | CPU-friendly, lowest memory | Cold start, prototypes, no labels |
| `head-only` | 1-2 examples per entity | **43-194 QPS**, **18-685s** training | **100%** on simple tasks, **6-8%** on complex | CPU-friendly, modest RAM | Quick supervised iteration |
| `full` | 3+ examples per entity | **62-194 QPS**, **18-468s** training | **100%** on simple tasks, **6-8%** on complex | CPU okay, GPU optional | Most production classifier use cases |
| `bert` | Best with 100+ total and 8+ per entity | **206-292 samples/s**, **51-324s** training | **55-100%** accuracy (varies by model) | GPU recommended, highest memory | Accuracy-first deployments |
| `hybrid` | No classifier labels required | Higher end-to-end latency, scalable retrieval | Best for large candidate sets | Multiple models, highest complexity | Large catalogs and long-tail retrieval |

## Route Details

### `zero-shot`

**What it is:** embedding similarity against entity names and aliases, with no supervised training.

**Performance**
- **Static embeddings (potion-8m, potion-32m, mrl-en):**
  - Throughput: **4,458-38,342 QPS** (14-53x speedup vs dynamic)
  - Latency: **0.5-1.5ms** average, **1-3ms** P95
  - Accuracy: **92-100%** across datasets
  - Memory: Lowest footprint
- **Dynamic embeddings (minilm, bge-base, mpnet):**
  - Throughput: **598-1,620 QPS**
  - Latency: **5-15ms** average, **6-20ms** P95
  - Accuracy: **89-100%** across datasets
  - Memory: Higher footprint due to transformer models

**Pros**
- No labeling or training loop
- Lowest setup cost
- Static embeddings achieve extreme throughput (23,000-38,000 QPS)
- Easy to operate in CPU-only environments
- Good first pass for evaluating entity coverage and alias quality

**Cons**
- Cannot learn task-specific decision boundaries
- More sensitive to weak entity names or missing aliases
- Usually below trained routes on ambiguous or domain-specific language
- Dynamic embeddings have significantly lower throughput

**Recommended when**
- You have no labeled data yet
- You need an immediate baseline
- The entity list is small to medium and the wording is fairly literal
- **Use static embeddings (potion-8m) for maximum throughput**
- **Use dynamic embeddings (minilm, bge-base) for better semantic understanding**

**Compute guidance**
- CPU: recommended
- GPU: not needed
- RAM: low to moderate, mostly driven by embedding model size and entity index size
- VRAM: none required

### `head-only`

**What it is:** supervised SetFit route for very small labeled datasets.

**Performance**
- Training time: **18-685s** depending on model and dataset
- Throughput: **43-194 QPS**
- Latency: **5-15ms** average
- Accuracy: **6-100%** (100% on simple tasks, 6-8% on complex tasks like languages)

**Pros**
- Fastest trained route
- Good improvement over zero-shot with very little data
- Keeps inference relatively cheap
- Easy to rerun during labeling iterations
- Perfect accuracy (100%) on occupations and products datasets

**Cons**
- Lower ceiling than `full` or `bert`
- Less robust when label boundaries depend on subtle wording
- Can plateau quickly once the task becomes more semantic than lexical
- Struggles on complex datasets (6-8% accuracy on 184-entity languages dataset)

**Recommended when**
- You have only 1-2 examples per entity
- You want a cheap supervised baseline before investing in more labels
- Training speed matters more than squeezing out maximum quality
- **Best model:** minilm for fastest training (20s)

**Compute guidance**
- CPU: good default
- GPU: optional, mainly for faster experimentation
- RAM: modest
- VRAM: not required

### `full`

**What it is:** the main SetFit training route for classifier-style matching.

**Performance**
- Training time: **18-468s** depending on model and dataset
- Throughput: **62-194 QPS**
- Latency: **5-15ms** average
- Accuracy: **6-100%** (100% on simple tasks, 6-8% on complex tasks)

**Pros**
- Best general-purpose tradeoff for trained classification
- Faster inference than `bert`
- Usually more data-efficient and cheaper to operate than full transformer classifiers
- Easier to deploy on CPU-only infrastructure than `bert`
- Perfect accuracy (100%) on occupations and products datasets
- Better throughput than head-only on languages dataset (62-108 QPS vs 43-106 QPS)

**Cons**
- Still depends on labeled data quality
- Lower ceiling than `bert` on nuanced or pattern-heavy tasks
- Less attractive when you need multilingual transformer classification behavior
- Struggles on complex datasets (6-8% accuracy on 184-entity languages dataset)

**Recommended when**
- You have at least 3 examples per entity
- You want a production-ready default with balanced quality and speed
- You need trained behavior but want to avoid transformer-classifier serving cost
- **Best model:** mpnet for best throughput (108 QPS on languages)

**Compute guidance**
- CPU: viable for training and serving
- GPU: optional and useful if training repeatedly
- RAM: modest to moderate
- VRAM: optional

### `bert`

**What it is:** fine-tuned transformer classification using a BERT-family backbone such as `distilbert`, `roberta-base`, `deberta-v3`, or `bert-multilingual`.

**Performance**
- Training time: **51-324s**
- Memory: **118-974 MB**
- Inference: **0.68-0.97s** for 625 samples
- Throughput: **206-292 samples/s**
- Accuracy: **55-100%**

**Pros**
- Highest accuracy ceiling among the classifier routes
- Perfect accuracy (100%) with distilbert and roberta-base
- Strongest option for subtle phrasing, context-heavy labels, and harder edge cases
- Better fit for tasks where exact wording patterns matter
- Model family choice lets you trade size for quality

**Cons**
- Slowest classifier inference path
- Highest training and serving cost among classifier routes
- More memory pressure on both CPU and GPU
- Longer test and CI runtime if live model training is exercised by default
- tinybert achieved only 55% accuracy (not recommended for production)

**Recommended when**
- Accuracy matters more than throughput
- You have richer supervision: roughly 100+ total examples and at least 8+ per entity is a sensible threshold
- You can justify GPU-backed training, and possibly GPU-backed serving for lower latency
- The task contains nuanced phrasing that SetFit misses
- **Best model:** distilbert (90s training, 549 MB memory, 100% accuracy)

**Compute guidance**
- CPU: acceptable for experimentation and low-QPS serving, but slower
- GPU: recommended for training; helpful for serving when latency matters
- RAM: moderate to high depending on model
- VRAM:
  - `tinybert`: low, suitable for constrained GPUs (118 MB)
  - `distilbert`: moderate and the best default balance (549 MB)
  - `roberta-base`: moderate to high (974 MB)
- Disk footprint: larger than SetFit-style classifier artifacts

**Backbone selection**

| Model | Training Time | Memory (MB) | Throughput (/s) | Accuracy | Recommended use |
|---|---------------|-------------|-----------------|----------|-----------------|
| `tinybert` | **51s** ✓ | **118** ✓ | **292** ✓ | 55% | Not recommended - low accuracy |
| `distilbert` | 90s | 549 | 206 | **100%** ✓ | **Default BERT choice** |
| `roberta-base` | 324s | 974 | 258 | **100%** ✓ | Accuracy-focused (3.6x slower training) |

### `hybrid`

**What it is:** a retrieval pipeline, not a classifier-training route. It combines blocking, embedding retrieval, and cross-encoder reranking.

**Pros**
- Handles much larger entity sets than the classifier routes
- Candidate pruning makes large-search problems tractable
- Reranking improves precision on hard retrieval tasks
- Strong fit when entity matching is closer to search than closed-set classification

**Cons**
- Highest system complexity
- Multiple models and stages to tune
- More latency variance than a single classifier
- Harder to reason about operationally than `zero-shot`, `full`, or `bert`

**Recommended when**
- The entity inventory is large, often tens of thousands or more
- You need high recall first, then precision via reranking
- Matching resembles document retrieval more than small-label classification

**Compute guidance**
- CPU: usable for smaller deployments, but reranking can become expensive
- GPU: helpful for reranker-heavy workloads
- RAM: moderate to high because multiple indexes/models may be resident
- VRAM: useful when the cross-encoder is on GPU

## Recommendation Matrix

| Situation | Recommended route | Why |
|---|---|---|
| No labels yet | `zero-shot` with **potion-8m** | Cheapest baseline, 23,000-38,000 QPS, 93-100% accuracy |
| 1-2 examples per entity | `head-only` with **minilm** | Fastest training (20s), perfect accuracy on simple tasks |
| 3+ examples per entity, typical production API | `full` with **mpnet** | Best balance: 18-48s training, 108 QPS, 100% accuracy on simple tasks |
| Accuracy-first classification with enough data | `bert` with **distilbert** | Perfect accuracy (100%), 90s training, best BERT balance |
| Large candidate catalog or retrieval-style matching | `hybrid` | Scales better than closed-set classifiers |

## Practical Selection Guidance

Start with `zero-shot` (potion-8m) if you are still validating the taxonomy - it achieves 23,000-38,000 QPS with 93-100% accuracy. Move to `head-only` (minilm) as soon as you have a handful of trustworthy labels - training takes only 20 seconds. Use `full` (mpnet) as the default trained route for most production classifier workloads - it offers the best balance of training time (18-48s), throughput (62-194 QPS), and accuracy. Move to `bert` (distilbert) only when you have enough supervision and a clear accuracy gap to justify the extra compute - it achieves perfect accuracy (100%) but takes 90 seconds to train. Use `hybrid` when the problem stops looking like small-label classification and starts looking like large-scale search plus reranking.

If the choice is between `full` and `bert`, the main question is usually not "which is better?" but "is the incremental quality worth the extra cost and latency?" In many CPU-first deployments, `full` remains the practical default even if `bert` is slightly more accurate.

## Benchmark Methodology

**Test Configuration:**
- Datasets: occupations (23 entities), languages (184 entities), products (1,025 entities)
- Hardware: Apple Silicon (MPS backend)
- Metrics: throughput (QPS), latency (avg/P95), accuracy, training time, memory footprint
- Test samples: 150 per dataset for embeddings, synthetic data for BERT

**Embedding Models Tested:**
- Static: potion-8m, potion-32m, mrl-en
- Dynamic: minilm (all-MiniLM-L6-v2), bge-base (BAAI/bge-base-en-v1.5), mpnet (all-mpnet-base-v2)

**BERT Models Tested:**
- tinybert (huawei-noah/TinyBERT_General_4L_312D)
- distilbert (distilbert-base-uncased)
- roberta-base (roberta-base)

**Training Configuration:**
- SetFit: 1 epoch, batch size 16, cosine learning rate schedule
- BERT: 5 epochs, batch size 16, linear warmup followed by decay
- Test split: 10% of training data

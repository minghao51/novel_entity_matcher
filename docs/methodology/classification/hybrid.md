# Hybrid Mode

**Phase 1: Classification** | **Large-scale mode** | **Benchmark: 90-95% accuracy**

---

## Mathematical Formulation

### Stage 1: Blocking (Candidate Filtering)

**BM25 Blocking:**

```
score_BM25(d, q) = Σ_{t ∈ q ∩ d} IDF(t) · (tf(t, d) · (k1 + 1)) / (tf(t, d) + k1 · (1 - b + b · |d|/avgdl))

where:
  t = term
  d = document (entity name/description)
  q = query
  IDF(t) = log((N - n(t) + 0.5) / (n(t) + 0.5) + 1)
  tf(t, d) = term frequency
  k1, b = tuning parameters (default: k1=1.5, b=0.75)
  |d| = document length
  avgdl = average document length

Output: top-k candidates by BM25 score
```

**TF-IDF Blocking:**

```
score_TFIDF(d, q) = Σ_{t ∈ q ∩ d} tf(t, d) · IDF(t)

Output: top-k candidates by TF-IDF cosine similarity
```

**Fuzzy Blocking:**

```
score_fuzzy(a, b) = LevenshteinSimilarity(a, b)
                  = 1 - edit_distance(a, b) / max(|a|, |b|)

Output: candidates with score ≥ fuzzy_threshold
```

### Stage 2: Retrieval (Embedding Similarity)

```
For each candidate c from blocking stage:
  c_embedding = f(c.name)
  q_embedding = f(query)

  score(c) = cosine_similarity(q_embedding, c_embedding)

Output: top-k candidates by embedding similarity
```

### Stage 3: Reranking (Cross-Encoder)

```
For each candidate c from retrieval stage:
  score(c) = CrossEncoder(query, c.name)

CrossEncoder(query, candidate) = sigmoid(MLP([CLS] output))

Output: final ranked list by cross-encoder score
```

### Full Pipeline

```
10,000 entities
    │
    ▼ [Blocking: BM25/TF-IDF/Fuzzy]
1,000 candidates (blocking_top_k)
    │
    ▼ [Retrieval: Embedding Similarity]
50 candidates (retrieval_top_k)
    │
    ▼ [Reranking: Cross-Encoder]
5 final results (final_top_k)
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Query text: str                                           │
│  • Entity definitions: List[dict] (id, name, aliases, desc)  │
│  • Blocking strategy: BM25Blocking, TFIDFBlocking,           │
│    FuzzyBlocking, NoOpBlocking                               │
│  • Reranker model: bge-m3 (default), bge-large, ms-marco     │
│  • Pipeline parameters: blocking_top_k, retrieval_top_k,     │
│    final_top_k                                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: BLOCKING                               │
├─────────────────────────────────────────────────────────────┤
│  Technique: BM25 / TF-IDF / Fuzzy Matching                   │
│  • Build inverted index over entity names/descriptions       │
│  • Score each entity against query using chosen strategy     │
│  • Select top-k candidates (blocking_top_k, default: 1000)   │
│  • Output: Candidate list (n_candidates × entity_info)       │
│                                                              │
│  Reduction: 10,000 entities → 1,000 candidates               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: RETRIEVAL                              │
├─────────────────────────────────────────────────────────────┤
│  Technique: Embedding Similarity (cosine)                    │
│  • Encode query and candidate entities                       │
│  • Compute cosine similarity for each candidate              │
│  • Rank by similarity score                                  │
│  • Select top-k candidates (retrieval_top_k, default: 50)    │
│  • Output: Ranked candidate list with similarity scores      │
│                                                              │
│  Reduction: 1,000 candidates → 50 candidates                 │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 3: RERANKING                              │
├─────────────────────────────────────────────────────────────┤
│  Technique: Cross-Encoder (bge-m3, bge-large, ms-marco)      │
│  • For each candidate, run cross-encoder on (query, text)    │
│  • Cross-encoder jointly processes query + candidate         │
│  • Produces precise relevance score                          │
│  • Rank by cross-encoder score                               │
│  • Select top-k results (final_top_k, default: 5)            │
│  • Output: Final ranked list with relevance scores           │
│                                                              │
│  Reduction: 50 candidates → 5 final results                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • MatchResult:                                              │
│    - matched: bool (any match above threshold)               │
│    - best_match: (entity_id, score) or None                  │
│    - top_k: List[(entity_id, score)]                         │
│    - confidence: float (best score)                          │
│    - stage_scores: dict (per-stage scores for debugging)     │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Pipeline Design

The hybrid mode uses a **coarse-to-fine** retrieval strategy:

1. **Blocking**: Fast, approximate filtering to reduce candidate set
2. **Retrieval**: Embedding-based similarity ranking on filtered candidates
3. **Reranking**: Precise but expensive cross-encoder scoring on top candidates

### Key Design Decisions

- **Progressive filtering**: Each stage reduces the candidate set by ~10-20x
- **Cheap to expensive**: Fast methods handle large sets, expensive methods handle small sets
- **Configurable top-k**: Each stage's output size can be tuned for speed/accuracy tradeoff

### Implementation Details

- **Routes to**: `HybridMatcher`
- **Blocking strategies**: BM25 (default), TF-IDF, Fuzzy, NoOp
- **Reranker models**: bge-m3 (default, multilingual), bge-large (higher accuracy), ms-marco (lightweight)
- **Parallel processing**: Supports `n_jobs=-1` for batch processing

---

## Findings

### Benchmark Performance

| Metric | Value |
|--------|-------|
| **Expected Accuracy** | 90-95% |
| **Training Time** | None (index build only) |
| **Inference Speed** | Medium (~50-100ms with reranking) |
| **Scalability** | 10k-100k+ entities |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| 10k+ entities | **Use Hybrid** |
| Need both speed and accuracy at scale | **Use Hybrid** |
| Large-scale production systems | **Use Hybrid** |
| Entity directory with many entries | **Use Hybrid** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| < 10k entities | Use **Full SetFit** (simpler, faster) |
| Need fastest possible inference | Use **Zero-Shot** with static embeddings |
| No entity descriptions available | Use **Full SetFit** (blocking needs text) |
| Simple entity names, no descriptions | Use **Full SetFit** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | None (index build only) |
| **Inference speed** | Medium (~50-100ms with reranking) |
| **Memory usage** | High (multiple models: BM25 index, embeddings, cross-encoder) |
| **GPU required** | No (but reranker benefits from GPU) |
| **Data requirement** | None (but entity descriptions help blocking) |
| **Scalability** | 10k-100k+ entities |

### Available Blocking Strategies

| Strategy | Method | Best For |
|----------|--------|----------|
| `BM25Blocking` (default) | Keyword-based | General purpose |
| `TFIDFBlocking` | Document similarity | Longer descriptions |
| `FuzzyBlocking` | Edit distance | Typos and variations |
| `NoOpBlocking` | No filtering | Small datasets |

### Available Reranker Models

| Model | Language | Accuracy | Speed |
|-------|----------|----------|-------|
| `bge-m3` (default) | Multilingual | High | Medium |
| `bge-large` | English | Highest | Slow |
| `ms-marco` | English | Medium | Fast |

### Strengths

- **Scales to very large datasets**: 10k-100k+ entities
- **High accuracy**: Cross-encoder reranking provides precise scoring
- **Efficient candidate pruning**: Each stage reduces search space
- **No training required**: Works with entity definitions only
- **Configurable**: Tune each stage for speed/accuracy tradeoff

### Weaknesses

- **Complex setup**: Multiple models and stages to configure
- **Higher memory usage**: BM25 index + embeddings + cross-encoder
- **Slower inference**: ~50-100ms vs ~10ms for SetFit
- **Requires entity descriptions**: Blocking works best with rich text
- **More tuning parameters**: blocking_top_k, retrieval_top_k, final_top_k

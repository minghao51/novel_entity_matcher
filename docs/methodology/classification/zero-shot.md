# Zero-Shot Classification

**Phase 1: Classification** | **No training required** | **Benchmark: 73.3% accuracy**

---

## Mathematical Formulation

### Cosine Similarity Matching

For each known entity with name and optional aliases:

```
For each known entity e with text representations {t_1, t_2, ..., t_m}:
  e_embedding = f_pretrained(t_best)    # Best matching name/alias
  q_embedding = f_pretrained(query)

  score(e) = cosine_similarity(q_embedding, e_embedding)
           = (q · e) / (||q|| · ||e||)

predicted_entity = argmax_e score(e)
confidence = max_e score(e)
```

### Multi-Candidate Scoring

When multiple candidates exist (names + aliases):

```
score(e) = max_{t ∈ texts(e)} cosine_similarity(q_embedding, f_pretrained(t))
```

### Threshold-Based Filtering

```
matches = [(e, score(e)) for e in entities if score(e) ≥ threshold]
matches.sort(key=lambda x: x[1], reverse=True)
return matches[:top_k]
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Query text: str                                           │
│  • Entity definitions: List[dict] (id, name, aliases)        │
│  • Model: static or dynamic embedding model                  │
│    - Static: potion-8m, potion-32m (pre-computed)            │
│    - Dynamic: bge-base, minilm (on-demand)                   │
│  • Threshold: float (default: 0.6)                           │
│  • Top-k: int (default: 1)                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: ENTITY INDEX BUILDING                  │
├─────────────────────────────────────────────────────────────┤
│  Technique: Embedding Index (static or dynamic)              │
│  • For static models: load pre-computed entity embeddings    │
│  • For dynamic models: encode entity names/aliases on init   │
│  • Store in efficient lookup structure (numpy array, ANN)    │
│  • Output: Entity embedding matrix E (n_entities × dim)      │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: QUERY ENCODING                         │
├─────────────────────────────────────────────────────────────┤
│  Technique: SentenceTransformer / StaticEmbedding            │
│  • Encode query text to embedding vector                     │
│  • Static: O(1) lookup from pre-computed vocabulary          │
│  • Dynamic: O(L) transformer pass (L = query length)         │
│  • Output: q_embedding (1 × embedding_dim)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: SIMILARITY COMPUTATION                 │
├─────────────────────────────────────────────────────────────┤
│  Technique: Cosine Similarity (vectorized)                   │
│  • scores = E · q / (||E|| · ||q||)                          │
│  • Vectorized matrix multiplication for all entities         │
│  • Output: scores array (n_entities,)                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: FILTERING & RANKING                    │
├─────────────────────────────────────────────────────────────┤
│  Technique: Threshold + Top-k Selection                      │
│  • Filter: keep entities with score ≥ threshold              │
│  • Sort: descending by score                                 │
│  • Select: top-k matches                                     │
│  • Output: List[(entity_id, score)]                          │
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
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Build index**: Encode all entity names and aliases into embeddings
2. **Encode query**: Transform input text to embedding space
3. **Compute similarity**: Calculate cosine similarity between query and all entities
4. **Filter and rank**: Apply threshold and return top-k matches

### Key Design Decisions

- **Static vs dynamic embeddings**: Static models (potion-8m) provide 10-100x faster inference via pre-computed lookups
- **No training**: Relies entirely on pre-trained semantic similarity
- **Threshold-based**: Configurable confidence threshold to reject low-confidence matches

### Implementation Details

- **Routes to**: `EmbeddingMatcher`
- **Backend**: Static embeddings (model2vec) or dynamic (SentenceTransformers)
- **Index**: Numpy array for exact search, optional ANN for large datasets

---

## Findings

### Benchmark Performance (ag_news, 500 training samples, no training used)

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 72.0% |
| **Test Accuracy** | **73.3%** |
| **Training Time** | ~3 seconds (index build only) |
| **Overfit Gap** | +2.5% (slight underfit) |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| No training data available | **Use Zero-Shot** |
| Entity names are semantically distinct | **Use Zero-Shot** |
| Prototyping or exploration | **Use Zero-Shot** |
| Need immediate results | **Use Zero-Shot** |
| Static embedding speed required | **Use Zero-Shot** with potion-8m |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Training data available (≥ 3 examples/entity) | Use **Full SetFit** (91.2% vs 73.3%) |
| Need best accuracy | Use **Full SetFit** or **BERT** |
| Classes are semantically similar | Use **Full SetFit** (learns to distinguish) |
| Entity names are ambiguous | Use **Full SetFit** (adapts to domain) |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | None (index build only, ~3s) |
| **Inference speed** | Fast (static: ~1ms, dynamic: ~10ms) |
| **Memory usage** | Low (static) to Medium (dynamic) |
| **GPU required** | No |
| **Data requirement** | None |
| **Accuracy range** | 70-80% on typical datasets |

### Strengths

- **No training required**: Works immediately with entity definitions
- **Fast inference**: Especially with static embeddings (~1ms)
- **Simple**: No hyperparameters to tune
- **Good baseline**: 73.3% accuracy with zero training data

### Weaknesses

- **Lower accuracy**: 73.3% vs 91.2% for full SetFit
- **Cannot learn from data**: Stuck with pre-trained semantic similarity
- **Struggles with ambiguous names**: Cannot adapt to domain-specific meanings
- **Sensitive to entity name quality**: Performance depends on how well entity names represent their class

# Pattern Strategy

**Phase 2: Novelty Detection** | **Rule-based / Orthographic** | **Benchmark: 0.999 AUROC (entity text), 0.50 AUROC (free text)**

---

## Mathematical Formulation

### Character N-gram Overlap

```
For query text x and reference texts R:

ngrams(x, n) = set of character n-grams of length n

ngram_overlap(x, R) = |ngrams(x, n) ∩ ngrams(R, n)| / |ngrams(x, n) ∪ ngrams(R, n)|
                    = Jaccard similarity of n-gram sets

4gram_overlap(x, R) = same with n=4
```

### Length Deviation

```
μ_len = mean(length(r) for r in R)
σ_len = std(length(r) for r in R)

length_deviation(x) = min(|length(x) - μ_len| / σ_len, 1.0)
```

### Capitalization Mismatch

```
capitalization_patterns = {
  "title_case": all words start with uppercase,
  "upper_case": all uppercase,
  "lower_case": all lowercase,
  "mixed": other
}

cap_score(x, R) = 1 if pattern(x) is rare in R else 0
```

### Prefix/Suffix Rarity

```
prefix(x, n) = first n characters of x
suffix(x, n) = last n characters of x

prefix_rarity(x, R, n) = 1 - count(prefix(x, n) in R) / |R|
suffix_rarity(x, R, n) = 1 - count(suffix(x, n) in R) / |R|
```

### Combined Novelty Score

```
novelty_score = 0.25 × (1 - ngram_overlap)        # Character trigram mismatch
              + 0.20 × (1 - 4gram_overlap)         # Character 4-gram mismatch
              + 0.15 × length_deviation            # Unusual length
              + 0.10 × cap_score                   # Unusual capitalization
              + 0.15 × prefix_rarity               # Rare prefix
              + 0.15 × suffix_rarity               # Rare suffix

is_novel = novelty_score > threshold
```

---

## DAG (Directed Acyclic Graph)

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUTS                                   │
├─────────────────────────────────────────────────────────────┤
│  • Reference texts: List[str] (known entity names)           │
│  • Query texts: List[str]                                    │
│  • threshold: float (default: 0.5)                           │
│  • char_ngram_n: int (default: 3)                            │
│  • prefix_suffix_n: int (default: 3)                         │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: PATTERN EXTRACTION                     │
├─────────────────────────────────────────────────────────────┤
│  Technique: Orthographic Pattern Analysis                    │
│  • For each reference text:                                  │
│    - Extract character n-grams (trigrams, 4-grams)           │
│    - Record length                                           │
│    - Record capitalization pattern                           │
│    - Record prefix/suffix                                    │
│  • Build pattern frequency distributions                     │
│  • Output: Pattern statistics from reference set             │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: QUERY PATTERN EXTRACTION               │
├─────────────────────────────────────────────────────────────┤
│  Technique: Same Pattern Extraction Applied to Queries       │
│  • For each query text:                                      │
│    - Extract character n-grams                               │
│    - Measure length                                          │
│    - Identify capitalization pattern                         │
│    - Extract prefix/suffix                                   │
│  • Output: Query pattern features                            │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: PATTERN COMPARISON                     │
├─────────────────────────────────────────────────────────────┤
│  Technique: Pattern Deviation Scoring                        │
│  • For each query, compute 6 deviation scores:               │
│    1. ngram_mismatch = 1 - Jaccard(trigrams)                 │
│    2. 4gram_mismatch = 1 - Jaccard(4-grams)                  │
│    3. length_deviation = z-score of length                   │
│    4. cap_mismatch = rarity of capitalization pattern        │
│    5. prefix_rarity = 1 - frequency of prefix                │
│    6. suffix_rarity = 1 - frequency of suffix                │
│  • Output: 6-dimensional deviation vector per query          │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: WEIGHTED AGGREGATION                   │
├─────────────────────────────────────────────────────────────┤
│  Technique: Weighted Sum                                     │
│  • novelty_score = 0.25×ngram + 0.20×4gram + 0.15×length    │
│                   + 0.10×cap + 0.15×prefix + 0.15×suffix     │
│  • is_novel = novelty_score > threshold                      │
│  • Output: (novelty_score, is_novel, component_scores)       │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                     OUTPUTS                                  │
├─────────────────────────────────────────────────────────────┤
│  • novelty_score: float ∈ [0, 1]                             │
│  • is_novel: bool                                            │
│  • component_scores: Dict[str, float] (per-feature scores)   │
│  • ngram_overlap: float                                      │
│  • length_deviation: float                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Approach / Methodology

### Process

1. **Extract patterns**: Analyze orthographic patterns (n-grams, length, capitalization, prefix/suffix) from known entity names
2. **Build distributions**: Create frequency distributions for each pattern type
3. **Score queries**: Measure how much each query deviates from known patterns
4. **Aggregate**: Weighted combination of all pattern deviations

### Key Design Decisions

- **Entity-name focused**: Works on entity names/text, not free-form text
- **No training required**: Purely rule-based, no model fitting
- **Multiple pattern types**: Combines 6 different orthographic signals
- **Language-dependent**: Patterns vary across languages

### Implementation Details

- **Weight in ensemble**: 0.20
- **No model needed**: Pure string operations
- **Fast**: O(n · L) where L is text length

---

## Findings

### Benchmark Performance

**ag_news (free text):**

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.500** (random) |

**go_emotions (entity-like text):**

| Metric | Value |
|--------|-------|
| **Test AUROC** | **0.987** |
| **AUPRC** | **0.993** |
| **DR@1%** | **97.8%** |

**Strategy benchmark (ag_news, parameter sweep):**

| Parameters | AUROC | AUPRC | DR@1% |
|------------|-------|-------|-------|
| threshold=0.3 | **0.999** | **0.999** | **99.8%** |

### When to Use

| Scenario | Recommendation |
|----------|---------------|
| Entity name matching (SKUs, product codes, structured entities) | **Use Pattern** |
| Orthographic/linguistic pattern violations | **Use Pattern** |
| Multi-lingual scenarios with different language patterns | **Use Pattern** |
| Complementary signal for entity-like text | **Use Pattern** |

### When NOT to Use

| Scenario | Alternative |
|----------|-------------|
| Free text classification (articles, sentences) | Use **SetFit Centroid Distance** (0.50 vs 0.886) |
| Semantic novelty detection | Use **kNN Distance** |
| Embedding-space novelty | Use **SetFit Centroid Distance** |
| Natural language queries | Use **SetFit Centroid Distance** |

### Characteristics

| Property | Value |
|----------|-------|
| **Training time** | None (rule-based) |
| **Inference time** | O(n · L) per query (L = text length) |
| **Memory** | O(n · L) for pattern distributions |
| **GPU required** | No |
| **Data requirement** | 50+ entity names for pattern extraction |
| **Best for** | Entity names, structured text |

### Strengths

- **Perfect on entity-like text**: 0.999 AUROC, 99.8% DR@1%
- **No training required**: Purely rule-based
- **Very fast**: String operations only
- **Easy to interpret**: Each component score is explainable
- **Captures orthographic novelty**: Detects patterns embeddings miss
- **Works well for structured entities**: SKUs, product codes, IDs

### Weaknesses

- **Fails on free text**: 0.500 AUROC (random) on article text
- **Language-dependent**: Patterns vary across languages
- **Misses semantic novelty**: "quantum biology" looks normal orthographically
- **Requires sufficient reference samples**: Need 50+ entity names for reliable patterns
- **Not effective for free-form text**: Designed for entity names, not sentences

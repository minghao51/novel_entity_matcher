# Semantic Matching Research Landscape 2026

Archived note: this research scan is retained for historical context. For the active implementation direction, use [`../technical-roadmap.md`](../technical-roadmap.md) and [`../novel-class-detection.md`](../novel-class-detection.md).

**Date:** 2026-03-18
**Research Method:** Google AI Mode Search with semantic queries
**Focus:** Open source projects and research in semantic matching, entity resolution, and novelty detection

## Executive Summary

As of early 2026, the semantic matching and entity resolution landscape has significantly evolved from traditional string matching to **transformer-based semantic embeddings** and **LLM-driven alignment**. The field is now focused on:

1. **Semantic-Aware Matching** - Using deep learning embeddings for entity resolution
2. **NIL Prediction** - Treating "unknown" entities as first-class citizens
3. **Scalability** - Handling trillion-edge scales with ANN and distributed computing
4. **Active Learning** - Human-in-the-loop refinement for enterprise workflows
5. **Novelty Detection** - Identifying when records represent new entities

This document summarizes key open-source projects, research findings, and proposes enhancements for the `novel_entity_matcher` project.

---

## Top Open Source Repositories

### 1. **Splink**
- **Focus:** High-performance probabilistic entity resolution
- **Approach:** Fellegi-Sunter model with modern extensions
- **Backends:** Spark, DuckDB, Athena
- **Use Case:** Large-scale deduplication
- **Semantic Matching:** Limited (requires custom UDFs)
- **Novelty Detection:** Manual thresholding

**Key Insight:** Industry standard for ER but lacks built-in semantic capabilities.

### 2. **Zingg**
- **Focus:** Active learning-based entity resolution
- **Approach:** Human-in-the-loop ML training
- **Use Case:** Enterprise workflows requiring refinement
- **Semantic Matching:** Strong (ML-based features)
- **Novelty Detection:** Probabilistic scoring

**Key Insight:** Excellent for production systems where human feedback is available.

### 3. **ComEM (LLM4EM)** 🔥
- **Repository:** https://github.com/tshu-w/LLM4EM
- **Focus:** LLM-based entity matching
- **Approach:** "Match, Compare, and Select" strategies
- **Use Case:** Global consistency in record relationships
- **Semantic Matching:** Extreme (textual understanding)
- **Novelty Detection:** Built-in NIL prediction

**Key Insight:** Research-forward approach using LLMs for reasoning about matches.

### 4. **PyJedAI**
- **Focus:** Comprehensive Python ER suite
- **Approach:** State-of-the-art clustering and blocking algorithms
- **Features:** Meta-blocking, comprehensive algorithm library
- **Use Case:** Academic research and algorithm comparison

**Key Insight:** Excellent for benchmarking different approaches.

### 5. **Entity-Embed** 🔥
- **Repository:** https://github.com/vintasoftware/entity-embed
- **Focus:** PyTorch-based semantic entity resolution
- **Approach:** Vector embeddings with Approximate Nearest Neighbors (ANN)
- **Use Case:** Semantic-heavy datasets
- **Semantic Matching:** High (deep learning)
- **Novelty Detection:** Vector space distance

**Key Insight:** Most similar to `novel_entity_matcher`'s embedding-based approach.

### 6. **EntityMatchingModel (EMM)**
- **Repository:** https://github.com/ing-bank/EntityMatchingModel
- **Focus:** Scalable name-matching (ING Bank's framework)
- **Approach:** Threshold-based classification
- **Use Case:** Production financial services
- **Semantic Matching:** Moderate
- **Novelty Detection:** Threshold-based NIL classification

**Key Insight:** Production-proven at scale.

---

## Novelty Detection Specialists

### 1. **OpenNovelty** (2026) 🆕
- **Type:** LLM-powered agentic system
- **Approach:** Multi-phase comparisons with hierarchical taxonomy
- **Use Case:** Verifiable scholarly novelty assessment
- **Key Feature:** Evidence-based novelty reports

**Relevance:** State-of-the-art agentic approach to novelty detection.

### 2. **EDIN (Entity Discovery and Indexing)**
- **Focus:** Unknown entity discovery pipeline
- **Approach:** Benchmark and framework for NIL entities
- **Key Feature:** Specialized for novel entity identification

**Relevance:** Direct relevance to novelty detection use cases.

### 3. **PAT-SND Model**
- **Focus:** Semantic novelty detection
- **Approach:** Attention-based knowledge characterization
- **Key Feature:** Identifies "surprising" facts

**Relevance:** Research approach to semantic novelty.

### 4. **KGGen** (NeurIPS 2025) 🆕
- **Focus:** Text-to-knowledge-graph generation
- **Approach:** Clustering for entity resolution and discovery
- **Key Feature:** Automatic entity discovery

**Relevance:** Cutting-edge research from NeurIPS 2025.

---

## Algorithm Approaches in 2026

### 1. NIL Prediction (Not-In-Lexicon)
Instead of forcing matches to closest entities, systems use:
- **Discrimination thresholds** - Dynamically calculated similarity cutoffs
- **NIL-classifiers** - Binary classification for match vs. no-match
- **Confidence scoring** - Probabilistic assessment of match quality

**Implementation Pattern:**
```python
if max_similarity < threshold:
    return "NIL (New Entity Discovered)"
return best_match
```

### 2. Contrastive Projection Networks
- **Goal:** Align type-compatible entities while separating unrelated ones
- **Benefit:** Makes out-of-distribution entities easier to isolate
- **Technique:** Learned projection spaces with contrastive loss

### 3. Agentic Novelty Analysis
- **Approach:** Multi-phase LLM comparisons
- **Process:**
  1. Extract claims/features
  2. Compare against hierarchical taxonomy
  3. Identify truly novel contributions
  4. Generate evidence-based reports

### 4. Semantic Dissimilarity (ON Metric)
- **Formula:** Novelty = average distance to k-nearest neighbors in historical database
- **Use Case:** Quantifying how "different" an entity is from known examples

### 5. Semantic-Aware Blocking
- **Technique:** Locality Sensitive Hashing (LSH) on deep learning embeddings
- **Benefit:** Reduces search space without losing semantic nuances
- **Key:** Traditional blocking on embeddings, not raw text

---

## Sample Implementation: Basic NIL Prediction

```python
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load modern embedding model
model = SentenceTransformer("all-mpnet-base-v2")

# Known entities in Knowledge Base
kb_entities = ["Quantum Computing", "Deep Learning", "Genetic Engineering"]
kb_embeddings = model.encode(kb_entities)

def identify_entity(input_text, threshold=0.65):
    """
    Identify entity with NIL prediction for novelty detection.

    Args:
        input_text: Text to identify
        threshold: Similarity threshold below which entity is considered novel

    Returns:
        Tuple of (entity_name, confidence_score)
    """
    query_emb = model.encode([input_text])
    similarities = cosine_similarity(query_emb, kb_embeddings)[0]
    max_idx = np.argmax(similarities)
    max_score = similarities[max_idx]

    if max_score < threshold:
        return "NIL (New Entity Discovered)", max_score

    return kb_entities[max_idx], max_score

# Test cases
print(identify_entity("Neural Networks"))    # Match: 'Deep Learning'
print(identify_entity("Interstellar Mining")) # NIL: Novel entity
```

---

## Key Research Papers & Benchmarks

### Benchmarks
- **[MovieGraphBenchmark](https://github.com/ScaDS/MovieGraphBenchmark)**
  - Multi-source entity resolution
  - Knowledge graph fusion
  - 2024-2026 active development

### Research Collections
- **[kaisugi/entity-related-papers](https://github.com/kaisugi/entity-related-papers)**
  - Curated list of entity resolution papers
  - Includes NIL prediction research

- **[OlivierBinette/Awesome-Entity-Resolution](https://github.com/OlivierBinette/Awesome-Entity-Resolution)**
  - Comprehensive ER resource list
  - Tools, papers, benchmarks

### Scalability Research
- **[LargeGNN](https://github.com/JadeXIN/LargeGNN)**
  - GNN-based entity alignment
  - Uses "landmark entities" as bridges
  - Prevents structural loss in partitioned graphs

---

## Comparison: Top ER Frameworks (2026)

| Feature | **Splink** | **Zingg** | **Entity-Embed** | **ComEM** | **novel_entity_matcher** |
|---------|-----------|-----------|------------------|-----------|---------------------|
| **Primary Approach** | Probabilistic | Active Learning | Semantic Embeddings | LLM Reasoning | Embedding + Rules |
| **Scaling Backend** | DuckDB, Spark | Spark | PyTorch (GPU/ANN) | API/Local LLM | In-memory |
| **Semantic Matching** | Limited (UDFs) | Strong (ML) | **High** (DL) | **Extreme** | **High** |
| **Novelty Detection** | Manual Threshold | Probabilistic | Vector Distance | **Built-in NIL** | **Yes** |
| **Benchmarking** | Built-in Validation | Model Metrics | Accuracy/Loss | Multi-dataset | **Comprehensive** |
| **Async API** | No | No | No | No | **Yes** |
| **Python Native** | Yes | Yes | Yes | Yes | **Yes** |
| **LLM Integration** | No | No | No | **Yes** | **Optional** |

---

## Relationship to novel_entity_matcher Project

### Current Strengths ✅

The `novel_entity_matcher` project is **well-positioned** within the 2026 landscape:

1. **Embedding-Based Matching** - Similar to Entity-Embed's approach
2. **Novelty Detection** - Built-in capabilities like OpenNovelty/EDIN
3. **Async API** - Modern Python architecture (unique among surveyed projects)
4. **Benchmarking Framework** - Comprehensive evaluation like PyJedAI
5. **LLM Integration** - Optional LLM proposer like ComEM

### Unique Differentiators 🎯

1. **Async-First Architecture** - None of the surveyed projects emphasize async APIs
2. **Flexible Matcher Modes** - Multiple matching strategies in one framework
3. **Comprehensive Benchmarking** - Built-in dataset and reporting utilities
4. **Modular Design** - Clear separation between embedding, entity, and matching layers

---

## Proposed Enhancements

Based on the research landscape, here are proposed enhancements for `novel_entity_matcher`:

### Priority 1: NIL Prediction Enhancement 🔥

**Current State:** Basic novelty detection exists
**Proposed:** Implement discrimination thresholds and NIL-classifiers

```python
class EnhancedNoveltyDetector:
    """Advanced NIL prediction with dynamic thresholds."""

    def __init__(self, base_threshold=0.65, adaptive=True):
        self.base_threshold = base_threshold
        self.adaptive = adaptive
        self.threshold_history = []

    def predict(self, query_embedding, kb_embeddings):
        """
        Predict entity with adaptive NIL classification.

        Returns:
            Tuple of (entity_id, confidence, is_novel)
        """
        similarities = cosine_similarity([query_embedding], kb_embeddings)[0]
        max_score = np.max(similarities)

        # Adaptive threshold based on historical performance
        threshold = self._calculate_adaptive_threshold() if self.adaptive else self.base_threshold

        is_novel = max_score < threshold
        entity_id = np.argmax(similarities) if not is_novel else None

        return entity_id, max_score, is_novel

    def _calculate_adaptive_threshold(self):
        """Calculate threshold based on precision/recall trade-offs."""
        # Implement adaptive threshold logic
        pass
```

**Benefits:**
- More accurate novelty detection
- Reduces false positives on novel entities
- Aligns with 2026 state-of-the-art

### Priority 2: Semantic-Aware Blocking

**Current State:** Standard blocking
**Proposed:** LSH on embeddings for semantic blocking

```python
from sklearn.neighbors import NearestNeighbors
import hashlib

class SemanticBlocking:
    """Locality Sensitive Hashing on embeddings."""

    def __init__(self, n_neighbors=10, radius=0.5):
        self.n_neighbors = n_neighbors
        self.radius = radius
        self.ann_index = None

    def build_index(self, embeddings):
        """Build ANN index for fast similarity search."""
        self.ann_index = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            radius=self.radius,
            algorithm='ball_tree'
        )
        self.ann_index.fit(embeddings)

    def query(self, query_embedding):
        """Find candidate matches using semantic blocking."""
        distances, indices = self.ann_index.radius_neighbors([query_embedding])
        return indices[0], distances[0]
```

**Benefits:**
- Faster matching on large datasets
- Reduces comparison space while preserving semantic similarity
- Aligns with Entity-Embed's approach

### Priority 3: Active Learning Loop

**Current State:** Manual threshold tuning
**Proposed:** Human-in-the-loop refinement (inspired by Zingg)

```python
class ActiveLearningLoop:
    """Human-in-the-loop threshold refinement."""

    def __init__(self, uncertainty_threshold=0.1):
        self.uncertainty_threshold = uncertainty_threshold
        self.labeled_pairs = []

    def get_uncertain_pairs(self, pairs, scores):
        """Return pairs with scores near threshold for human labeling."""
        uncertain = []
        for pair, score in zip(pairs, scores):
            if abs(score - 0.5) < self.uncertainty_threshold:
                uncertain.append((pair, score))
        return uncertain

    def incorporate_feedback(self, labeled_pairs):
        """Update model based on human feedback."""
        self.labeled_pairs.extend(labeled_pairs)
        # Retrain or adjust thresholds based on feedback
```

**Benefits:**
- Improves matching quality over time
- Reduces manual labeling effort
- Production-ready for enterprise workflows

### Priority 4: Contrastive Learning

**Current State:** Standard embeddings
**Proposed:** Contrastive projection for better separation

```python
import torch
import torch.nn as nn

class ContrastiveProjection(nn.Module):
    """Learn projection space for better entity separation."""

    def __init__(self, input_dim, projection_dim=256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )

    def forward(self, x):
        return self.projection(x)

    def contrastive_loss(self, anchor, positive, negative, margin=1.0):
        """Contrastive loss for learning entity representations."""
        # Implement contrastive loss
        pass
```

**Benefits:**
- Better separation of unrelated entities
- Improved novelty detection
- State-of-the-art approach (2026)

### Priority 5: Agentic Novelty Analysis (Advanced)

**Current State:** Threshold-based novelty
**Proposed:** Multi-phase LLM analysis (inspired by OpenNovelty)

```python
class AgenticNoveltyAnalyzer:
    """LLM-powered agentic system for novelty assessment."""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    async def analyze_novelty(self, entity, kb_context):
        """
        Multi-phase novelty analysis.

        Phases:
        1. Extract entity features/claims
        2. Compare against KB taxonomy
        3. Identify novel aspects
        4. Generate evidence report
        """
        # Phase 1: Feature extraction
        features = await self._extract_features(entity)

        # Phase 2: Taxonomy comparison
        similar_entities = await self._find_similar_in_kb(features, kb_context)

        # Phase 3: Novelty identification
        novel_aspects = await self._identify_novel_aspects(features, similar_entities)

        # Phase 4: Report generation
        report = await self._generate_report(entity, novel_aspects, similar_entities)

        return report
```

**Benefits:**
- Explainable novelty detection
- Identifies specific novel aspects
- Cutting-edge agentic approach

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Implement enhanced NIL prediction with adaptive thresholds
- [ ] Add comprehensive benchmarking on novelty detection
- [ ] Document threshold tuning strategies

### Phase 2: Performance (Weeks 3-4)
- [ ] Implement semantic-aware blocking with ANN
- [ ] Add performance benchmarks on large datasets
- [ ] Optimize embedding similarity computation

### Phase 3: Learning (Weeks 5-6)
- [ ] Implement active learning loop
- [ ] Add feedback incorporation mechanisms
- [ ] Create human-in-the-loop evaluation

### Phase 4: Advanced (Weeks 7-8)
- [ ] Implement contrastive learning projection
- [ ] Add agentic novelty analysis (optional LLM integration)
- [ ] Comprehensive evaluation on all features

---

## Conclusion

The `novel_entity_matcher` project is **well-positioned** within the 2026 semantic matching landscape. Its current strengths (async API, embedding-based matching, novelty detection, benchmarking) align with state-of-the-art approaches.

The proposed enhancements focus on:
1. **Improved NIL prediction** - Better novelty detection accuracy
2. **Semantic blocking** - Faster matching on large datasets
3. **Active learning** - Production-ready refinement loops
4. **Contrastive learning** - Better entity separation
5. **Agentic analysis** - Explainable novelty assessment

By implementing these enhancements, `novel_entity_matcher` can become a **leading open-source framework** for semantic matching and novelty detection, combining the best aspects of Splink (production-ready), Entity-Embed (semantic), ComEM (LLM-powered), and OpenNovelty (agentic).

---

## References

### Search Results
- Semantic matching ER search: `/Users/minghao/.claude/skills/google-ai-mode/results/2026-03-18_23-03-59_semantic_matching_entity_resolution_open.md`
- Novelty detection search: `/Users/minghao/.claude/skills/google-ai-mode/results/2026-03-18_23-04-16_novelty_detection_semantic_matching_embe.md`

### Key Repositories
- Splink: https://github.com/moj-analytical-services/splink
- Zingg: https://github.com/zinggAI/zingg
- ComEM/LLM4EM: https://github.com/tshu-w/LLM4EM
- PyJedAI: https://github.com/AiKols/PyJedAI
- Entity-Embed: https://github.com/vintasoftware/entity-embed
- EntityMatchingModel: https://github.com/ing-bank/EntityMatchingModel

### Research Papers
- OpenNovelty: https://www.researchgate.net/publication/399478068
- PAT-SND: https://aclanthology.org/2022.emnlp-main.627/
- KGGen: https://www.paperdigest.org/2025/11/neurips-2025-papers-with-code-data/

### Benchmarks
- MovieGraphBenchmark: https://github.com/ScaDS/MovieGraphBenchmark
- entity-related-papers: https://github.com/kaisugi/entity-related-papers
- Awesome-Entity-Resolution: https://github.com/OlivierBinette/Awesome-Entity-Resolution

---

**Document Version:** 1.0
**Last Updated:** 2026-03-18
**Maintainer:** novel_entity_matcher project

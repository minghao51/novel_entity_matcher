# Hierarchical Entity Categorization Design

**Date:** 2026-03-04
**Status:** Approved
**Author:** Claude (Sonnet 4.5)

## Overview

Add a new `HierarchicalMatcher` module to enable hierarchical entity categorization with multi-parent support, hierarchy-aware confidence scoring, and flexible granularity matching. This design aligns with Phase 2-3 capabilities outlined in the alternative methods roadmap.

## Requirements

Based on user requirements, the system must support:

1. **Multi-parent hierarchies** - entities can belong to multiple taxonomies simultaneously
2. **Hierarchy-aware confidence** - scoring that accounts for relationship depth and strength
3. **Flexible granularity matching** - match queries at any level in the hierarchy
4. **Clean separation** - standalone module that doesn't modify existing matchers
5. **Performance** - efficient for large-scale hierarchical datasets

## Proposed Architecture

### High-Level Design

```
HierarchicalMatcher
├── HierarchyIndex (new)
│   ├── Multi-parent graph structure
│   ├── Ancestor/descendant queries
│   └── Relationship strength weights
├── HierarchicalScoring (new)
│   ├── Depth-aware confidence calculation
│   ├── Multi-parent aggregation
│   └── Granularity penalties
└── HierarchicalMatcher (new API)
    ├── match_at_level(query, level)
    ├── match_with_ancestors(query)
    ├── get_hierarchy_path(entity_id)
    └── batch_match_hierarchical(queries)
```

### Data Structure

**Hierarchical Entity Definition:**

```python
entities = [
    {
        "id": "DE",
        "name": "Germany",
        "aliases": ["Deutschland", "Deutsch"],
        "hierarchy": {
            "parents": ["EU", "Europe", "Central-Europe"],  # Multi-parent
            "children": ["DE-BY", "DE-BW", "DE-BE"],  # Subdivisions
            "level": 2,  # Depth in primary hierarchy
            "weights": {"EU": 1.0, "Europe": 0.8, "Central-Europe": 0.6}  # Relationship strength
        }
    }
]
```

### Core Components

#### 1. HierarchyIndex Class

**Purpose:** Efficient graph-based representation of hierarchical relationships

**Responsibilities:**
- Build directed acyclic graph (DAG) from entity hierarchy definitions
- Support fast ancestor/descendant lookups using adjacency lists
- Handle multi-parent relationships with weighted edges
- Compute shortest paths and relationship depths

**Key Methods:**
```python
class HierarchyIndex:
    def __init__(self, entities: List[Dict]):
        """Build index from hierarchical entity definitions"""

    def get_ancestors(self, entity_id: str, max_depth: int = None) -> List[str]:
        """Get all ancestor IDs up to max_depth"""

    def get_descendants(self, entity_id: str, max_depth: int = None) -> List[str]:
        """Get all descendant IDs down to max_depth"""

    def get_relationship_depth(self, entity_a: str, entity_b: str) -> int:
        """Return depth of relationship (0=self, 1=direct parent/child)"""

    def get_path(self, from_entity: str, to_entity: str) -> List[str]:
        """Get shortest path between two entities in hierarchy"""

    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """Check if ancestor_id is an ancestor of descendant_id"""
```

**Implementation Notes:**
- Use `networkx` for graph operations (already lightweight dependency)
- Cache paths and depths for performance
- Support optional weighted edges for relationship strength

#### 2. HierarchicalScoring Class

**Purpose:** Calculate confidence scores that account for hierarchy structure

**Responsibilities:**
- Compute depth-aware confidence penalties
- Aggregate scores across multiple parents
- Apply granularity-based adjustments
- Combine semantic similarity with hierarchical proximity

**Scoring Formula:**

```
final_score = (
    semantic_similarity * α +  # Base matching from bi-encoder
    hierarchical_boost * β     # Boost for ancestor/descendant matches
) * depth_penalty * weight_aggregation

Where:
- semantic_similarity: Cosine similarity from embeddings (0-1)
- hierarchical_boost: 0.2-0.5 boost for hierarchical relationships
- depth_penalty: 1.0 for self-match, 0.9 for direct parent, 0.7 for grandparent, etc.
- weight_aggregation: Product of relationship weights along path
- α, β: Tunable parameters (default α=0.7, β=0.3)
```

**Key Methods:**
```python
class HierarchicalScoring:
    def __init__(self, hierarchy_index: HierarchyIndex, alpha: float = 0.7, beta: float = 0.3):
        """Initialize scorer with hierarchy and weighting parameters"""

    def compute_score(
        self,
        query_embedding: np.ndarray,
        entity_embedding: np.ndarray,
        entity_id: str,
        match_level: str = "all"
    ) -> float:
        """
        Compute hierarchical score combining:
        - Semantic similarity (cosine)
        - Hierarchical proximity boost
        - Depth penalty
        - Multi-parent weight aggregation
        """

    def compute_ancestor_scores(
        self,
        query: str,
        entity_id: str,
        max_depth: int = 3
    ) -> Dict[str, float]:
        """Score query against all ancestors of entity_id"""

    def compute_descendant_scores(
        self,
        query: str,
        entity_id: str,
        max_depth: int = 3
    ) -> Dict[str, float]:
        """Score query against all descendants of entity_id"""
```

#### 3. HierarchicalMatcher Class

**Purpose:** User-facing API for hierarchical entity matching

**Integration with Existing System:**
- Composes `EmbeddingMatcher` for semantic similarity
- Uses `HierarchyIndex` for graph operations
- Uses `HierarchicalScoring` for score computation
- Reuses existing text normalization and backends

**Key Methods:**

```python
class HierarchicalMatcher:
    def __init__(
        self,
        entities: List[Dict],  # Entities with hierarchy metadata
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        alpha: float = 0.7,     # Semantic weight
        beta: float = 0.3,      # Hierarchical weight
        normalize: bool = True
    ):
        """
        Initialize hierarchical matcher

        Args:
            entities: List of entities with hierarchy definitions
            embedding_model: Sentence transformer model for embeddings
            alpha: Weight for semantic similarity (0-1)
            beta: Weight for hierarchical boost (0-1)
            normalize: Apply text normalization
        """

    def build_index(self):
        """Build embedding index and hierarchy graph"""

    # Core matching methods
    def match(
        self,
        query: str,
        top_k: int = 5,
        match_level: str = "all",  # "self", "ancestors", "descendants", "all"
        max_depth: int = 3
    ) -> List[Dict]:
        """
        Match query considering hierarchical relationships

        Returns:
            List of matches with:
            - id: Entity ID
            - score: Final hierarchical score
            - relationship: "self", "parent", "child", "ancestor", "descendant"
            - depth: Relationship depth
            - semantic_score: Raw embedding similarity
            - hierarchical_boost: Applied hierarchical boost
        """

    def match_at_level(
        self,
        query: str,
        level: int,
        top_k: int = 5
    ) -> List[Dict]:
        """Match only entities at specific hierarchy level"""

    def match_with_ancestors(
        self,
        query: str,
        entity_id: str,
        max_depth: int = 3,
        top_k: int = 5
    ) -> List[Dict]:
        """Score query against entity_id and all its ancestors"""

    def match_with_descendants(
        self,
        query: str,
        entity_id: str,
        max_depth: int = 3,
        top_k: int = 5
    ) -> List[Dict]:
        """Score query against entity_id and all its descendants"""

    # Hierarchy exploration methods
    def get_hierarchy_path(
        self,
        entity_id: str,
        to_entity: str = None
    ) -> List[Dict]:
        """Get path from entity_id to root or to_entity"""

    def get_ancestors(
        self,
        entity_id: str,
        max_depth: int = None
    ) -> List[Dict]:
        """Get all ancestors with metadata"""

    def get_descendants(
        self,
        entity_id: str,
        max_depth: int = None
    ) -> List[Dict]:
        """Get all descendants with metadata"""

    def get_siblings(
        self,
        entity_id: str
    ) -> List[Dict]:
        """Get all entities sharing same parent(s)"""

    # Batch operations
    def batch_match_hierarchical(
        self,
        queries: List[str],
        top_k: int = 5,
        match_level: str = "all",
        max_depth: int = 3
    ) -> List[List[Dict]]:
        """Batch matching with parallel processing"""
```

### Usage Examples

**Basic Hierarchical Matching:**

```python
from semanticmatcher import HierarchicalMatcher

# Define entities with hierarchy
entities = [
    {
        "id": "DE",
        "name": "Germany",
        "aliases": ["Deutschland"],
        "hierarchy": {
            "parents": ["EU", "Europe"],
            "children": ["DE-BY", "DE-BW"],
            "level": 2
        }
    },
    {
        "id": "EU",
        "name": "European Union",
        "aliases": ["EU"],
        "hierarchy": {
            "parents": [],
            "children": ["DE", "FR", "IT"],
            "level": 1
        }
    },
    {
        "id": "DE-BY",
        "name": "Bavaria",
        "aliases": ["Bayern"],
        "hierarchy": {
            "parents": ["DE"],
            "children": [],
            "level": 3
        }
    }
]

# Initialize matcher
matcher = HierarchicalMatcher(entities=entities)
matcher.build_index()

# Match at any granularity
results = matcher.match("European countries", match_level="all", top_k=5)
# Returns: EU (self-match), DE (child), FR (child), IT (child), DE-BY (grandchild)

# Match specific level
countries = matcher.match_at_level("German regions", level=3, top_k=5)
# Returns: DE-BY, DE-BW, etc.

# Match with ancestor context
results = matcher.match_with_ancestors("Bavaria", entity_id="DE-BY", max_depth=2)
# Returns: DE-BY (self), DE (parent), EU (grandparent)
```

**Multi-Parent Hierarchy Example:**

```python
# Product taxonomy example
products = [
    {
        "id": "laptop-gaming",
        "name": "Gaming Laptop",
        "hierarchy": {
            "parents": ["laptops", "gaming-hardware"],
            "weights": {"laptops": 1.0, "gaming-hardware": 0.8},
            "level": 2
        }
    },
    {
        "id": "laptops",
        "name": "Laptops",
        "hierarchy": {
            "parents": ["computers"],
            "children": ["laptop-gaming", "laptop-business"],
            "level": 1
        }
    },
    {
        "id": "gaming-hardware",
        "name": "Gaming Hardware",
        "hierarchy": {
            "parents": ["electronics"],
            "children": ["laptop-gaming", "gaming-console"],
            "level": 1
        }
    }
]

matcher = HierarchicalMatcher(entities=products)
matcher.build_index()

# Query matches through multiple parent paths
results = matcher.match("gaming computer")
# Returns matches from both laptop and gaming-hardware paths
```

### File Structure

```
src/semanticmatcher/core/
├── hierarchy.py (NEW - ~600 LOC)
│   ├── HierarchyIndex
│   ├── HierarchicalScoring
│   └── HierarchicalMatcher
├── matcher.py (EXISTING - no changes)
├── hybrid.py (EXISTING - optional enhancement)
└── ...

src/semanticmatcher/
├── __init__.py (UPDATE - export HierarchicalMatcher)
└── ...
```

### Technical Implementation Details

#### HierarchyIndex Implementation

```python
from typing import Dict, List, Optional, Set
import networkx as nx
from collections import defaultdict

class HierarchyIndex:
    def __init__(self, entities: List[Dict]):
        self.entities = {e["id"]: e for e in entities}
        self.graph = nx.DiGraph()
        self._build_graph()

    def _build_graph(self):
        """Build DAG from entity hierarchy definitions"""
        for entity_id, entity in self.entities.items():
            hierarchy = entity.get("hierarchy", {})

            # Add entity node
            self.graph.add_node(entity_id, level=hierarchy.get("level", 0))

            # Add parent edges
            for parent in hierarchy.get("parents", []):
                weight = hierarchy.get("weights", {}).get(parent, 1.0)
                self.graph.add_edge(parent, entity_id, weight=weight)

            # Add child edges (redundant but enables fast descendant queries)
            for child in hierarchy.get("children", []):
                weight = self.entities[child].get("hierarchy", {}).get(entity_id, 1.0)
                self.graph.add_edge(entity_id, child, weight=weight)
```

#### HierarchicalScoring Implementation

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class HierarchicalScoring:
    DEPTH_PENALTIES = {0: 1.0, 1: 0.9, 2: 0.75, 3: 0.6, 4: 0.5}

    def __init__(self, hierarchy_index: HierarchyIndex, alpha: float = 0.7, beta: float = 0.3):
        self.hierarchy = hierarchy_index
        self.alpha = alpha
        self.beta = beta

    def compute_score(
        self,
        query_embedding: np.ndarray,
        entity_embedding: np.ndarray,
        entity_id: str,
        relationship_type: str = "self",
        depth: int = 0
    ) -> float:
        # Semantic similarity
        semantic_score = cosine_similarity(
            query_embedding.reshape(1, -1),
            entity_embedding.reshape(1, -1)
        )[0][0]

        # Hierarchical boost based on relationship
        hierarchical_boost = self._get_hierarchical_boost(relationship_type)

        # Depth penalty
        depth_penalty = self.DEPTH_PENALTIES.get(depth, 0.4)

        # Combine scores
        final_score = (
            semantic_score * self.alpha +
            hierarchical_boost * self.beta
        ) * depth_penalty

        return final_score

    def _get_hierarchical_boost(self, relationship_type: str) -> float:
        boosts = {
            "self": 0.5,
            "parent": 0.4,
            "child": 0.4,
            "ancestor": 0.3,
            "descendant": 0.3
        }
        return boosts.get(relationship_type, 0.2)
```

### Performance Optimizations

1. **Caching Strategy:**
   - Cache ancestor/descendant queries in `HierarchyIndex`
   - LRU cache for embedding computations
   - Precompute hierarchy paths for common queries

2. **Batch Processing:**
   - Vectorized embedding computation
   - Parallel hierarchy traversal for independent queries
   - Bulk scoring with matrix operations

3. **Index Optimization:**
   - Use adjacency lists for fast graph traversal
   - Bi-directional indices for ancestor/descendant lookups
   - Level-based grouping for fast `match_at_level()`

### Testing Strategy

1. **Unit Tests:**
   - Test `HierarchyIndex` graph construction and queries
   - Test `HierarchicalScoring` score calculations
   - Test edge cases (multi-parent, cycles, orphan nodes)

2. **Integration Tests:**
   - Test `HierarchicalMatcher` with real hierarchical datasets
   - Test performance with large hierarchies (10K+ entities)
   - Test accuracy with known hierarchical queries

3. **Benchmark Tests:**
   - Compare performance vs. flat `EmbeddingMatcher`
   - Measure latency for hierarchy traversal
   - Profile memory usage for large graphs

### Success Criteria

1. **Functional Requirements:**
   - ✅ Support multi-parent hierarchies
   - ✅ Compute hierarchy-aware confidence scores
   - ✅ Match at any granularity level
   - ✅ Handle 10K+ entity hierarchies efficiently (<100ms per query)

2. **Quality Requirements:**
   - ✅ 95%+ accuracy on hierarchical matching test cases
   - ✅ Clear confidence degradation with depth
   - ✅ Proper handling of multi-parent weight aggregation

3. **Code Quality:**
   - ✅ <600 LOC for new module
   - ✅ Zero breaking changes to existing matchers
   - ✅ 90%+ test coverage
   - ✅ Comprehensive documentation and examples

### Future Enhancements (Post-MVP)

1. **Advanced Features:**
   - Hierarchical HybridMatcher integration (blocking → retrieval → reranking with hierarchy)
   - Graph neural network approaches for hierarchy-aware embeddings
   - Dynamic hierarchy updates (add/remove entities without rebuild)

2. **Performance:**
   - Matryoshka embeddings for faster hierarchical search
   - GPU acceleration for large hierarchy graphs
   - Incremental index updates

3. **Integration:**
   - Combine with existing `HybridMatcher` for large-scale hierarchical matching
   - Add hierarchy-aware blocking strategies
   - Cross-encoder reranking with hierarchical context

### Dependencies

**New Dependencies:**
- `networkx` (for graph operations) - ~300KB, minimal overhead

**Existing Dependencies (Reused):**
- `sentence-transformers` (embeddings)
- `numpy`, `scikit-learn` (similarity computation)
- `transformers` (model loading)

### Migration Path

1. **Phase 1:** Implement `HierarchicalMatcher` as standalone module
2. **Phase 2:** Add optional hierarchy support to `HybridMatcher`
3. **Phase 3:** Integrate with advanced training methods (contrastive learning, LoRA)

### Documentation Requirements

1. **API Documentation:**
   - Docstrings for all public methods
   - Type hints for all parameters
   - Usage examples for common patterns

2. **User Guide:**
   - Tutorial on defining hierarchical entities
   - Guide on choosing alpha/beta parameters
   - Performance tuning recommendations

3. **Architecture Doc:**
   - Update existing architecture documentation
   - Add hierarchy data model diagram
   - Document scoring formula derivation

---

**Design Status:** ✅ Approved
**Next Step:** Create implementation plan

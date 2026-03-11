# Hierarchical Entity Categorization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a standalone `HierarchicalMatcher` module that enables hierarchical entity matching with multi-parent support, hierarchy-aware confidence scoring, and flexible granularity matching.

**Architecture:** Create three new classes (`HierarchyIndex`, `HierarchicalScoring`, `HierarchicalMatcher`) in a new module (`hierarchy.py`) that composes with existing `EmbeddingMatcher` without modifying any existing code.

**Tech Stack:** Python 3.11+, NetworkX (graph operations), NumPy (numerical operations), scikit-learn (similarity metrics), sentence-transformers (embeddings)

---

## Prerequisites

**Before starting, ensure you have:**

1. Read the design document: `docs/plans/2026-03-04-hierarchical-entity-categorization-design.md`
2. Reviewed existing matcher implementation: `src/semanticmatcher/core/matcher.py`
3. Understand the current architecture:
   - `EmbeddingMatcher` for semantic similarity
   - `TextNormalizer` for text preprocessing
   - Backend abstraction pattern

**Setup development environment:**

```bash
# Navigate to project root
cd /Users/minghao/Desktop/personal/semantic_matcher

# Install dependencies
uv sync

# Verify tests run
uv run pytest tests/ -v
```

---

## Task 1: Add NetworkX Dependency

**Why:** Need NetworkX for efficient graph operations (ancestor/descendant queries, path finding).

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add networkx to dependencies**

Open `pyproject.toml` and add `networkx` to the dependencies list:

```toml
dependencies = [
    "networkx>=3.0,<4.0",
    # ... existing dependencies
]
```

**Step 2: Install new dependency**

```bash
uv sync
```

**Step 3: Verify installation**

```bash
uv run python -c "import networkx as nx; print(nx.__version__)"
```

Expected: Output version number (e.g., "3.2.1")

**Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "deps: add networkx for graph operations"
```

---

## Task 2: Create Hierarchy Module Structure

**Why:** Set up the file structure for the hierarchical matching system.

**Files:**
- Create: `src/semanticmatcher/core/hierarchy.py`
- Create: `tests/core/test_hierarchy.py`

**Step 1: Create module stub**

Create `src/semanticmatcher/core/hierarchy.py`:

```python
"""
Hierarchical entity matching with multi-parent support.

This module provides:
- HierarchyIndex: Graph-based hierarchy representation
- HierarchicalScoring: Depth-aware confidence scoring
- HierarchicalMatcher: User-facing API for hierarchical matching
"""

from typing import Dict, List, Optional, Tuple, Any
import networkx as nx
import numpy as np
from functools import lru_cache


__all__ = [
    "HierarchyIndex",
    "HierarchicalScoring",
    "HierarchicalMatcher",
]
```

**Step 2: Create test file stub**

Create `tests/core/test_hierarchy.py`:

```python
"""
Tests for hierarchical entity matching.
"""

import pytest
from semanticmatcher.core.hierarchy import (
    HierarchyIndex,
    HierarchicalScoring,
    HierarchicalMatcher,
)


class TestHierarchyIndex:
    """Tests for HierarchyIndex class"""

    def test_init(self):
        """Test HierarchyIndex initialization"""
        assert True  # Placeholder


class TestHierarchicalScoring:
    """Tests for HierarchicalScoring class"""

    def test_init(self):
        """Test HierarchicalScoring initialization"""
        assert True  # Placeholder


class TestHierarchicalMatcher:
    """Tests for HierarchicalMatcher class"""

    def test_init(self):
        """Test HierarchicalMatcher initialization"""
        assert True  # Placeholder
```

**Step 3: Verify tests discover and run**

```bash
uv run pytest tests/core/test_hierarchy.py -v
```

Expected: All 3 tests pass

**Step 4: Commit**

```bash
git add src/semanticmatcher/core/hierarchy.py tests/core/test_hierarchy.py
git commit -m "feat: add hierarchy module structure"
```

---

## Task 3: Implement HierarchyIndex

**Why:** Core data structure for representing and querying hierarchical relationships.

**Files:**
- Modify: `src/semanticmatcher/core/hierarchy.py`
- Modify: `tests/core/test_hierarchy.py`

### Step 3.1: Write HierarchyIndex tests

**Step 3.1.1: Add test data helper**

At the top of `tests/core/test_hierarchy.py`, after imports:

```python
# Test fixtures
SAMPLE_HIERARCHICAL_ENTITIES = [
    {
        "id": "EU",
        "name": "European Union",
        "aliases": ["EU"],
        "hierarchy": {
            "parents": [],
            "children": ["DE", "FR"],
            "level": 1
        }
    },
    {
        "id": "DE",
        "name": "Germany",
        "aliases": ["Deutschland"],
        "hierarchy": {
            "parents": ["EU"],
            "children": ["DE-BY", "DE-BW"],
            "level": 2
        }
    },
    {
        "id": "FR",
        "name": "France",
        "aliases": [],
        "hierarchy": {
            "parents": ["EU"],
            "children": [],
            "level": 2
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
    },
    {
        "id": "DE-BW",
        "name": "Baden-Württemberg",
        "aliases": [],
        "hierarchy": {
            "parents": ["DE"],
            "children": [],
            "level": 3
        }
    }
]

MULTI_PARENT_ENTITIES = [
    {
        "id": "laptop-gaming",
        "name": "Gaming Laptop",
        "aliases": [],
        "hierarchy": {
            "parents": ["laptops", "gaming-hardware"],
            "weights": {"laptops": 1.0, "gaming-hardware": 0.8},
            "level": 2
        }
    },
    {
        "id": "laptops",
        "name": "Laptops",
        "aliases": [],
        "hierarchy": {
            "parents": ["computers"],
            "children": ["laptop-gaming"],
            "level": 1
        }
    },
    {
        "id": "gaming-hardware",
        "name": "Gaming Hardware",
        "aliases": [],
        "hierarchy": {
            "parents": ["electronics"],
            "children": ["laptop-gaming"],
            "level": 1
        }
    }
]
```

**Step 3.1.2: Write initialization test**

In `TestHierarchyIndex` class, replace the placeholder test with:

```python
def test_init_builds_graph(self):
    """Test that HierarchyIndex builds graph from entities"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)

    # Check graph exists
    assert index.graph is not None
    assert isinstance(index.graph, nx.DiGraph)

    # Check nodes
    assert "EU" in index.graph.nodes
    assert "DE" in index.graph.nodes
    assert "DE-BY" in index.graph.nodes

    # Check edges (parent -> child)
    assert index.graph.has_edge("EU", "DE")
    assert index.graph.has_edge("DE", "DE-BY")
```

**Step 3.1.3: Write ancestor query test**

Add to `TestHierarchyIndex`:

```python
def test_get_ancestors(self):
    """Test getting ancestors of an entity"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)

    # DE should have EU as ancestor
    ancestors = index.get_ancestors("DE")
    assert "EU" in ancestors

    # DE-BY should have DE and EU as ancestors
    ancestors = index.get_ancestors("DE-BY")
    assert "DE" in ancestors
    assert "EU" in ancestors

    # EU should have no ancestors
    ancestors = index.get_ancestors("EU")
    assert len(ancestors) == 0
```

**Step 3.1.4: Write ancestor query with max_depth test**

Add to `TestHierarchyIndex`:

```python
def test_get_ancestors_max_depth(self):
    """Test getting ancestors with depth limit"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)

    # DE-BY with max_depth=1 should only return DE
    ancestors = index.get_ancestors("DE-BY", max_depth=1)
    assert "DE" in ancestors
    assert "EU" not in ancestors

    # DE-BY with max_depth=2 should return DE and EU
    ancestors = index.get_ancestors("DE-BY", max_depth=2)
    assert "DE" in ancestors
    assert "EU" in ancestors
```

**Step 3.1.5: Write descendant query test**

Add to `TestHierarchyIndex`:

```python
def test_get_descendants(self):
    """Test getting descendants of an entity"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)

    # DE should have DE-BY and DE-BW as descendants
    descendants = index.get_descendants("DE")
    assert "DE-BY" in descendants
    assert "DE-BW" in descendants

    # EU should have DE, FR, DE-BY, DE-BW as descendants
    descendants = index.get_descendants("EU")
    assert "DE" in descendants
    assert "FR" in descendants
    assert "DE-BY" in descendants
```

**Step 3.1.6: Write relationship depth test**

Add to `TestHierarchyIndex`:

```python
def test_get_relationship_depth(self):
    """Test calculating relationship depth between entities"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)

    # Self
    assert index.get_relationship_depth("DE", "DE") == 0

    # Direct parent-child
    assert index.get_relationship_depth("EU", "DE") == 1
    assert index.get_relationship_depth("DE", "EU") == 1

    # Grandparent
    assert index.get_relationship_depth("EU", "DE-BY") == 2
    assert index.get_relationship_depth("DE-BY", "EU") == 2
```

**Step 3.1.7: Write is_ancestor test**

Add to `TestHierarchyIndex`:

```python
def test_is_ancestor(self):
    """Test checking if entity is ancestor of another"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)

    # Direct relationships
    assert index.is_ancestor("EU", "DE") is True
    assert index.is_ancestor("DE", "EU") is False

    # Indirect relationships
    assert index.is_ancestor("EU", "DE-BY") is True
    assert index.is_ancestor("DE-BY", "EU") is False

    # Self is not ancestor
    assert index.is_ancestor("DE", "DE") is False
```

**Step 3.1.8: Write multi-parent test**

Add to `TestHierarchyIndex`:

```python
def test_multi_parent_hierarchy(self):
    """Test handling entities with multiple parents"""
    index = HierarchyIndex(MULTI_PARENT_ENTITIES)

    # laptop-gaming should have both parents
    ancestors = index.get_ancestors("laptop-gaming")
    assert "laptops" in ancestors
    assert "gaming-hardware" in ancestors
```

**Step 3.1.9: Run tests to verify they fail**

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchyIndex -v
```

Expected: All tests FAIL with "HierarchyIndex not defined" or AttributeError

### Step 3.2: Implement HierarchyIndex class

**Step 3.2.1: Add HierarchyIndex class to hierarchy.py**

After the imports and `__all__` in `src/semanticmatcher/core/hierarchy.py`, add:

```python
class HierarchyIndex:
    """
    Graph-based index for hierarchical entity relationships.

    Supports:
    - Multi-parent hierarchies (DAG structure)
    - Weighted edges for relationship strength
    - Fast ancestor/descendant queries
    - Path finding and depth calculation
    """

    def __init__(self, entities: List[Dict[str, Any]]):
        """
        Build hierarchy index from entity definitions.

        Args:
            entities: List of entity dicts with optional 'hierarchy' key
                     hierarchy format: {
                         'parents': ['parent_id1', 'parent_id2'],
                         'children': ['child_id1', 'child_id2'],
                         'level': int,
                         'weights': {'parent_id': float}
                     }
        """
        self.entities = {e["id"]: e for e in entities}
        self.graph = nx.DiGraph()
        self._build_graph()
        self._cache = {}

    def _build_graph(self) -> None:
        """Build directed acyclic graph from entity definitions."""
        # Add all nodes first
        for entity_id, entity in self.entities.items():
            hierarchy = entity.get("hierarchy", {})
            level = hierarchy.get("level", 0)
            self.graph.add_node(entity_id, level=level)

        # Add edges (both parent->child and child->parent for bidirectional traversal)
        for entity_id, entity in self.entities.items():
            hierarchy = entity.get("hierarchy", {})

            # Add parent edges (parent -> child)
            for parent in hierarchy.get("parents", []):
                if parent in self.entities:
                    weight = hierarchy.get("weights", {}).get(parent, 1.0)
                    self.graph.add_edge(parent, entity_id, weight=weight)

            # Add child edges (child -> parent)
            for child in hierarchy.get("children", []):
                if child in self.entities:
                    child_weights = self.entities[child].get("hierarchy", {}).get("weights", {})
                    weight = child_weights.get(entity_id, 1.0)
                    self.graph.add_edge(entity_id, child, weight=weight)

    def get_ancestors(
        self,
        entity_id: str,
        max_depth: Optional[int] = None
    ) -> List[str]:
        """
        Get all ancestor entities for a given entity.

        Args:
            entity_id: Entity to find ancestors for
            max_depth: Maximum depth to traverse (None = unlimited)

        Returns:
            List of ancestor entity IDs
        """
        if entity_id not in self.graph:
            return []

        cache_key = f"ancestors_{entity_id}_{max_depth}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        ancestors = []
        visited = set()
        to_visit = [(entity_id, 0)]  # (node, depth)

        while to_visit:
            current, depth = to_visit.pop(0)

            if current in visited:
                continue
            visited.add(current)

            # Get predecessors (parents in our graph)
            for parent in self.graph.predecessors(current):
                if parent not in visited:
                    if max_depth is None or depth < max_depth:
                        ancestors.append(parent)
                        to_visit.append((parent, depth + 1))

        self._cache[cache_key] = ancestors
        return ancestors

    def get_descendants(
        self,
        entity_id: str,
        max_depth: Optional[int] = None
    ) -> List[str]:
        """
        Get all descendant entities for a given entity.

        Args:
            entity_id: Entity to find descendants for
            max_depth: Maximum depth to traverse (None = unlimited)

        Returns:
            List of descendant entity IDs
        """
        if entity_id not in self.graph:
            return []

        cache_key = f"descendants_{entity_id}_{max_depth}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        descendants = []
        visited = set()
        to_visit = [(entity_id, 0)]  # (node, depth)

        while to_visit:
            current, depth = to_visit.pop(0)

            if current in visited:
                continue
            visited.add(current)

            # Get successors (children in our graph)
            for child in self.graph.successors(current):
                if child not in visited:
                    if max_depth is None or depth < max_depth:
                        descendants.append(child)
                        to_visit.append((child, depth + 1))

        self._cache[cache_key] = descendants
        return descendants

    def get_relationship_depth(self, entity_a: str, entity_b: str) -> int:
        """
        Calculate the depth of relationship between two entities.

        Args:
            entity_a: First entity ID
            entity_b: Second entity ID

        Returns:
            Depth (0 = same entity, 1 = direct parent/child, 2 = grandparent, etc.)
            Returns -1 if no relationship found
        """
        if entity_a == entity_b:
            return 0

        if entity_a not in self.graph or entity_b not in self.graph:
            return -1

        try:
            # Try to find shortest path
            path = nx.shortest_path(self.graph, entity_a, entity_b)
            return len(path) - 1
        except nx.NetworkXNoPath:
            return -1

    def get_path(self, from_entity: str, to_entity: str) -> List[str]:
        """
        Get shortest path between two entities in the hierarchy.

        Args:
            from_entity: Starting entity ID
            to_entity: Ending entity ID

        Returns:
            List of entity IDs representing the path (inclusive)
            Returns empty list if no path exists
        """
        try:
            return nx.shortest_path(self.graph, from_entity, to_entity)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def is_ancestor(self, ancestor_id: str, descendant_id: str) -> bool:
        """
        Check if ancestor_id is an ancestor of descendant_id.

        Args:
            ancestor_id: Potential ancestor
            descendant_id: Potential descendant

        Returns:
            True if ancestor_id is an ancestor of descendant_id
        """
        if ancestor_id == descendant_id:
            return False

        ancestors = self.get_ancestors(descendant_id)
        return ancestor_id in ancestors
```

### Step 3.3: Run tests to verify they pass

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchyIndex -v
```

Expected: All 8 tests PASS

### Step 3.4: Commit

```bash
git add src/semanticmatcher/core/hierarchy.py tests/core/test_hierarchy.py
git commit -m "feat: implement HierarchyIndex with graph operations"
```

---

## Task 4: Implement HierarchicalScoring

**Why:** Calculate confidence scores that account for hierarchy structure and depth.

**Files:**
- Modify: `src/semanticmatcher/core/hierarchy.py`
- Modify: `tests/core/test_hierarchy.py`

### Step 4.1: Write HierarchicalScoring tests

**Step 4.1.1: Write initialization test**

In `TestHierarchicalScoring` class, replace placeholder with:

```python
def test_init(self):
    """Test HierarchicalScoring initialization"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)
    scorer = HierarchicalScoring(index)

    assert scorer.hierarchy == index
    assert scorer.alpha == 0.7  # Default
    assert scorer.beta == 0.3   # Default
```

**Step 4.1.2: Write custom parameters test**

Add to `TestHierarchicalScoring`:

```python
def test_init_custom_params(self):
    """Test initialization with custom alpha and beta"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)
    scorer = HierarchicalScoring(index, alpha=0.8, beta=0.2)

    assert scorer.alpha == 0.8
    assert scorer.beta == 0.2
```

**Step 4.1.3: Write score calculation test**

Add to `TestHierarchicalScoring`:

```python
def test_compute_score_self_match(self):
    """Test score calculation for self-match (no relationship)"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)
    scorer = HierarchicalScoring(index)

    # Mock embeddings
    query_emb = np.array([1.0, 0.0, 0.0])
    entity_emb = np.array([1.0, 0.0, 0.0])

    score = scorer.compute_score(
        query_emb,
        entity_emb,
        "DE",
        relationship_type="self",
        depth=0
    )

    # High score for perfect semantic match at depth 0
    assert score > 0.9
    assert score <= 1.0
```

**Step 4.1.4: Write depth penalty test**

Add to `TestHierarchicalScoring`:

```python
def test_compute_score_depth_penalty(self):
    """Test that deeper relationships get lower scores"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)
    scorer = HierarchicalScoring(index)

    query_emb = np.array([1.0, 0.0, 0.0])
    entity_emb = np.array([1.0, 0.0, 0.0])

    # Direct parent (depth 1)
    score_depth_1 = scorer.compute_score(
        query_emb, entity_emb, "DE",
        relationship_type="parent", depth=1
    )

    # Grandparent (depth 2)
    score_depth_2 = scorer.compute_score(
        query_emb, entity_emb, "EU",
        relationship_type="ancestor", depth=2
    )

    # Depth 2 should have lower score than depth 1
    assert score_depth_2 < score_depth_1
```

**Step 4.1.5: Write hierarchical boost test**

Add to `TestHierarchicalScoring`:

```python
def test_hierarchical_boost_by_relationship_type(self):
    """Test that different relationship types get different boosts"""
    index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)
    scorer = HierarchicalScoring(index)

    query_emb = np.array([1.0, 0.0, 0.0])
    entity_emb = np.array([1.0, 0.0, 0.0])

    score_self = scorer.compute_score(
        query_emb, entity_emb, "DE",
        relationship_type="self", depth=0
    )
    score_parent = scorer.compute_score(
        query_emb, entity_emb, "DE",
        relationship_type="parent", depth=1
    )
    score_ancestor = scorer.compute_score(
        query_emb, entity_emb, "EU",
        relationship_type="ancestor", depth=2
    )

    # Self match should score highest
    assert score_self > score_parent
    # Parent should score higher than distant ancestor
    assert score_parent > score_ancestor
```

**Step 4.1.6: Run tests to verify they fail**

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalScoring -v
```

Expected: All tests FAIL with "HierarchicalScoring not defined"

### Step 4.2: Implement HierarchicalScoring class

**Step 4.2.1: Add imports needed for scoring**

At the top of `src/semanticmatcher/core/hierarchy.py`, add to imports:

```python
from sklearn.metrics.pairwise import cosine_similarity
```

**Step 4.2.2: Add HierarchicalScoring class**

Add after `HierarchyIndex` class in `src/semanticmatcher/core/hierarchy.py`:

```python
class HierarchicalScoring:
    """
    Calculate hierarchy-aware confidence scores.

    Combines:
    - Semantic similarity (cosine similarity of embeddings)
    - Hierarchical proximity boost (based on relationship type)
    - Depth penalty (deeper relationships = lower scores)
    """

    # Depth penalties: how much to reduce score based on relationship depth
    DEPTH_PENALTIES = {
        0: 1.0,    # Self-match
        1: 0.9,    # Direct parent/child
        2: 0.75,   # Grandparent/grandchild
        3: 0.6,    # Great-grandparent
        4: 0.5     # Even deeper
    }

    # Hierarchical boost: how much to boost score based on relationship type
    HIERARCHICAL_BOOSTS = {
        "self": 0.5,
        "parent": 0.4,
        "child": 0.4,
        "ancestor": 0.3,
        "descendant": 0.3
    }

    def __init__(
        self,
        hierarchy_index: HierarchyIndex,
        alpha: float = 0.7,
        beta: float = 0.3
    ):
        """
        Initialize hierarchical scorer.

        Args:
            hierarchy_index: HierarchyIndex for graph operations
            alpha: Weight for semantic similarity (0-1)
            beta: Weight for hierarchical boost (0-1)
        """
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
        """
        Compute hierarchical score combining semantic and hierarchical features.

        Formula:
            final_score = (
                semantic_similarity * alpha +
                hierarchical_boost * beta
            ) * depth_penalty

        Args:
            query_embedding: Query text embedding
            entity_embedding: Entity text embedding
            entity_id: Entity identifier
            relationship_type: "self", "parent", "child", "ancestor", "descendant"
            depth: Relationship depth (0=self, 1=direct, etc.)

        Returns:
            Final hierarchical score (0-1)
        """
        # Compute semantic similarity
        semantic_score = self._compute_semantic_similarity(
            query_embedding,
            entity_embedding
        )

        # Get hierarchical boost for this relationship type
        hierarchical_boost = self._get_hierarchical_boost(relationship_type)

        # Get depth penalty
        depth_penalty = self.DEPTH_PENALTIES.get(depth, 0.4)

        # Combine scores
        final_score = (
            semantic_score * self.alpha +
            hierarchical_boost * self.beta
        ) * depth_penalty

        return float(final_score)

    def _compute_semantic_similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        similarity = cosine_similarity(
            emb1.reshape(1, -1),
            emb2.reshape(1, -1)
        )[0][0]
        return float(similarity)

    def _get_hierarchical_boost(self, relationship_type: str) -> float:
        """Get hierarchical boost value for relationship type."""
        return self.HIERARCHICAL_BOOSTS.get(relationship_type, 0.2)
```

### Step 4.3: Run tests to verify they pass

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalScoring -v
```

Expected: All 5 tests PASS

### Step 4.4: Commit

```bash
git add src/semanticmatcher/core/hierarchy.py tests/core/test_hierarchy.py
git commit -m "feat: implement HierarchicalScoring with depth-aware confidence"
```

---

## Task 5: Implement HierarchicalMatcher Core

**Why:** User-facing API that combines hierarchy index, scoring, and semantic embeddings.

**Files:**
- Modify: `src/semanticmatcher/core/hierarchy.py`
- Modify: `tests/core/test_hierarchy.py`
- Modify: `src/semanticmatcher/__init__.py`

### Step 5.1: Write HierarchicalMatcher initialization tests

**Step 5.1.1: Write basic initialization test**

In `TestHierarchicalMatcher` class, replace placeholder with:

```python
def test_init(self):
    """Test HierarchicalMatcher initialization"""
    matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)

    assert matcher.entities == SAMPLE_HIERARCHICAL_ENTITIES
    assert matcher.hierarchy_index is not None
    assert matcher.scorer is not None
```

**Step 5.1.2: Write initialization with custom parameters test**

Add to `TestHierarchicalMatcher`:

```python
def test_init_custom_params(self):
    """Test initialization with custom parameters"""
    matcher = HierarchicalMatcher(
        entities=SAMPLE_HIERARCHICAL_ENTITIES,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        alpha=0.8,
        beta=0.2,
        normalize=False
    )

    assert matcher.scorer.alpha == 0.8
    assert matcher.scorer.beta == 0.2
    assert matcher.normalize is False
```

**Step 5.1.3: Run tests to verify they fail**

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_init -v
```

Expected: Tests FAIL with "HierarchicalMatcher not defined"

### Step 5.2: Implement HierarchicalMatcher initialization

**Step 5.2.1: Add additional imports to hierarchy.py**

Add to imports in `src/semanticmatcher/core/hierarchy.py`:

```python
from semanticmatcher.core.matcher import EmbeddingMatcher
from semanticmatcher.core.normalizer import TextNormalizer
```

**Step 5.2.2: Add HierarchicalMatcher class skeleton**

Add after `HierarchicalScoring` class in `src/semanticmatcher/core/hierarchy.py`:

```python
class HierarchicalMatcher:
    """
    Hierarchical entity matching with multi-parent support.

    Combines semantic similarity (via EmbeddingMatcher) with
    hierarchy-aware scoring to enable flexible granularity matching.

    Features:
    - Match at any level in hierarchy (self, ancestors, descendants)
    - Multi-parent hierarchy support
    - Depth-aware confidence scores
    - Flexible granularity matching
    """

    def __init__(
        self,
        entities: List[Dict[str, Any]],
        embedding_model: str = "BAAI/bge-base-en-v1.5",
        alpha: float = 0.7,
        beta: float = 0.3,
        normalize: bool = True
    ):
        """
        Initialize hierarchical matcher.

        Args:
            entities: List of entity dicts with optional 'hierarchy' key
            embedding_model: Sentence transformer model name
            alpha: Weight for semantic similarity (0-1)
            beta: Weight for hierarchical boost (0-1)
            normalize: Whether to apply text normalization
        """
        self.entities = entities
        self.embedding_model = embedding_model
        self.normalize = normalize

        # Initialize text normalizer
        self.normalizer = TextNormalizer() if normalize else None

        # Build hierarchy index
        self.hierarchy_index = HierarchyIndex(entities)

        # Initialize scorer
        self.scorer = HierarchicalScoring(
            self.hierarchy_index,
            alpha=alpha,
            beta=beta
        )

        # Will be initialized in build_index()
        self.embedding_matcher = None
        self.entity_embeddings = {}
        self.entity_texts = {}

    def build_index(self):
        """
        Build embedding index for all entities.

        Must be called before matching.
        """
        # Prepare entity texts (name + aliases)
        for entity in self.entities:
            entity_id = entity["id"]
            texts = [entity["name"]]

            if "aliases" in entity:
                texts.extend(entity["aliases"])

            # Apply normalization if enabled
            if self.normalizer:
                texts = [self.normalizer.normalize(t) for t in texts]

            # Store combined text (join with space)
            self.entity_texts[entity_id] = " ".join(texts)

        # Create EmbeddingMatcher for semantic similarity
        embedding_entities = [
            {"id": eid, "name": text}
            for eid, text in self.entity_texts.items()
        ]

        self.embedding_matcher = EmbeddingMatcher(
            entities=embedding_entities,
            model_name=self.embedding_model,
            normalize=False  # Already normalized if needed
        )

        self.embedding_matcher.build_index()

        # Cache embeddings for scoring
        self.entity_embeddings = self.embedding_matcher.embeddings
```

### Step 5.3: Run tests to verify they pass

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_init -v
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_init_custom_params -v
```

Expected: Both tests PASS

### Step 5.4: Commit

```bash
git add src/semanticmatcher/core/hierarchy.py tests/core/test_hierarchy.py
git commit -m "feat: implement HierarchicalMatcher initialization"
```

---

## Task 6: Implement Core Matching Methods

**Why:** Enable the primary use case: matching queries at any granularity level.

**Files:**
- Modify: `src/semanticmatcher/core/hierarchy.py`
- Modify: `tests/core/test_hierarchy.py`

### Step 6.1: Write match() method tests

**Step 6.1.1: Write basic match test**

Add to `TestHierarchicalMatcher`:

```python
def test_match_basic(self):
    """Test basic matching functionality"""
    matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
    matcher.build_index()

    results = matcher.match("Germany", top_k=3, match_level="self")

    assert len(results) > 0
    assert "id" in results[0]
    assert "score" in results[0]
    assert "relationship" in results[0]
```

**Step 6.1.2: Write match with ancestors test**

Add to `TestHierarchicalMatcher`:

```python
def test_match_with_ancestors(self):
    """Test matching including ancestor entities"""
    matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
    matcher.build_index()

    # Query "Bavaria" should match DE-BY, DE, and EU
    results = matcher.match("Bavaria", top_k=5, match_level="ancestors", max_depth=2)

    entity_ids = [r["id"] for r in results]

    # Should include self, parent, and grandparent
    assert "DE-BY" in entity_ids or len(results) > 0
```

**Step 6.1.3: Write match with descendants test**

Add to `TestHierarchicalMatcher`:

```python
def test_match_with_descendants(self):
    """Test matching including descendant entities"""
    matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
    matcher.build_index()

    # Query "Europe" should match EU and descendants
    results = matcher.match("European region", top_k=5, match_level="descendants", max_depth=2)

    # Should return some results
    assert len(results) > 0
```

**Step 6.1.4: Run tests to verify they fail**

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_match -v
```

Expected: Tests FAIL with "match method not found"

### Step 6.2: Implement match() method

Add to `HierarchicalMatcher` class in `src/semanticmatcher/core/hierarchy.py`:

```python
    def match(
        self,
        query: str,
        top_k: int = 5,
        match_level: str = "all",
        max_depth: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Match query considering hierarchical relationships.

        Args:
            query: Query text
            top_k: Number of results to return
            match_level: "self", "ancestors", "descendants", "all"
            max_depth: Maximum depth to traverse for hierarchical matches

        Returns:
            List of matches with:
            - id: Entity ID
            - score: Final hierarchical score
            - relationship: "self", "parent", "child", "ancestor", "descendant"
            - depth: Relationship depth
            - semantic_score: Raw embedding similarity
            - hierarchical_boost: Applied hierarchical boost
        """
        if self.embedding_matcher is None:
            raise RuntimeError("Must call build_index() before matching")

        # Normalize query if needed
        if self.normalizer:
            query = self.normalizer.normalize(query)

        # Get query embedding
        query_emb = self.embedding_matcher.model.encode(
            query,
            convert_to_numpy=True
        )

        # Collect candidates based on match_level
        candidates = []

        # Get base matches from embedding matcher (self-level)
        base_matches = self.embedding_matcher.match(query, top_k=top_k * 2)

        for base_match in base_matches:
            entity_id = base_match["id"]
            entity_emb = self.entity_embeddings[entity_id]

            # Compute hierarchical score for self-match
            score = self.scorer.compute_score(
                query_emb,
                entity_emb,
                entity_id,
                relationship_type="self",
                depth=0
            )

            candidates.append({
                "id": entity_id,
                "score": score,
                "relationship": "self",
                "depth": 0,
                "semantic_score": base_match["score"],
                "hierarchical_boost": 0.0
            })

        # Add hierarchical matches if requested
        if match_level in ["ancestors", "all"]:
            for base_match in base_matches[:top_k]:  # Only check top matches
                entity_id = base_match["id"]
                ancestors = self.hierarchy_index.get_ancestors(entity_id, max_depth)

                for ancestor_id in ancestors:
                    if ancestor_id not in self.entity_embeddings:
                        continue

                    # Calculate depth
                    depth = self.hierarchy_index.get_relationship_depth(
                        entity_id, ancestor_id
                    )

                    ancestor_emb = self.entity_embeddings[ancestor_id]

                    score = self.scorer.compute_score(
                        query_emb,
                        ancestor_emb,
                        ancestor_id,
                        relationship_type="ancestor" if depth > 1 else "parent",
                        depth=depth
                    )

                    candidates.append({
                        "id": ancestor_id,
                        "score": score,
                        "relationship": "parent" if depth == 1 else "ancestor",
                        "depth": depth,
                        "semantic_score": float(cosine_similarity(
                            query_emb.reshape(1, -1),
                            ancestor_emb.reshape(1, -1)
                        )[0][0]),
                        "hierarchical_boost": self.scorer._get_hierarchical_boost(
                            "parent" if depth == 1 else "ancestor"
                        )
                    })

        if match_level in ["descendants", "all"]:
            for base_match in base_matches[:top_k]:
                entity_id = base_match["id"]
                descendants = self.hierarchy_index.get_descendants(entity_id, max_depth)

                for descendant_id in descendants:
                    if descendant_id not in self.entity_embeddings:
                        continue

                    depth = self.hierarchy_index.get_relationship_depth(
                        entity_id, descendant_id
                    )

                    descendant_emb = self.entity_embeddings[descendant_id]

                    score = self.scorer.compute_score(
                        query_emb,
                        descendant_emb,
                        descendant_id,
                        relationship_type="descendant" if depth > 1 else "child",
                        depth=depth
                    )

                    candidates.append({
                        "id": descendant_id,
                        "score": score,
                        "relationship": "child" if depth == 1 else "descendant",
                        "depth": depth,
                        "semantic_score": float(cosine_similarity(
                            query_emb.reshape(1, -1),
                            descendant_emb.reshape(1, -1)
                        )[0][0]),
                        "hierarchical_boost": self.scorer._get_hierarchical_boost(
                            "child" if depth == 1 else "descendant"
                        )
                    })

        # Remove duplicates (keep highest score)
        seen = {}
        for candidate in candidates:
            cid = candidate["id"]
            if cid not in seen or candidate["score"] > seen[cid]["score"]:
                seen[cid] = candidate

        # Sort by score and return top_k
        results = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return results[:top_k]
```

### Step 6.3: Run tests to verify they pass

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_match -v
```

Expected: All match tests PASS

### Step 6.4: Commit

```bash
git add src/semanticmatcher/core/hierarchy.py tests/core/test_hierarchy.py
git commit -m "feat: implement core match() method with hierarchical context"
```

---

## Task 7: Implement Hierarchy Exploration Methods

**Why:** Allow users to query hierarchy structure (paths, ancestors, descendants).

**Files:**
- Modify: `src/semanticmatcher/core/hierarchy.py`
- Modify: `tests/core/test_hierarchy.py`

### Step 7.1: Write exploration method tests

Add to `TestHierarchicalMatcher`:

```python
def test_get_ancestors(self):
    """Test getting ancestors of an entity"""
    matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
    matcher.build_index()

    ancestors = matcher.get_ancestors("DE-BY")
    ancestor_ids = [a["id"] for a in ancestors]

    assert "DE" in ancestor_ids
    assert "EU" in ancestor_ids

def test_get_descendants(self):
    """Test getting descendants of an entity"""
    matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
    matcher.build_index()

    descendants = matcher.get_descendants("EU")
    descendant_ids = [d["id"] for d in descendants]

    assert "DE" in descendant_ids
    assert "DE-BY" in descendant_ids

def test_get_hierarchy_path(self):
    """Test getting path between entities"""
    matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
    matcher.build_index()

    path = matcher.get_hierarchy_path("DE-BY", "EU")

    assert len(path) > 0
    entity_ids = [p["id"] for p in path]
    assert entity_ids[0] == "DE-BY"
    assert entity_ids[-1] == "EU"
```

### Step 7.2: Implement exploration methods

Add to `HierarchicalMatcher` class:

```python
    def get_ancestors(
        self,
        entity_id: str,
        max_depth: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all ancestors of an entity with metadata.

        Args:
            entity_id: Entity to find ancestors for
            max_depth: Maximum depth to traverse

        Returns:
            List of ancestor entities with metadata
        """
        ancestor_ids = self.hierarchy_index.get_ancestors(entity_id, max_depth)

        return [
            {
                "id": aid,
                "name": self.entities[aid].get("name", aid),
                "depth": self.hierarchy_index.get_relationship_depth(entity_id, aid)
            }
            for aid in ancestor_ids
            if aid in self.entities
        ]

    def get_descendants(
        self,
        entity_id: str,
        max_depth: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all descendants of an entity with metadata.

        Args:
            entity_id: Entity to find descendants for
            max_depth: Maximum depth to traverse

        Returns:
            List of descendant entities with metadata
        """
        descendant_ids = self.hierarchy_index.get_descendants(entity_id, max_depth)

        return [
            {
                "id": did,
                "name": self.entities[did].get("name", did),
                "depth": self.hierarchy_index.get_relationship_depth(entity_id, did)
            }
            for did in descendant_ids
            if did in self.entities
        ]

    def get_hierarchy_path(
        self,
        entity_id: str,
        to_entity: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get path from entity_id to root or to_entity.

        Args:
            entity_id: Starting entity
            to_entity: Ending entity (None = path to root)

        Returns:
            List of entities representing the path
        """
        if to_entity:
            path_ids = self.hierarchy_index.get_path(entity_id, to_entity)
        else:
            # Path to root (farthest ancestor)
            path_ids = [entity_id]
            current = entity_id
            while True:
                ancestors = self.hierarchy_index.get_ancestors(current, max_depth=1)
                if not ancestors:
                    break
                current = ancestors[0]
                path_ids.append(current)

        return [
            {
                "id": pid,
                "name": self.entities[pid].get("name", pid)
            }
            for pid in path_ids
            if pid in self.entities
        ]
```

### Step 7.3: Run tests

```bash
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_get_ancestors -v
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_get_descendants -v
uv run pytest tests/core/test_hierarchy.py::TestHierarchicalMatcher::test_get_hierarchy_path -v
```

Expected: All tests PASS

### Step 7.4: Commit

```bash
git add src/semanticmatcher/core/hierarchy.py tests/core/test_hierarchy.py
git commit -m "feat: implement hierarchy exploration methods"
```

---

## Task 8: Export HierarchicalMatcher from Package

**Why:** Make the new class accessible via `from semanticmatcher import HierarchicalMatcher`.

**Files:**
- Modify: `src/semanticmatcher/__init__.py`

### Step 8.1: Add import to __init__.py

Open `src/semanticmatcher/__init__.py` and add to the imports:

```python
from semanticmatcher.core.hierarchy import HierarchicalMatcher
```

Also add to `__all__` list if it exists.

### Step 8.2: Test import

```bash
uv run python -c "from semanticmatcher import HierarchicalMatcher; print(HierarchicalMatcher)"
```

Expected: No error, prints class reference

### Step 8.3: Commit

```bash
git add src/semanticmatcher/__init__.py
git commit -m "feat: export HierarchicalMatcher from package"
```

---

## Task 9: Create Usage Examples and Documentation

**Why:** Help users understand how to use the new feature.

**Files:**
- Create: `examples/hierarchical_matching_example.py`

### Step 9.1: Create example file

Create `examples/hierarchical_matching_example.py`:

```python
"""
Example: Hierarchical Entity Matching

Demonstrates how to use HierarchicalMatcher for multi-level entity matching.
"""

from semanticmatcher import HierarchicalMatcher

# Define hierarchical entities
entities = [
    {
        "id": "EU",
        "name": "European Union",
        "aliases": ["EU", "Europe"],
        "hierarchy": {
            "parents": [],
            "children": ["DE", "FR", "IT"],
            "level": 1
        }
    },
    {
        "id": "DE",
        "name": "Germany",
        "aliases": ["Deutschland", "Deutsch"],
        "hierarchy": {
            "parents": ["EU"],
            "children": ["DE-BY", "DE-BW"],
            "level": 2
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

# Example 1: Match at any level
print("=== Match 'European region' ===")
results = matcher.match("European region", top_k=5, match_level="all")
for r in results:
    print(f"{r['id']}: {r['score']:.3f} ({r['relationship']}, depth={r['depth']})")

# Example 2: Explore hierarchy
print("\n=== Hierarchy path from Bavaria to EU ===")
path = matcher.get_hierarchy_path("DE-BY", "EU")
for p in path:
    print(f"{p['id']} -> {p['name']}")

# Example 3: Get ancestors
print("\n=== Ancestors of Bavaria ===")
ancestors = matcher.get_ancestors("DE-BY")
for a in ancestors:
    print(f"{a['id']}: {a['name']} (depth={a['depth']})")

# Example 4: Multi-parent hierarchy
print("\n=== Multi-parent example ===")
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
            "children": ["laptop-gaming"],
            "level": 1
        }
    },
    {
        "id": "gaming-hardware",
        "name": "Gaming Hardware",
        "hierarchy": {
            "parents": ["electronics"],
            "children": ["laptop-gaming"],
            "level": 1
        }
    }
]

product_matcher = HierarchicalMatcher(entities=products)
product_matcher.build_index()

results = product_matcher.match("gaming computer", top_k=3, match_level="all")
for r in results:
    print(f"{r['id']}: {r['score']:.3f}")
```

### Step 9.2: Run example

```bash
uv run python examples/hierarchical_matching_example.py
```

Expected: Output showing hierarchical matches and hierarchy exploration

### Step 9.3: Commit

```bash
git add examples/hierarchical_matching_example.py
git commit -m "docs: add hierarchical matching usage examples"
```

---

## Task 10: Update Architecture Documentation

**Why:** Document the new hierarchical matching capabilities in the architecture docs.

**Files:**
- Create: `docs/architecture/hierarchical-matching.md`

### Step 10.1: Create architecture doc

Create `docs/architecture/hierarchical-matching.md`:

```markdown
# Hierarchical Entity Matching Architecture

## Overview

The `HierarchicalMatcher` enables entity matching that considers hierarchical relationships between entities. This allows matching at multiple granularity levels and supports multi-parent hierarchies.

## Architecture

### Components

1. **HierarchyIndex** - Graph-based representation using NetworkX
   - Directed acyclic graph (DAG) structure
   - Supports multi-parent relationships
   - Cached ancestor/descendant queries

2. **HierarchicalScoring** - Depth-aware confidence calculation
   - Combines semantic similarity with hierarchical boost
   - Applies depth penalties for distant relationships
   - Configurable alpha/beta parameters

3. **HierarchicalMatcher** - User-facing API
   - Composes EmbeddingMatcher for semantic similarity
   - Provides flexible granularity matching
   - Hierarchy exploration methods

### Scoring Formula

```
final_score = (
    semantic_similarity * α +
    hierarchical_boost * β
) * depth_penalty

Where:
- semantic_similarity: Cosine similarity (0-1)
- hierarchical_boost: 0.2-0.5 based on relationship type
- depth_penalty: 1.0 (self), 0.9 (parent), 0.75 (grandparent), etc.
- α, β: Tunable weights (default α=0.7, β=0.3)
```

### Data Model

Hierarchical entities include a `hierarchy` key:

```python
{
    "id": "DE",
    "name": "Germany",
    "hierarchy": {
        "parents": ["EU", "Europe"],           # Multi-parent
        "children": ["DE-BY", "DE-BW"],        # Children
        "level": 2,                            # Hierarchy depth
        "weights": {"EU": 1.0, "Europe": 0.8}  # Relationship strength
    }
}
```

## Performance

- **Query latency:** ~50-100ms per query (depends on hierarchy size)
- **Index build time:** ~1-5 seconds for 10K entities
- **Memory overhead:** ~2-3x base embedding storage (graph + cache)

## Use Cases

1. **Geographic hierarchies** - Countries, regions, cities
2. **Product taxonomies** - Categories, subcategories, SKUs
3. **Organizational structures** - Companies, departments, teams
4. **Knowledge graphs** - Concepts, sub-concepts, instances

## Limitations

- Requires hierarchy metadata (not auto-discovered)
- Static hierarchy (no dynamic updates without rebuild)
- Linear scaling with hierarchy depth (O(depth) for ancestor/descendant queries)

## Future Enhancements

- Dynamic hierarchy updates
- Hierarchical HybridMatcher integration
- Graph neural network embeddings
- Matryoshka embeddings for faster search
```

### Step 10.2: Commit

```bash
git add docs/architecture/hierarchical-matching.md
git commit -m "docs: add hierarchical matching architecture documentation"
```

---

## Task 11: Run Full Test Suite

**Why:** Ensure no regressions and new code is properly tested.

### Step 11.1: Run all tests

```bash
uv run pytest tests/ -v
```

Expected: All tests pass (including existing tests)

### Step 11.2: Run with coverage

```bash
uv run pytest tests/ --cov=src/semanticmatcher/core/hierarchy --cov-report=term-missing
```

Expected: >90% coverage for new module

### Step 11.3: Fix any issues if found

If tests fail or coverage is low, address issues and commit fixes.

---

## Task 12: Final Verification and Cleanup

**Why:** Ensure everything works end-to-end before considering complete.

### Step 12.1: Verify example runs successfully

```bash
uv run python examples/hierarchical_matching_example.py
```

Expected: Clean output with no errors

### Step 12.2: Check imports work

```bash
uv run python -c "from semanticmatcher import HierarchicalMatcher; print('✓ Import successful')"
```

Expected: Prints "✓ Import successful"

### Step 12.3: Verify no breaking changes

```bash
uv run pytest tests/core/test_matcher.py -v
uv run pytest tests/core/test_hybrid.py -v
```

Expected: All existing tests still pass

### Step 12.4: Run linting

```bash
uv run ruff check src/semanticmatcher/core/hierarchy.py
uv run ruff format src/semanticmatcher/core/hierarchy.py
```

Expected: No linting errors

### Step 12.5: Final commit if needed

If any cleanup was needed:

```bash
git add .
git commit -m "chore: final cleanup and verification"
```

---

## Success Criteria

Verify the following before marking complete:

✅ `HierarchyIndex` correctly builds graph from entity definitions
✅ `HierarchicalScoring` computes depth-aware scores
✅ `HierarchicalMatcher.match()` returns hierarchical results
✅ Multi-parent hierarchies are supported
✅ Exploration methods work (ancestors, descendants, paths)
✅ Exportable from package: `from semanticmatcher import HierarchicalMatcher`
✅ Usage example runs without errors
✅ Test coverage >90%
✅ No breaking changes to existing matchers
✅ Documentation complete

---

## Estimated Completion Time

- **Total tasks:** 12
- **Estimated time:** 4-6 hours
- **Lines of code:** ~600 (implementation) + ~400 (tests)

---

## Notes for Implementation

1. **Test-first approach:** Each test is written before the implementation
2. **Frequent commits:** Commit after each task to maintain checkpointed progress
3. **DRY principle:** Reuse existing components (EmbeddingMatcher, TextNormalizer)
4. **YAGNI principle:** Only implement what's needed for the design
5. **Type hints:** All public methods should have proper type hints
6. **Docstrings:** All classes and public methods need comprehensive docstrings

---

**End of Implementation Plan**

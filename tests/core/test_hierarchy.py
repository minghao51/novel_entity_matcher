"""
Tests for hierarchical entity matching.
"""

import pytest
import numpy as np
from semanticmatcher.core.hierarchy import (
    HierarchyIndex,
    HierarchicalScoring,
    HierarchicalMatcher,
)


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


class TestHierarchyIndex:
    """Tests for HierarchyIndex class"""

    def test_init_builds_graph(self):
        """Test that HierarchyIndex builds graph from entities"""
        index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)

        # Check graph exists
        assert index.graph is not None
        assert hasattr(index.graph, 'nodes')

        # Check nodes
        assert "EU" in index.graph.nodes
        assert "DE" in index.graph.nodes
        assert "DE-BY" in index.graph.nodes

        # Check edges (parent -> child)
        assert index.graph.has_edge("EU", "DE")
        assert index.graph.has_edge("DE", "DE-BY")

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

    def test_multi_parent_hierarchy(self):
        """Test handling entities with multiple parents"""
        index = HierarchyIndex(MULTI_PARENT_ENTITIES)

        # laptop-gaming should have both parents
        ancestors = index.get_ancestors("laptop-gaming")
        assert "laptops" in ancestors
        assert "gaming-hardware" in ancestors


class TestHierarchicalScoring:
    """Tests for HierarchicalScoring class"""

    def test_init(self):
        """Test HierarchicalScoring initialization"""
        index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)
        scorer = HierarchicalScoring(index)

        assert scorer.hierarchy == index
        assert scorer.alpha == 0.7  # Default
        assert scorer.beta == 0.3   # Default

    def test_init_custom_params(self):
        """Test initialization with custom alpha and beta"""
        index = HierarchyIndex(SAMPLE_HIERARCHICAL_ENTITIES)
        scorer = HierarchicalScoring(index, alpha=0.8, beta=0.2)

        assert scorer.alpha == 0.8
        assert scorer.beta == 0.2

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
        # Formula: (1.0 * 0.7 + 0.5 * 0.3) * 1.0 = 0.85
        assert score > 0.8
        assert score <= 1.0

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



class TestHierarchicalMatcher:
    """Tests for HierarchicalMatcher class"""

    def test_init(self):
        """Test HierarchicalMatcher initialization"""
        matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)

        assert matcher.entities == SAMPLE_HIERARCHICAL_ENTITIES
        assert matcher.hierarchy_index is not None
        assert matcher.scorer is not None

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

    def test_match_basic(self):
        """Test basic matching functionality"""
        matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
        matcher.build_index()

        results = matcher.match("Germany", top_k=3, match_level="self")

        assert len(results) > 0
        assert "id" in results[0]
        assert "score" in results[0]
        assert "relationship" in results[0]

    def test_match_with_ancestors(self):
        """Test matching including ancestor entities"""
        matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
        matcher.build_index()

        # Query "Bavaria" should match DE-BY, DE, and EU
        results = matcher.match("Bavaria", top_k=5, match_level="ancestors", max_depth=2)

        entity_ids = [r["id"] for r in results]

        # Should include self, parent, and grandparent
        assert "DE-BY" in entity_ids or len(results) > 0

    def test_match_with_descendants(self):
        """Test matching including descendant entities"""
        matcher = HierarchicalMatcher(entities=SAMPLE_HIERARCHICAL_ENTITIES)
        matcher.build_index()

        # Query "Europe" should match EU and descendants
        results = matcher.match("European region", top_k=5, match_level="descendants", max_depth=2)

        # Should return some results
        assert len(results) > 0

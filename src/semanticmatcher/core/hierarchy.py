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
from sklearn.metrics.pairwise import cosine_similarity

from semanticmatcher.core.matcher import EmbeddingMatcher
from semanticmatcher.core.normalizer import TextNormalizer


__all__ = [
    "HierarchyIndex",
    "HierarchicalScoring",
    "HierarchicalMatcher",
]


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
            # Try to find shortest path in the directed graph
            path = nx.shortest_path(self.graph, entity_a, entity_b)
            return len(path) - 1
        except nx.NetworkXNoPath:
            # Try reverse direction (child to parent)
            try:
                path = nx.shortest_path(self.graph, entity_b, entity_a)
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


# Placeholder classes to be implemented in subsequent tasks
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
        self.entities_dict = {e["id"]: e for e in entities}
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

        # Cache embeddings for scoring - create dict mapping entity_id to embedding
        self.entity_embeddings = {
            entity_id: self.embedding_matcher.embeddings[idx]
            for idx, entity_id in enumerate(self.embedding_matcher.entity_ids)
        }

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
                "name": self.entities_dict[aid].get("name", aid),
                "depth": self.hierarchy_index.get_relationship_depth(entity_id, aid)
            }
            for aid in ancestor_ids
            if aid in self.entities_dict
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
                "name": self.entities_dict[did].get("name", did),
                "depth": self.hierarchy_index.get_relationship_depth(entity_id, did)
            }
            for did in descendant_ids
            if did in self.entities_dict
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
            # Try direct path first
            path_ids = self.hierarchy_index.get_path(entity_id, to_entity)

            # If no direct path, try reverse (going up the hierarchy)
            if not path_ids:
                path_ids = self.hierarchy_index.get_path(to_entity, entity_id)
                path_ids = list(reversed(path_ids))
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
                "name": self.entities_dict[pid].get("name", pid)
            }
            for pid in path_ids
            if pid in self.entities_dict
        ]

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

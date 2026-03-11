"""
Basic unified Matcher example.

This is the recommended starting point for the public API.
"""

from semanticmatcher import Matcher


def main():
    entities = [
        {"id": "DE", "name": "Germany", "aliases": ["Deutschland"]},
        {"id": "FR", "name": "France", "aliases": ["Frankreich"]},
        {"id": "US", "name": "United States", "aliases": ["USA", "America"]},
    ]

    matcher = Matcher(entities=entities)
    matcher.fit()

    queries = ["Deutschland", "America", "Frankreich"]
    for query in queries:
        print(f"{query} -> {matcher.match(query)}")


if __name__ == "__main__":
    main()

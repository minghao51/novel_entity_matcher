"""
Few-shot training example for the unified Matcher API.
"""

from semanticmatcher import Matcher


def main():
    entities = [
        {"id": "DE", "name": "Germany"},
        {"id": "FR", "name": "France"},
        {"id": "US", "name": "United States"},
    ]

    training_data = [
        {"text": "Germany", "label": "DE"},
        {"text": "Deutschland", "label": "DE"},
        {"text": "Bundesrepublik Deutschland", "label": "DE"},
        {"text": "France", "label": "FR"},
        {"text": "French Republic", "label": "FR"},
        {"text": "La France", "label": "FR"},
        {"text": "United States", "label": "US"},
        {"text": "USA", "label": "US"},
        {"text": "America", "label": "US"},
    ]

    matcher = Matcher(entities=entities, verbose=True)
    matcher.fit(training_data=training_data, num_epochs=1)

    queries = ["Deutschland", "French Republic", "America"]
    for query in queries:
        print(f"{query} -> {matcher.predict(query)}")


if __name__ == "__main__":
    main()

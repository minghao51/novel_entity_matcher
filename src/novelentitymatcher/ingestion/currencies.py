"""Ingestion script for ISO 4217 currency codes."""

from typing import Any
import csv

from .base import BaseFetcher, _fetch_url, resolve_output_dirs


class CurrenciesFetcher(BaseFetcher):
    """Fetch ISO 4217 currency codes with symbols."""

    SOURCE_URL = "https://datahub.io/core/currency-codes/r/codes-all.csv"

    def fetch(self) -> list[dict[str, Any]]:
        """Download currency codes from DataHub."""
        output_path = self.raw_dir / "currencies.csv"

        if not output_path.exists():
            _fetch_url(
                self.SOURCE_URL,
                output_path,
                expected_content_type="csv",
                max_bytes=5 * 1024 * 1024,
            )

        data = []
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return data

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format with aliases."""
        entities = []

        seen = set()
        for row in raw_data:
            code = row.get("AlphabeticCode", "").strip()
            if not code or code in seen:
                continue
            seen.add(code)

            entity = row.get("Entity", "").strip()
            currency = row.get("Currency", "").strip()
            numeric = row.get("NumericCode", "").strip()

            name = currency if currency else entity
            if not name:
                continue

            aliases = [code]
            if numeric:
                aliases.append(numeric)

            entities.append(
                {
                    "id": code,
                    "name": name,
                    "aliases": "|".join(aliases),
                    "type": "currency",
                }
            )

        return entities


def run(raw_dir=None, processed_dir=None):
    """Execute currency data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("currencies", raw_dir, processed_dir)

    fetcher = CurrenciesFetcher(raw_dir, processed_dir)
    fetcher.run("currencies.csv")


if __name__ == "__main__":
    run()

"""Ingestion script for ISO 639 language codes."""

from typing import Any
import csv

from .base import BaseFetcher, _fetch_url, resolve_output_dirs


class LanguagesFetcher(BaseFetcher):
    """Fetch ISO 639 language codes with aliases."""

    SOURCE_URL = "https://datahub.io/core/language-codes/r/language-codes-full.csv"

    def fetch(self) -> list[dict[str, Any]]:
        """Download language codes from DataHub."""
        output_path = self.raw_dir / "languages.csv"

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

        for row in raw_data:
            alpha2 = row.get("alpha2", "").strip()
            alpha3b = row.get("alpha3_b", "").strip()
            alpha3t = row.get("alpha3_t", "").strip()
            english = row.get("English", "").strip()
            french = row.get("French", "").strip()

            if not english:
                continue

            lang_id = alpha2 if alpha2 else alpha3b
            if not lang_id:
                continue

            aliases = []
            if alpha2:
                aliases.append(alpha2)
            if alpha3b and alpha3b != alpha2:
                aliases.append(alpha3b)
            if alpha3t and alpha3t != alpha3b:
                aliases.append(alpha3t)
            if french and french.lower() != english.lower():
                aliases.append(french)

            entities.append(
                {
                    "id": lang_id,
                    "name": english,
                    "aliases": "|".join(aliases) if aliases else "",
                    "type": "language",
                }
            )

        return entities


def run(raw_dir=None, processed_dir=None):
    """Execute language data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("languages", raw_dir, processed_dir)

    fetcher = LanguagesFetcher(raw_dir, processed_dir)
    fetcher.run("languages.csv")


if __name__ == "__main__":
    run()

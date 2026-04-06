"""Ingestion script for NAICS/SIC industry codes."""

from typing import Any
import csv
import json
import requests

from .base import BaseFetcher, resolve_output_dirs
from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class IndustriesFetcher(BaseFetcher):
    """Fetch NAICS and SIC industry classification codes."""

    NAICS_URLS = [
        "https://raw.githubusercontent.com/erickogore/country-code-json/refs/heads/master/industry-codes.json",
        "https://raw.githubusercontent.com/datasets/industry-codes/master/data/industry-codes.csv",
    ]

    FALLBACK_NAICS = [
        {"Code": "11", "Title": "Agriculture, Forestry, Fishing and Hunting"},
        {"Code": "21", "Title": "Mining, Quarrying, and Oil and Gas Extraction"},
        {"Code": "22", "Title": "Utilities"},
        {"Code": "23", "Title": "Construction"},
        {"Code": "31-33", "Title": "Manufacturing"},
        {"Code": "42", "Title": "Wholesale Trade"},
        {"Code": "44-45", "Title": "Retail Trade"},
        {"Code": "48-49", "Title": "Transportation and Warehousing"},
        {"Code": "51", "Title": "Information"},
        {"Code": "52", "Title": "Finance and Insurance"},
        {"Code": "53", "Title": "Real Estate and Rental and Leasing"},
        {"Code": "54", "Title": "Professional, Scientific, and Technical Services"},
        {"Code": "55", "Title": "Management of Companies and Enterprises"},
        {"Code": "56", "Title": "Administrative and Support and Waste Management"},
        {"Code": "61", "Title": "Educational Services"},
        {"Code": "62", "Title": "Health Care and Social Assistance"},
        {"Code": "71", "Title": "Arts, Entertainment, and Recreation"},
        {"Code": "72", "Title": "Accommodation and Food Services"},
        {"Code": "81", "Title": "Other Services (except Public Administration)"},
    ]

    def fetch(self) -> list[dict[str, Any]]:
        """Download industry codes from various sources."""
        output_path = self.raw_dir / "naics_2022.json"

        for url in self.NAICS_URLS:
            try:
                if not output_path.exists():
                    response = requests.get(url, timeout=30)
                    response.raise_for_status()
                    content_type = response.headers.get("Content-Type", "")
                    if "json" not in content_type and "text" not in content_type:
                        raise ValueError(f"Unexpected Content-Type: {content_type}")
                    if len(response.content) > 10 * 1024 * 1024:
                        raise ValueError("Response exceeds 10MB size limit")
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(response.text)
                    break
            except (requests.RequestException, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"BLS fetch failed: {e}, using fallback data")
                with open(output_path, "w", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["SIC Code", "SIC Industry Title"]
                    )
                    writer.writeheader()
                    writer.writerows(self.FALLBACK_SIC)

        data = []
        with open(output_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)

        return data if data else self.FALLBACK_SIC

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []

        for row in raw_data:
            code = row.get("SIC Code", "").strip()
            title = row.get("SIC Industry Title", "").strip()

            if not code or not title:
                continue

            aliases = []
            if len(code) >= 2:
                aliases.append(code[:2])

            entities.append(
                {
                    "id": code,
                    "name": title,
                    "aliases": "|".join(aliases) if aliases else "",
                    "type": "industry",
                    "system": "SIC",
                }
            )

        return entities


def run(raw_dir=None, processed_dir=None):
    """Execute industry data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("industries", raw_dir, processed_dir)

    fetcher = IndustriesFetcher(raw_dir, processed_dir)
    fetcher.run("industries_naics.csv")

    sic_fetcher = SICFetcher(raw_dir, processed_dir)
    sic_fetcher.run("industries_sic.csv")


if __name__ == "__main__":
    run()

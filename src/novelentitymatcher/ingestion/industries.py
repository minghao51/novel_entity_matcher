"""Ingestion script for NAICS/SIC industry codes."""

import csv
import json
from typing import Any

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

    FALLBACK_SIC = [
        {"SIC Code": "0111", "SIC Industry Title": "Wheat"},
        {"SIC Code": "1311", "SIC Industry Title": "Crude Petroleum and Natural Gas"},
        {"SIC Code": "2711", "SIC Industry Title": "Newspapers"},
        {"SIC Code": "3571", "SIC Industry Title": "Electronic Computers"},
        {"SIC Code": "5812", "SIC Industry Title": "Eating Places"},
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
            except (
                requests.RequestException,
                ConnectionError,
                TimeoutError,
                ValueError,
            ) as e:
                logger.warning("Industry fetch failed for %s: %s", url, e)

        if not output_path.exists():
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(self.FALLBACK_NAICS, f, indent=2)
            return list(self.FALLBACK_NAICS)

        with open(output_path, "r", encoding="utf-8") as f:
            if output_path.suffix == ".json":
                payload = json.load(f)
                if isinstance(payload, list):
                    return payload
                raise ValueError("Expected list payload in NAICS source")

            reader = csv.DictReader(f)
            return list(reader)

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []

        for row in raw_data:
            code = str(row.get("Code") or row.get("SIC Code") or "").strip()
            title = str(row.get("Title") or row.get("SIC Industry Title") or "").strip()

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
    fallback_sic = IndustriesFetcher(raw_dir, processed_dir)
    fallback_sic.fetch = lambda: list(fallback_sic.FALLBACK_SIC)  # type: ignore[method-assign]
    fallback_sic.run("industries_sic.csv")


if __name__ == "__main__":
    run()

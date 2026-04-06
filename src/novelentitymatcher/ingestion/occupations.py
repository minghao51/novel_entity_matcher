"""Ingestion script for SOC occupation codes (O*NET)."""

from typing import Any
import csv
import requests

from .base import BaseFetcher, resolve_output_dirs
from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class OccupationsFetcher(BaseFetcher):
    """Fetch SOC occupation codes from O*NET."""

    ONET_URL = "https://www.onetcenter.org/dl/30_2/occupation_data.zip"

    FALLBACK_OCCUPATIONS = [
        {"O*NET-SOC Code": "11-0000", "Title": "Management Occupations"},
        {
            "O*NET-SOC Code": "13-0000",
            "Title": "Business and Financial Operations Occupations",
        },
        {"O*NET-SOC Code": "15-0000", "Title": "Computer and Mathematical Occupations"},
        {
            "O*NET-SOC Code": "17-0000",
            "Title": "Architecture and Engineering Occupations",
        },
        {
            "O*NET-SOC Code": "19-0000",
            "Title": "Life, Physical, and Social Science Occupations",
        },
        {
            "O*NET-SOC Code": "21-0000",
            "Title": "Community and Social Service Occupations",
        },
        {"O*NET-SOC Code": "23-0000", "Title": "Legal Occupations"},
        {
            "O*NET-SOC Code": "25-0000",
            "Title": "Educational Instruction and Library Occupations",
        },
        {
            "O*NET-SOC Code": "27-0000",
            "Title": "Arts, Design, Entertainment, Sports, and Media Occupations",
        },
        {
            "O*NET-SOC Code": "29-0000",
            "Title": "Healthcare Practitioners and Technical Occupations",
        },
        {"O*NET-SOC Code": "31-0000", "Title": "Healthcare Support Occupations"},
        {"O*NET-SOC Code": "33-0000", "Title": "Protective Service Occupations"},
        {
            "O*NET-SOC Code": "35-0000",
            "Title": "Food Preparation and Serving Related Occupations",
        },
        {
            "O*NET-SOC Code": "37-0000",
            "Title": "Building and Grounds Cleaning and Maintenance Occupations",
        },
        {"O*NET-SOC Code": "39-0000", "Title": "Personal Care and Service Occupations"},
        {"O*NET-SOC Code": "41-0000", "Title": "Sales and Related Occupations"},
        {
            "O*NET-SOC Code": "43-0000",
            "Title": "Office and Administrative Support Occupations",
        },
        {
            "O*NET-SOC Code": "45-0000",
            "Title": "Farming, Fishing, and Forestry Occupations",
        },
        {
            "O*NET-SOC Code": "47-0000",
            "Title": "Construction and Extraction Occupations",
        },
        {
            "O*NET-SOC Code": "49-0000",
            "Title": "Installation, Maintenance, and Repair Occupations",
        },
        {"O*NET-SOC Code": "51-0000", "Title": "Production Occupations"},
        {
            "O*NET-SOC Code": "53-0000",
            "Title": "Transportation and Material Moving Occupations",
        },
    ]

    def fetch(self) -> list[dict[str, Any]]:
        """Download occupation data from O*NET."""
        import zipfile
        import io

        output_path = self.raw_dir / "occupation_data.zip"

        if not output_path.exists():
            try:
                response = requests.get(self.ONET_URL, timeout=60)
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "zip" not in content_type and "octet-stream" not in content_type:
                    raise ValueError(f"Unexpected Content-Type: {content_type}")
                if len(response.content) > 100 * 1024 * 1024:
                    raise ValueError("Response exceeds 100MB size limit")
                with open(output_path, "wb") as f:
                    f.write(response.content)
            except (requests.RequestException, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"O*NET fetch failed: {e}, using fallback data")
                fallback_path = self.raw_dir / "occupation_data.txt"
                with open(fallback_path, "w", encoding="utf-8") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["O*NET-SOC Code", "Title"], delimiter="\t"
                    )
                    writer.writeheader()
                    writer.writerows(self.FALLBACK_OCCUPATIONS)
                return self.FALLBACK_OCCUPATIONS

        try:
            with zipfile.ZipFile(output_path, "r") as z:
                with z.open("occupation_data.txt") as f:
                    content = f.read().decode("utf-8")

            data = []
            reader = csv.DictReader(io.StringIO(content), delimiter="\t")
            for row in reader:
                data.append(row)

            return data if data else self.FALLBACK_OCCUPATIONS
        except (ValueError, OSError, KeyError, zipfile.BadZipFile) as e:
            logger.warning(f"Failed to parse O*NET data: {e}, using fallback")
            return self.FALLBACK_OCCUPATIONS

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []

        seen = set()
        for row in raw_data:
            code = row.get("O*NET-SOC Code", "").strip()
            title = row.get("Title", "").strip()

            if not code or not title or code in seen:
                continue
            seen.add(code)

            aliases = []
            major_group = ""
            if len(code) >= 2:
                major_group = code[:2]
                aliases.append(major_group)
            if len(code) >= 4:
                minor_group = code[:4]
                if minor_group != major_group:
                    aliases.append(minor_group)

            entities.append(
                {
                    "id": code,
                    "name": title,
                    "aliases": "|".join(aliases) if aliases else "",
                    "type": "occupation",
                    "system": "O*NET-SOC",
                }
            )

        return entities


class SOCDirectFetcher(BaseFetcher):
    """Fetch direct SOC codes from BLS."""

    SOC_URL = "https://www.bls.gov/soc/2018/home.htm"

    def fetch(self) -> list[dict[str, Any]]:
        """Fetch SOC data from BLS (placeholder - requires HTML parsing)."""
        response = requests.get(self.SOC_URL, timeout=60)
        response.raise_for_status()

        output_path = self.raw_dir / "soc_2018.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)

        return [{"source": "BLS SOC 2018", "file": str(output_path)}]

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process requires HTML parsing - return empty for now."""
        logger.info("Note: Full SOC processing requires HTML parsing")
        return []


def run(raw_dir=None, processed_dir=None):
    """Execute occupation data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("occupations", raw_dir, processed_dir)

    fetcher = OccupationsFetcher(raw_dir, processed_dir)
    fetcher.run("occupations.csv")


if __name__ == "__main__":
    run()

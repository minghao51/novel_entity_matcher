"""Ingestion script for university/institution data."""

from typing import Any
import json
import requests

from .base import BaseFetcher, resolve_output_dirs
from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)


class UniversitiesFetcher(BaseFetcher):
    """Fetch university/institution data from multiple sources."""

    FALLBACK_UNIVERSITIES = [
        {
            "name": "Harvard University",
            "aliases": "Harvard,Harvard Univ",
            "country": "US",
        },
        {
            "name": "Stanford University",
            "aliases": "Stanford,Stanford Univ",
            "country": "US",
        },
        {
            "name": "Massachusetts Institute of Technology",
            "aliases": "MIT,Massachusetts Institute of Technology",
            "country": "US",
        },
        {
            "name": "University of Cambridge",
            "aliases": "Cambridge University,Univ of Cambridge",
            "country": "GB",
        },
        {
            "name": "University of Oxford",
            "aliases": "Oxford University,Univ of Oxford",
            "country": "GB",
        },
        {"name": "Yale University", "aliases": "Yale,Yale Univ", "country": "US"},
        {
            "name": "Princeton University",
            "aliases": "Princeton,Princeton Univ",
            "country": "US",
        },
        {
            "name": "Columbia University",
            "aliases": "Columbia,Columbia Univ",
            "country": "US",
        },
        {
            "name": "University of Chicago",
            "aliases": "UChicago,Chicago University",
            "country": "US",
        },
        {
            "name": "University of Pennsylvania",
            "aliases": "UPenn,Penn",
            "country": "US",
        },
        {
            "name": "California Institute of Technology",
            "aliases": "Caltech,California Tech",
            "country": "US",
        },
        {
            "name": "Johns Hopkins University",
            "aliases": "JHU,Johns Hopkins",
            "country": "US",
        },
        {
            "name": "Cornell University",
            "aliases": "Cornell,Cornell Univ",
            "country": "US",
        },
        {
            "name": "University of Michigan",
            "aliases": "UMich,Michigan",
            "country": "US",
        },
        {"name": "University of Toronto", "aliases": "UofT,Toronto", "country": "CA"},
        {"name": "Duke University", "aliases": "Duke,Duke Univ", "country": "US"},
        {
            "name": "Northwestern University",
            "aliases": "Northwestern,Northwestern Univ",
            "country": "US",
        },
        {
            "name": "New York University",
            "aliases": "NYU,New York Univ",
            "country": "US",
        },
        {
            "name": "University of California, Berkeley",
            "aliases": "UC Berkeley,Berkeley",
            "country": "US",
        },
        {
            "name": "University of Tokyo",
            "aliases": "UTokyo,Tokyo Univ",
            "country": "JP",
        },
        {"name": "Kyoto University", "aliases": "Kyoto Univ,KGU", "country": "JP"},
        {
            "name": "University of Melbourne",
            "aliases": "UniMelb,Melbourne",
            "country": "AU",
        },
        {
            "name": "Australian National University",
            "aliases": "ANU,Australian National Univ",
            "country": "AU",
        },
        {
            "name": "ETH Zurich",
            "aliases": "Swiss Federal Institute of Technology Zurich",
            "country": "CH",
        },
        {
            "name": "University of Edinburgh",
            "aliases": "Edinburgh Univ,Edinburgh",
            "country": "GB",
        },
        {
            "name": "University of Manchester",
            "aliases": "Manchester Univ,UMIST",
            "country": "GB",
        },
        {
            "name": "University of Sydney",
            "aliases": "Sydney Univ,USYD",
            "country": "AU",
        },
        {
            "name": "University of Queensland",
            "aliases": "UQ,Queensland",
            "country": "AU",
        },
        {"name": "McGill University", "aliases": "McGill,McGill Univ", "country": "CA"},
        {
            "name": "University of British Columbia",
            "aliases": "UBC,Vancouver",
            "country": "CA",
        },
        {
            "name": "National University of Singapore",
            "aliases": "NUS,National Univ of Singapore",
            "country": "SG",
        },
        {
            "name": "Nanyang Technological University",
            "aliases": "NTU,Nanyang",
            "country": "SG",
        },
        {
            "name": "Technical University of Munich",
            "aliases": "TUM,Munich Tech",
            "country": "DE",
        },
        {
            "name": "Heidelberg University",
            "aliases": "Heidelberg Univ,Ruprecht Karls University",
            "country": "DE",
        },
        {
            "name": "University of Paris",
            "aliases": "Sorbonne,Paris University",
            "country": "FR",
        },
        {
            "name": "Sorbonne University",
            "aliases": "Universite Sorbonne",
            "country": "FR",
        },
        {"name": "Ecole Polytechnique", "aliases": "X,Polytechnique", "country": "FR"},
        {
            "name": "University of Hong Kong",
            "aliases": "HKU,Hong Kong Univ",
            "country": "HK",
        },
        {
            "name": "Chinese University of Hong Kong",
            "aliases": "CUHK,Chinese Univ HK",
            "country": "HK",
        },
        {
            "name": "Seoul National University",
            "aliases": "SNU,Seoul Natl Univ",
            "country": "KR",
        },
        {"name": "Korea University", "aliases": "KU,Korea Univ", "country": "KR"},
        {
            "name": "KAIST",
            "aliases": "Korea Advanced Institute of Science and Technology",
            "country": "KR",
        },
        {"name": "Tsinghua University", "aliases": "Tsinghua,THU", "country": "CN"},
        {
            "name": "Peking University",
            "aliases": "Beijing University,PKU",
            "country": "CN",
        },
        {"name": "Fudan University", "aliases": "Fudan,Fudan Univ", "country": "CN"},
        {
            "name": "Shanghai Jiao Tong University",
            "aliases": "SJTU,Shanghai Jiao Tong",
            "country": "CN",
        },
        {
            "name": "Zhejiang University",
            "aliases": "ZJU,Zhejiang Univ",
            "country": "CN",
        },
        {
            "name": "Indian Institute of Technology Bombay",
            "aliases": "IIT Bombay,IITB",
            "country": "IN",
        },
        {
            "name": "Indian Institute of Technology Delhi",
            "aliases": "IIT Delhi,IITD",
            "country": "IN",
        },
        {
            "name": "Indian Institute of Science",
            "aliases": "IISc,Indian Institute of Science",
            "country": "IN",
        },
    ]

    def fetch(self) -> list[dict[str, Any]]:
        """Download university data."""
        output_path = self.raw_dir / "universities.json"

        wikidata_url = "https://query.wikidata.org/sparql"
        query = """
        SELECT ?university ?name ?country ?aliases WHERE {
          ?university wdt:P31 wd:Q3918.
          ?university wdt:P17 ?country.
          OPTIONAL { ?university rdfs:label ?name filter(lang(?name) = "en"). }
          OPTIONAL { ?university skos:altLabel ?aliases filter(lang(?aliases) = "en"). }
        } LIMIT 500
        """

        headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "SemanticMatcher/1.0",
        }

        if not output_path.exists():
            try:
                response = requests.get(
                    wikidata_url, params={"query": query}, headers=headers, timeout=120
                )
                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "")
                if "json" not in content_type and "sparql" not in content_type:
                    raise ValueError(f"Unexpected Content-Type: {content_type}")
                if len(response.content) > 50 * 1024 * 1024:
                    raise ValueError("Response exceeds 50MB size limit")
                results = response.json()
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(results, f)
            except (requests.RequestException, ConnectionError, TimeoutError, ValueError) as e:
                logger.warning(f"Wikidata fetch failed: {e}, using fallback data")
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump({"fallback": True, "data": self.FALLBACK_UNIVERSITIES}, f)

        try:
            with open(output_path, "r", encoding="utf-8") as f:
                content = json.load(f)
                if "fallback" in content:
                    return content.get("data", [])
                data = content.get("results", {}).get("bindings", [])
                return data if data else self.FALLBACK_UNIVERSITIES
        except (json.JSONDecodeError, KeyError, ValueError, OSError):
            return self.FALLBACK_UNIVERSITIES

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []
        seen = set()

        for item in raw_data:
            if isinstance(item, dict):
                if "name" in item:
                    name = item.get("name", "").strip()
                    aliases = item.get("aliases", "").strip()
                    country = item.get("country", "").strip()
                else:
                    name = (
                        item.get("name", {}).get("value", "").strip()
                        if isinstance(item.get("name"), dict)
                        else str(item.get("name", ""))
                    )
                    country_raw = (
                        item.get("country", {}).get("value", "")
                        if isinstance(item.get("country"), dict)
                        else str(item.get("country", ""))
                    )
                    country = country_raw.split("/")[-1] if country_raw else ""
                    aliases = ""
            else:
                continue

            if not name or name in seen:
                continue
            seen.add(name)

            aliases_list = []
            if aliases:
                for a in aliases.split(","):
                    if a.strip().lower() != name.lower():
                        aliases_list.append(a.strip())

            entities.append(
                {
                    "id": name.lower().replace(" ", "_").replace(",", ""),
                    "name": name,
                    "aliases": "|".join(aliases_list),
                    "type": "university",
                    "country": country,
                }
            )

        return entities


class TopUniversitiesFetcher(BaseFetcher):
    """Fetch top universities from Wikipedia/list sources."""

    SOURCE_URL = "https://en.wikipedia.org/wiki/List_of_oldest_universities"

    def fetch(self) -> list[dict[str, Any]]:
        """Download from Wikipedia (requires HTML parsing)."""
        output_path = self.raw_dir / "top_universities.json"

        if not output_path.exists():
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        with open(output_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process university data."""
        entities = []

        for item in raw_data:
            name = item.get("name", "").strip()
            if not name:
                continue

            entities.append(
                {
                    "id": item.get("id", name.lower().replace(" ", "_")),
                    "name": name,
                    "aliases": item.get("aliases", ""),
                    "type": "university",
                }
            )

        return entities


def run(raw_dir=None, processed_dir=None):
    """Execute university data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("universities", raw_dir, processed_dir)

    fetcher = UniversitiesFetcher(raw_dir, processed_dir)
    fetcher.run("universities.csv")


if __name__ == "__main__":
    run()

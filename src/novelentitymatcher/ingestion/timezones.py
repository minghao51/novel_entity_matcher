"""Ingestion script for IANA timezone data."""

from typing import Any
import json

from .base import BaseFetcher, _fetch_url, resolve_output_dirs


class TimezonesFetcher(BaseFetcher):
    """Fetch IANA timezone database."""

    TZ_URL = "https://raw.githubusercontent.com/eggert/tz/main/zone.tab"

    def fetch(self) -> list[dict[str, Any]]:
        """Download timezone data from IANA."""
        output_path = self.raw_dir / "zone.tab"

        if not output_path.exists():
            _fetch_url(
                self.TZ_URL,
                output_path,
                expected_content_type="text",
                max_bytes=5 * 1024 * 1024,
            )

        data = []
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    data.append(
                        {
                            "country": parts[0],
                            "coordinates": parts[1],
                            "timezone": parts[2],
                        }
                    )

        return data

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format with aliases."""
        entities = []

        for item in raw_data:
            tz = item.get("timezone", "").strip()
            if not tz:
                continue

            parts = tz.split("/")
            name = parts[-1] if len(parts) > 1 else tz

            aliases = [tz.replace("_", " ")]
            if len(parts) > 1:
                region = parts[0]
                aliases.append(region)
                aliases.append(region.replace("_", " "))

            entities.append(
                {
                    "id": tz,
                    "name": name.replace("_", " "),
                    "aliases": "|".join(aliases),
                    "type": "timezone",
                    "country_code": item.get("country", ""),
                }
            )

        return entities


class WorldTimeAPIFetcher(BaseFetcher):
    """Fetch timezone offset data from WorldTimeAPI."""

    TZ_LIST_URL = "https://worldtimeapi.org/api/timezone"

    def fetch(self) -> list[dict[str, Any]]:
        """Download timezone list."""
        output_path = self.raw_dir / "timezone_list.json"

        if not output_path.exists():
            _fetch_url(
                self.TZ_LIST_URL,
                output_path,
                expected_content_type="json",
                max_bytes=5 * 1024 * 1024,
            )

        with open(output_path, "r", encoding="utf-8") as f:
            timezones = json.load(f)
        return [{"timezone": tz} for tz in timezones]

    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Convert to standardized format."""
        entities = []

        for item in raw_data:
            tz = item.get("timezone", "").strip()
            if not tz:
                continue

            parts = tz.split("/")
            name = parts[-1] if len(parts) > 1 else tz

            aliases = [tz.replace("_", " ")]
            if len(parts) > 1:
                aliases.append(parts[0])

            entities.append(
                {
                    "id": tz,
                    "name": name.replace("_", " "),
                    "aliases": "|".join(aliases),
                    "type": "timezone",
                }
            )

        return entities


def run(raw_dir=None, processed_dir=None):
    """Execute timezone data ingestion."""
    raw_dir, processed_dir = resolve_output_dirs("timezones", raw_dir, processed_dir)

    fetcher = TimezonesFetcher(raw_dir, processed_dir)
    fetcher.run("timezones.csv")


if __name__ == "__main__":
    run()

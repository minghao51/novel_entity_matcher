"""Base classes for data ingestion."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union
import asyncio
import csv

from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


class BaseFetcher(ABC):
    """Base class for fetching and processing external datasets."""

    def __init__(self, raw_dir: Path, processed_dir: Path):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch(self) -> list[dict[str, Any]]:
        """Fetch raw data from source."""

    @abstractmethod
    def process(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Process raw data into standardized format."""

    def save_csv(self, data: list[dict[str, Any]], filename: str) -> Path:
        """Save data to CSV file."""
        if not data:
            raise ValueError("No data to save")

        output_path = self.processed_dir / filename
        fieldnames = list(data[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

        return output_path

    def run(self, output_filename: str) -> Path:
        """Execute full ingestion pipeline."""
        logger.info(f"Fetching {self.__class__.__name__} data...")
        raw_data = self.fetch()

        logger.info(f"Processing {len(raw_data)} records...")
        processed_data = self.process(raw_data)

        logger.info(f"Saving to {output_filename}...")
        output_path = self.save_csv(processed_data, output_filename)

        logger.info(f"Done! Saved {len(processed_data)} records to {output_path}")
        return output_path

    async def run_async(self, output_filename: str, semaphore: Optional[asyncio.Semaphore] = None) -> Path:
        """Execute full ingestion pipeline asynchronously with rate limiting."""
        async def _fetch():
            if semaphore:
                async with semaphore:
                    return self.fetch()
            return self.fetch()

        logger.info(f"Fetching {self.__class__.__name__} data...")
        loop = asyncio.get_running_loop()
        raw_data = await loop.run_in_executor(None, _fetch)

        logger.info(f"Processing {len(raw_data)} records...")
        processed_data = self.process(raw_data)

        logger.info(f"Saving to {output_filename}...")
        output_path = self.save_csv(processed_data, output_filename)

        logger.info(f"Done! Saved {len(processed_data)} records to {output_path}")
        return output_path


def resolve_output_dirs(
    dataset: str,
    raw_dir: Optional[PathLike] = None,
    processed_dir: Optional[PathLike] = None,
) -> tuple[Path, Path]:
    """Resolve per-dataset output directories without relying on repo layout."""
    raw_base = Path(raw_dir) if raw_dir is not None else Path.cwd() / "data" / "raw"
    processed_base = (
        Path(processed_dir)
        if processed_dir is not None
        else Path.cwd() / "data" / "processed"
    )
    return raw_base / dataset, processed_base / dataset


async def run_concurrent(
    fetchers: list[tuple[BaseFetcher, str]],
    max_concurrent: int = 4,
) -> list[Path]:
    """Run multiple fetchers concurrently with rate limiting.

    Args:
        fetchers: List of (BaseFetcher, output_filename) tuples.
        max_concurrent: Maximum number of concurrent fetch operations.

    Returns:
        List of paths to saved CSV files.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        fetcher.run_async(output_filename, semaphore)
        for fetcher, output_filename in fetchers
    ]
    return await asyncio.gather(*tasks)


def run_all_concurrent(
    fetchers: list[tuple[BaseFetcher, str]],
    max_concurrent: int = 4,
) -> list[Path]:
    """Synchronous wrapper for run_concurrent.

    Args:
        fetchers: List of (BaseFetcher, output_filename) tuples.
        max_concurrent: Maximum number of concurrent fetch operations.

    Returns:
        List of paths to saved CSV files.
    """
    return asyncio.run(run_concurrent(fetchers, max_concurrent))

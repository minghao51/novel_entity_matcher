"""Base classes for data ingestion."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Union
import asyncio
import csv
import requests

from novelentitymatcher.utils.logging_config import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]

DEFAULT_MAX_BYTES = 50 * 1024 * 1024


def _fetch_url(
    url: str,
    output_path: Path,
    expected_content_type: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    timeout: int = 60,
) -> None:
    """Fetch a URL and write its content to a file with validation.

    Args:
        url: URL to fetch.
        output_path: Path to write the response body.
        expected_content_type: If set, the response Content-Type header must
            contain this substring or a ValueError is raised.
        max_bytes: Maximum allowed response body size in bytes. Defaults to 50MB.
        timeout: Request timeout in seconds.

    Raises:
        ValueError: If Content-Type is unexpected or response exceeds max_bytes.
        requests.HTTPError: If the response status is 4xx/5xx.
    """
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    if expected_content_type:
        content_type = response.headers.get("Content-Type", "")
        if expected_content_type not in content_type:
            raise ValueError(
                f"Unexpected Content-Type '{content_type}' for {url}; "
                f"expected '{expected_content_type}'"
            )

    if len(response.content) > max_bytes:
        raise ValueError(
            f"Response from {url} exceeds {max_bytes} bytes "
            f"({len(response.content)} bytes received)"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(response.text)


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

    def save_csv(
        self, data: list[dict[str, Any]], filename: str, batch_size: int = 1000
    ) -> Path:
        """Save data to CSV file with optional batched writes.

        Args:
            data: List of records to save
            filename: Output filename
            batch_size: Number of records to buffer before flushing (default: 1000)

        Returns:
            Path to saved file
        """
        if not data:
            raise ValueError("No data to save")

        output_path = self.processed_dir / filename
        fieldnames = list(data[0].keys())

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            # Write in batches to reduce I/O operations for large datasets
            for i in range(0, len(data), batch_size):
                batch = data[i : i + batch_size]
                writer.writerows(batch)
                f.flush()  # Ensure data is written to disk

        return output_path

    def run(self, output_filename: str, batch_size: int = 1000) -> Path:
        """Execute full ingestion pipeline.

        Args:
            output_filename: Output filename
            batch_size: Batch size for CSV writes (default: 1000)

        Returns:
            Path to saved file
        """
        logger.info(f"Fetching {self.__class__.__name__} data...")
        raw_data = self.fetch()

        logger.info(f"Processing {len(raw_data)} records...")
        processed_data = self.process(raw_data)

        logger.info(f"Saving to {output_filename} (batch_size={batch_size})...")
        output_path = self.save_csv(
            processed_data, output_filename, batch_size=batch_size
        )

        logger.info(f"Done! Saved {len(processed_data)} records to {output_path}")
        return output_path

    async def run_async(
        self,
        output_filename: str,
        semaphore: Optional[asyncio.Semaphore] = None,
        batch_size: int = 1000,
    ) -> Path:
        """Execute full ingestion pipeline asynchronously with rate limiting.

        Args:
            output_filename: Output filename
            semaphore: Optional semaphore for rate limiting
            batch_size: Batch size for CSV writes (default: 1000)

        Returns:
            Path to saved file
        """

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

        logger.info(f"Saving to {output_filename} (batch_size={batch_size})...")
        output_path = self.save_csv(
            processed_data, output_filename, batch_size=batch_size
        )

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
    batch_size: int = 1000,
) -> list[Path]:
    """Run multiple fetchers concurrently with rate limiting.

    Args:
        fetchers: List of (BaseFetcher, output_filename) tuples.
        max_concurrent: Maximum number of concurrent fetch operations.
        batch_size: Batch size for CSV writes (default: 1000)

    Returns:
        List of paths to saved CSV files.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    tasks = [
        fetcher.run_async(output_filename, semaphore, batch_size)
        for fetcher, output_filename in fetchers
    ]
    return await asyncio.gather(*tasks)


def run_all_concurrent(
    fetchers: list[tuple[BaseFetcher, str]],
    max_concurrent: int = 4,
    batch_size: int = 1000,
) -> list[Path]:
    """Synchronous wrapper for run_concurrent.

    Args:
        fetchers: List of (BaseFetcher, output_filename) tuples.
        max_concurrent: Maximum number of concurrent fetch operations.
        batch_size: Batch size for CSV writes (default: 1000)

    Returns:
        List of paths to saved CSV files.
    """
    return asyncio.run(run_concurrent(fetchers, max_concurrent, batch_size))

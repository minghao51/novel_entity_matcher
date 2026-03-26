"""Async dataset loader for HuggingFace benchmarks with parquet caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from .registry import (
    CacheConfig,
    DATASET_REGISTRY,
    DatasetConfig,
    DEFAULT_CACHE_DIR,
    get_dataset_config,
)

logger = logging.getLogger(__name__)

try:
    from datasets import Dataset, load_dataset

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("datasets library not available. Install with: pip install datasets")


class DatasetLoader:
    def __init__(
        self,
        cache_dir: Path | None = None,
        cache_config: CacheConfig | None = None,
    ):
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_config = cache_config or CacheConfig()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_dataset_version(self, config: DatasetConfig) -> str | None:
        if not HF_AVAILABLE:
            return None
        try:
            import huggingface_hub

            info = huggingface_hub.get_dataset_config_info(config.hf_path)
            return info.version if hasattr(info, "version") else str(hash(info.sha))
        except Exception:
            return None

    def _compute_version_hash(self, config: DatasetConfig) -> str:
        version = self._get_dataset_version(config) or "unknown"
        return hashlib.sha256(
            f"{config.hf_path}:{config.split}:{version}".encode()
        ).hexdigest()[:12]

    def _load_metadata(self, config: DatasetConfig) -> dict[str, Any] | None:
        if not config.metadata_path.exists():
            return None
        with open(config.metadata_path) as f:
            return json.load(f)

    def _save_metadata(self, config: DatasetConfig, metadata: dict[str, Any]) -> None:
        config.cache_path.mkdir(parents=True, exist_ok=True)
        with open(config.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _is_cache_valid(self, config: DatasetConfig) -> bool:
        if self.cache_config.version_policy != "on_version_change":
            return True
        metadata = self._load_metadata(config)
        if metadata is None:
            return False
        current_hash = self._compute_version_hash(config)
        return metadata.get("version_hash") == current_hash

    def _convert_to_parquet(
        self, dataset: Dataset, config: DatasetConfig
    ) -> dict[str, Path]:
        if config.has_pairs:
            return self._convert_pairs_to_parquet(dataset, config)
        else:
            return self._convert_classification_to_parquet(dataset, config)

    def _convert_pairs_to_parquet(
        self, dataset: Dataset, config: DatasetConfig
    ) -> dict[str, Path]:
        if isinstance(dataset, dict):
            splits = {}
            for split_name, split_ds in dataset.items():
                if hasattr(split_ds, "to_pandas"):
                    df = split_ds.to_pandas()
                else:
                    df = pd.DataFrame(split_ds)

                output_path = config.cache_path / f"{split_name}.parquet"
                config.cache_path.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path, index=False)
                splits[split_name] = output_path
            return splits
        else:
            if hasattr(dataset, "to_pandas"):
                df = dataset.to_pandas()
            else:
                df = pd.DataFrame(dataset)

            output_path = config.cache_path / f"{config.split}.parquet"
            config.cache_path.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            return {config.split: output_path}

    def _convert_classification_to_parquet(
        self, dataset: Dataset, config: DatasetConfig
    ) -> dict[str, Path]:
        if isinstance(dataset, dict):
            splits = {}
            for split_name, split_ds in dataset.items():
                if hasattr(split_ds, "to_pandas"):
                    df = split_ds.to_pandas()
                else:
                    df = pd.DataFrame(split_ds)

                if config.max_samples and len(df) > config.max_samples:
                    df = df.sample(n=config.max_samples, random_state=42)

                output_path = config.cache_path / f"{split_name}.parquet"
                config.cache_path.mkdir(parents=True, exist_ok=True)
                df.to_parquet(output_path, index=False)
                splits[split_name] = output_path
            return splits
        else:
            if hasattr(dataset, "to_pandas"):
                df = dataset.to_pandas()
            else:
                df = pd.DataFrame(dataset)

            if config.max_samples and len(df) > config.max_samples:
                df = df.sample(n=config.max_samples, random_state=42)

            output_path = config.cache_path / f"{config.split}.parquet"
            config.cache_path.mkdir(parents=True, exist_ok=True)
            df.to_parquet(output_path, index=False)
            return {config.split: output_path}

    async def aload_dataset(
        self, name: str, force_redownload: bool = False
    ) -> dict[str, Any]:
        config = get_dataset_config(name)
        if config is None:
            raise ValueError(
                f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}"
            )

        if not force_redownload and self._is_cache_valid(config):
            logger.info(f"Loading {name} from cache")
            return self._load_from_cache(config)

        if config.download_url:
            logger.info(f"Downloading {name} from {config.download_url}")
            parquet_paths = await self._download_er_dataset(config)
        else:
            if not HF_AVAILABLE:
                raise ImportError(
                    "datasets library required. Install with: pip install datasets"
                )
            logger.info(f"Downloading {name} from HuggingFace: {config.hf_path}")

            def _load_sync():
                return load_dataset(config.hf_path)

            loop = asyncio.get_event_loop()
            dataset = await loop.run_in_executor(None, _load_sync)
            parquet_paths = self._convert_to_parquet(dataset, config)

        metadata = {
            "name": config.name,
            "hf_path": config.hf_path,
            "split": config.split,
            "version_hash": self._compute_version_hash(config),
            "num_rows": sum(
                pd.read_parquet(p).shape[0] for p in parquet_paths.values()
            ),
            "parquet_files": {k: str(v) for k, v in parquet_paths.items()},
        }
        self._save_metadata(config, metadata)

        return self._load_from_cache(config)

    async def _download_er_dataset(self, config: DatasetConfig) -> dict[str, Path]:
        import requests
        from io import StringIO

        base_url = config.download_url

        def _fetch_csv(path: str) -> pd.DataFrame:
            url = f"{base_url}/{path}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))

        def _fetch_all():
            tableA = _fetch_csv("tableA.csv")
            tableB = _fetch_csv("tableB.csv")
            test_pairs = _fetch_csv("test.csv")
            return tableA, tableB, test_pairs

        loop = asyncio.get_event_loop()
        tableA, tableB, test_pairs = await loop.run_in_executor(None, _fetch_all)

        merged = test_pairs.merge(
            tableA.rename(columns=lambda c: f"left_{c}" if c != "id" else "ltable_id"),
            on="ltable_id",
            how="left",
        ).merge(
            tableB.rename(columns=lambda c: f"right_{c}" if c != "id" else "rtable_id"),
            on="rtable_id",
            how="left",
        )

        name_col = (
            [c for c in tableA.columns if c != "id"][0]
            if len(tableA.columns) > 1
            else "name"
        )
        right_name_col = (
            [c for c in tableB.columns if c != "id"][0]
            if len(tableB.columns) > 1
            else "name"
        )

        merged["left"] = merged[f"left_{name_col}"].fillna("")
        merged["right"] = merged[f"right_{right_name_col}"].fillna("")
        merged["label"] = merged["label"].astype(int)

        output_df = merged[["left", "right", "label"]].copy()

        output_path = config.cache_path / f"{config.split}.parquet"
        config.cache_path.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_path, index=False)

        return {config.split: output_path}

    def _load_from_cache(self, config: DatasetConfig) -> dict[str, Any]:
        metadata = self._load_metadata(config)
        if metadata is None:
            raise FileNotFoundError(
                f"No cache found for {config.name}. Run aload_dataset first."
            )

        result = {
            "name": config.name,
            "task_type": config.task_type,
            "num_classes": config.num_classes,
            "classes": config.classes,
            "metadata": metadata,
        }

        parquet_files = metadata.get("parquet_files", {})
        for split_name, path_str in parquet_files.items():
            result[split_name] = pd.read_parquet(Path(path_str))

        return result

    async def aload_all(
        self,
        datasets: list[str] | None = None,
        force_redownload: bool = False,
    ) -> dict[str, dict[str, Any]]:
        if datasets is None:
            datasets = list(DATASET_REGISTRY.keys())

        tasks = [
            self.aload_dataset(name, force_redownload=force_redownload)
            for name in datasets
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: Dict[str, Any] = {}
        for name, result in zip(datasets, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to load {name}: {result}")
                output[name] = {"error": str(result)}
            else:
                output[name] = result

        return output

    def load_dataset(self, name: str, force_redownload: bool = False) -> dict[str, Any]:
        return asyncio.run(self.aload_dataset(name, force_redownload))

    def load_all(
        self,
        datasets: list[str] | None = None,
        force_redownload: bool = False,
    ) -> dict[str, dict[str, Any]]:
        return asyncio.run(self.aload_all(datasets, force_redownload))

    def get_cached_datasets(self) -> list[str]:
        cached = []
        for name in DATASET_REGISTRY:
            config = get_dataset_config(name)
            if config and config.metadata_path.exists():
                cached.append(name)
        return cached

    def clear_cache(self, name: str | None = None) -> None:
        if name:
            config = get_dataset_config(name)
            if config and config.cache_path.exists():
                import shutil

                shutil.rmtree(config.cache_path)
                logger.info(f"Cleared cache for {name}")
        else:
            if self.cache_dir.exists():
                import shutil

                shutil.rmtree(self.cache_dir)
                logger.info("Cleared all caches")

"""Dataset registry for HuggingFace benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_CACHE_DIR = Path("data/hf_benchmarks")
DEFAULT_MAX_SAMPLES = 1_000_000

DEEPMATCHER_DATA_BASE = "http://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data"


@dataclass
class DatasetConfig:
    name: str
    hf_path: str
    task_type: Literal["entity_matching", "classification", "novelty"]
    num_classes: int | None = None
    classes: list[str] | None = None
    split: str = "test"
    max_samples: int = DEFAULT_MAX_SAMPLES
    convert_to_entities: bool = True
    has_pairs: bool = False
    pair_columns: dict[str, str] | None = None
    text_column: str = "text"
    label_column: str = "label"
    cache_dir: Path = DEFAULT_CACHE_DIR
    download_url: str | None = None

    @property
    def cache_path(self) -> Path:
        return self.cache_dir / self.name

    @property
    def metadata_path(self) -> Path:
        return self.cache_path / "metadata.json"

    @property
    def version_key(self) -> str:
        return f"{self.hf_path.replace('/', '_')}_{self.split}"

    @property
    def is_hf_native(self) -> bool:
        return self.download_url is None


@dataclass
class CacheConfig:
    version_policy: Literal["on_version_change"] = "on_version_change"
    max_samples: int = DEFAULT_MAX_SAMPLES
    storage_dir: Path = DEFAULT_CACHE_DIR


ENTITY_RESOLUTION_DATASETS: dict[str, DatasetConfig] = {
    "walmart_amazon": DatasetConfig(
        name="walmart_amazon",
        hf_path="walmart_amazon",
        task_type="entity_matching",
        download_url=f"{DEEPMATCHER_DATA_BASE}/Structured/Walmart-Amazon/exp_data",
        has_pairs=True,
        pair_columns={"left": "ltable_name", "right": "rtable_name", "label": "label"},
        split="test",
    ),
    "amazon_google": DatasetConfig(
        name="amazon_google",
        hf_path="amazon_google",
        task_type="entity_matching",
        download_url=f"{DEEPMATCHER_DATA_BASE}/Structured/Amazon-Google/exp_data",
        has_pairs=True,
        pair_columns={"left": "ltable_name", "right": "rtable_name", "label": "label"},
        split="test",
    ),
    "fodors_zagats": DatasetConfig(
        name="fodors_zagats",
        hf_path="fodors_zagats",
        task_type="entity_matching",
        download_url=f"{DEEPMATCHER_DATA_BASE}/Structured/Fodors-Zagats/exp_data",
        has_pairs=True,
        pair_columns={"left": "ltable_name", "right": "rtable_name", "label": "label"},
        split="test",
    ),
    "beer": DatasetConfig(
        name="beer",
        hf_path="beer",
        task_type="entity_matching",
        download_url=f"{DEEPMATCHER_DATA_BASE}/Structured/Beer/exp_data",
        has_pairs=True,
        pair_columns={"left": "ltable_name", "right": "rtable_name", "label": "label"},
        split="test",
    ),
    "dblp_acm": DatasetConfig(
        name="dblp_acm",
        hf_path="dblp_acm",
        task_type="entity_matching",
        download_url=f"{DEEPMATCHER_DATA_BASE}/Structured/DBLP-ACM/exp_data",
        has_pairs=True,
        pair_columns={"left": "ltable_name", "right": "rtable_name", "label": "label"},
        split="test",
    ),
    "dblp_googlescholar": DatasetConfig(
        name="dblp_googlescholar",
        hf_path="dblp_googlescholar",
        task_type="entity_matching",
        download_url=f"{DEEPMATCHER_DATA_BASE}/Structured/DBLP-GoogleScholar/exp_data",
        has_pairs=True,
        pair_columns={"left": "ltable_name", "right": "rtable_name", "label": "label"},
        split="test",
    ),
    "itunes_amazon": DatasetConfig(
        name="itunes_amazon",
        hf_path="itunes_amazon",
        task_type="entity_matching",
        download_url=f"{DEEPMATCHER_DATA_BASE}/Structured/iTunes-Amazon/exp_data",
        has_pairs=True,
        pair_columns={"left": "ltable_name", "right": "rtable_name", "label": "label"},
        split="test",
    ),
}

CLASSIFICATION_DATASETS: dict[str, DatasetConfig] = {
    "ag_news": DatasetConfig(
        name="ag_news",
        hf_path="ag_news",
        task_type="classification",
        num_classes=4,
        classes=["World", "Sports", "Business", "Sci/Tech"],
        label_column="label",
        text_column="text",
        split="test",
        has_pairs=False,
    ),
    "yahoo_answers": DatasetConfig(
        name="yahoo_answers",
        hf_path="yahoo_answers_topics",
        task_type="classification",
        num_classes=10,
        classes=[
            "Society",
            "Science",
            "Health",
            "Education",
            "Computer",
            "Sports",
            "Business",
            "Entertainment",
            "Music",
            "Family",
        ],
        label_column="topic",
        text_column="question_title",
        split="test",
        max_samples=500_000,
        has_pairs=False,
    ),
    "goemotions": DatasetConfig(
        name="goemotions",
        hf_path="go_emotions",
        task_type="classification",
        num_classes=28,
        classes=[
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ],
        label_column="labels",
        text_column="text",
        split="test",
        max_samples=100_000,
        has_pairs=False,
    ),
}

NOVELTY_DATASETS: dict[str, DatasetConfig] = {
    "ag_news_novelty": DatasetConfig(
        name="ag_news_novelty",
        hf_path="ag_news",
        task_type="novelty",
        num_classes=4,
        classes=["World", "Sports", "Business", "Sci/Tech"],
        label_column="label",
        text_column="text",
        split="test",
        convert_to_entities=True,
        has_pairs=False,
    ),
    "goemotions_novelty": DatasetConfig(
        name="goemotions_novelty",
        hf_path="go_emotions",
        task_type="novelty",
        num_classes=28,
        classes=[
            "admiration",
            "amusement",
            "anger",
            "annoyance",
            "approval",
            "caring",
            "confusion",
            "curiosity",
            "desire",
            "disappointment",
            "disapproval",
            "disgust",
            "embarrassment",
            "excitement",
            "fear",
            "gratitude",
            "grief",
            "joy",
            "love",
            "nervousness",
            "optimism",
            "pride",
            "realization",
            "relief",
            "remorse",
            "sadness",
            "surprise",
            "neutral",
        ],
        label_column="labels",
        text_column="text",
        split="test",
        convert_to_entities=True,
        has_pairs=False,
    ),
}

DATASET_REGISTRY: dict[str, DatasetConfig] = {
    **ENTITY_RESOLUTION_DATASETS,
    **CLASSIFICATION_DATASETS,
    **NOVELTY_DATASETS,
}


def get_dataset_config(name: str) -> DatasetConfig | None:
    return DATASET_REGISTRY.get(name)


def get_datasets_by_task(
    task_type: Literal["entity_matching", "classification", "novelty"],
) -> dict[str, DatasetConfig]:
    return {
        name: config
        for name, config in DATASET_REGISTRY.items()
        if config.task_type == task_type
    }


def get_default_datasets() -> dict[str, DatasetConfig]:
    return {
        name: config
        for name, config in DATASET_REGISTRY.items()
        if config.max_samples <= DEFAULT_MAX_SAMPLES
    }

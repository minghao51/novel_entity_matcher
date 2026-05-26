import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

PathLike = str | Path

MODEL_SPECS = {
    "potion-8m": {
        "name": "minishlab/potion-base-8M",
        "backend": "static",
        "supports_training": False,
        "language": "en",
    },
    "potion-32m": {
        "name": "minishlab/potion-base-32M",
        "backend": "static",
        "supports_training": False,
        "language": "en",
    },
    "mrl-en": {
        "name": "RikkaBotan/stable-static-embedding-fast-retrieval-mrl-en",
        "backend": "static",
        "supports_training": False,
        "language": "en",
    },
    "mrl-multi": {
        "name": "sentence-transformers/static-similarity-mrl-multilingual-v1",
        "backend": "static",
        "supports_training": False,
        "language": "multilingual",
    },
    "bge-base": {
        "name": "BAAI/bge-base-en-v1.5",
        "backend": "sentence-transformers",
        "supports_training": True,
        "language": "en",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "backend": "sentence-transformers",
        "supports_training": True,
        "language": "multilingual",
    },
    "nomic": {
        "name": "nomic-ai/nomic-embed-text-v1",
        "backend": "sentence-transformers",
        "supports_training": True,
        "language": "en",
    },
    "mpnet": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "backend": "sentence-transformers",
        "supports_training": True,
        "language": "en",
    },
    "minilm": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "backend": "sentence-transformers",
        "supports_training": True,
        "language": "en",
    },
    "distilbert": {
        "name": "distilbert-base-uncased",
        "backend": "bert",
        "supports_training": True,
        "language": "en",
        "model_type": "bert",
        "params": "66M",
        "speed": "fast",
        "accuracy": "high",
    },
    "tinybert": {
        "name": "huawei-noah/TinyBERT_General_4L_312D",
        "backend": "bert",
        "supports_training": True,
        "language": "en",
        "model_type": "bert",
        "params": "4.4M",
        "speed": "very_fast",
        "accuracy": "medium",
    },
    "roberta-base": {
        "name": "roberta-base",
        "backend": "bert",
        "supports_training": True,
        "language": "en",
        "model_type": "bert",
        "params": "125M",
        "speed": "medium",
        "accuracy": "very_high",
    },
    "deberta-v3": {
        "name": "microsoft/deberta-v3-base",
        "backend": "bert",
        "supports_training": True,
        "language": "en",
        "model_type": "bert",
        "params": "184M",
        "speed": "slow",
        "accuracy": "state_of_the_art",
    },
    "bert-multilingual": {
        "name": "bert-base-multilingual-cased",
        "backend": "bert",
        "supports_training": True,
        "language": "multilingual",
        "model_type": "bert",
        "params": "179M",
        "speed": "slow",
        "accuracy": "high",
    },
}

STATIC_MODEL_REGISTRY = {
    alias: spec["name"]
    for alias, spec in MODEL_SPECS.items()
    if spec["backend"] == "static"
}

DYNAMIC_MODEL_REGISTRY = {
    alias: spec["name"]
    for alias, spec in MODEL_SPECS.items()
    if spec["backend"] != "static"
}

MODEL_REGISTRY = {alias: spec["name"] for alias, spec in MODEL_SPECS.items()}
RETRIEVAL_DEFAULT_MODEL = "potion-32m"
TRAINING_DEFAULT_MODEL = "mpnet"
BERT_DEFAULT_MODEL = "distilbert"
MODEL_REGISTRY["default"] = MODEL_SPECS[RETRIEVAL_DEFAULT_MODEL]["name"]

RERANKER_REGISTRY = {
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "bge-large": "BAAI/bge-reranker-large",
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "default": "BAAI/bge-reranker-v2-m3",
}

MATCHER_MODE_REGISTRY = {
    "zero-shot": "EmbeddingMatcher",
    "head-only": "_EntityMatcher",
    "full": "_EntityMatcher",
    "hybrid": "HybridMatcher",
    "bert": "_EntityMatcher",
    "auto": "SmartSelection",
}

NOVEL_DETECTION_CONFIG = {
    "default_strategies": ["confidence", "knn_distance", "clustering"],
    "confidence_threshold": 0.7,
    "distance_threshold": 0.3,
    "uncertainty_top_k": 5,
    "knn_k": 5,
    "knn_distance_threshold": 0.55,
    "min_cluster_size": 5,
    "ann_config": {
        "backend": "hnswlib",
        "hnswlib": {
            "ef_construction": 200,
            "M": 16,
        },
        "faiss": {
            "index_type": "IndexFlatIP",
        },
    },
    "llm_config": {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4",
        "temperature": 0.3,
        "max_tokens": 4096,
    },
    "storage_config": {
        "output_dir": "./proposals",
        "format": "yaml",
    },
    "combine_method": "weighted",
}

LLM_PROVIDERS = {
    "openrouter": {
        "api_key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "models": [
            "anthropic/claude-sonnet-4",
            "openai/gpt-4o",
            "google/gemini-pro-1.5",
        ],
    },
    "anthropic": {
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "https://api.anthropic.com",
        "models": [
            "claude-sonnet-4",
            "claude-opus-4",
            "claude-haiku-4-20251001",
        ],
    },
    "openai": {
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "models": [
            "gpt-4o",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ],
    },
}

_TRUST_REMOTE_CODE_MODELS_ENV = "NOVEL_ENTITY_MATCHER_TRUST_REMOTE_CODE_MODELS"


def resolve_model_alias(model_name: str) -> str:
    result = MODEL_REGISTRY.get(model_name, model_name)
    assert isinstance(result, str)
    return result


def get_trusted_remote_code_models() -> set[str]:
    """Get model allowlist for `trust_remote_code=True` from env."""
    raw = os.getenv(_TRUST_REMOTE_CODE_MODELS_ENV, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def get_model_spec(model_name: str) -> dict[str, Any] | None:
    if model_name == "default":
        model_name = RETRIEVAL_DEFAULT_MODEL

    if model_name in MODEL_SPECS:
        return dict(MODEL_SPECS[model_name])

    resolved_name = resolve_model_alias(model_name)
    for alias, spec in MODEL_SPECS.items():
        if spec["name"] == resolved_name:
            model_spec = dict(spec)
            model_spec["alias"] = alias
            return model_spec
    return None


def is_static_embedding_model(model_name: str) -> bool:
    spec = get_model_spec(model_name)
    if spec is not None:
        return bool(spec["backend"] == "static")
    resolved_name = resolve_model_alias(model_name)
    return resolved_name.startswith(("RikkaBotan/", "minishlab/")) or (
        "static-" in resolved_name
    )


def supports_training_model(model_name: str) -> bool:
    spec = get_model_spec(model_name)
    if spec is not None:
        return bool(spec["supports_training"]) and spec.get("backend") != "bert"
    return not is_static_embedding_model(model_name)


def resolve_training_model_alias(model_name: str) -> str:
    if model_name == "default":
        return resolve_model_alias(TRAINING_DEFAULT_MODEL)
    if supports_training_model(model_name):
        return resolve_model_alias(model_name)
    return resolve_model_alias(TRAINING_DEFAULT_MODEL)


def is_bert_model(model_name: str) -> bool:
    spec = get_model_spec(model_name)
    if spec is not None:
        return spec.get("model_type") == "bert"
    return False


def resolve_bert_model_alias(model_name: str) -> str:
    if model_name == "default":
        return resolve_model_alias(BERT_DEFAULT_MODEL)
    if is_bert_model(model_name):
        return resolve_model_alias(model_name)
    return resolve_model_alias(BERT_DEFAULT_MODEL)


def get_embedding_model_aliases() -> list[str]:
    return [
        alias for alias, spec in MODEL_SPECS.items() if spec.get("backend") != "bert"
    ]


def get_training_model_aliases() -> list[str]:
    return [
        alias
        for alias, spec in MODEL_SPECS.items()
        if spec["supports_training"] and spec.get("backend") != "bert"
    ]


def get_bert_model_aliases() -> list[str]:
    return [
        alias for alias, spec in MODEL_SPECS.items() if spec.get("model_type") == "bert"
    ]


def resolve_reranker_alias(model_name: str) -> str:
    return RERANKER_REGISTRY.get(model_name, model_name)


def resolve_matcher_mode(mode: str) -> str:
    return MATCHER_MODE_REGISTRY.get(mode, mode)


def recommend_model(use_case: str = "general", language: str = "en") -> str:
    recommendations = {
        ("general", "en"): "potion-32m",
        ("fast", "en"): "potion-32m",
        ("accurate", "en"): "potion-32m",
        ("general", "multilingual"): "mrl-multi",
        ("fast", "multilingual"): "mrl-multi",
        ("accurate", "multilingual"): "mrl-multi",
        ("dynamic", "en"): "bge-base",
        ("dynamic", "multilingual"): "bge-m3",
    }
    return recommendations.get((use_case, language), RETRIEVAL_DEFAULT_MODEL)


__all__ = [
    "BERT_DEFAULT_MODEL",
    "DYNAMIC_MODEL_REGISTRY",
    "LLM_PROVIDERS",
    "MATCHER_MODE_REGISTRY",
    "MODEL_REGISTRY",
    "MODEL_SPECS",
    "NOVEL_DETECTION_CONFIG",
    "RERANKER_REGISTRY",
    "RETRIEVAL_DEFAULT_MODEL",
    "STATIC_MODEL_REGISTRY",
    "TRAINING_DEFAULT_MODEL",
    "Config",
    "get_bert_model_aliases",
    "get_embedding_model_aliases",
    "get_model_spec",
    "get_training_model_aliases",
    "get_trusted_remote_code_models",
    "is_bert_model",
    "is_static_embedding_model",
    "recommend_model",
    "resolve_bert_model_alias",
    "resolve_matcher_mode",
    "resolve_model_alias",
    "resolve_reranker_alias",
    "resolve_training_model_alias",
    "supports_training_model",
]


class Config:
    """Configuration loader with optional custom override merging."""

    def __init__(self, custom_path: PathLike | None = None):
        self._config: dict[str, Any] = self._load_default_config()
        if custom_path:
            self._merge_custom_config(custom_path)

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> "Config":
        obj = cls.__new__(cls)
        obj._config = dict(data)
        return obj

    def _load_default_config(self) -> dict[str, Any]:
        for path in self._default_config_candidates():
            if path is None or not path.exists():
                continue
            loaded = self._safe_load_file(path)
            if loaded is not None:
                return loaded
        return {}

    def _default_config_candidates(self) -> list[Path | None]:
        return [
            self._find_repo_root_config(),
            self._cwd_config(),
        ]

    def _find_repo_root_config(self) -> Path | None:
        current = Path(__file__).resolve().parent
        repo_markers = (".git", "setup.py", "pyproject.toml")

        for directory in (current, *current.parents):
            if any((directory / marker).exists() for marker in repo_markers):
                config_path = directory / "config.yaml"
                if config_path.exists():
                    return config_path
        return None

    def _cwd_config(self) -> Path | None:
        cwd_path = Path.cwd() / "config.yaml"
        return cwd_path if cwd_path.exists() else None

    def _safe_load_file(self, path: Any) -> dict[str, Any] | None:
        try:
            return self._load_file(path)
        except (OSError, yaml.YAMLError, json.JSONDecodeError, ValueError):
            return None

    def _load_file(self, path: Any) -> dict[str, Any]:
        if isinstance(path, (str, Path)):
            file_path = Path(path)
            text = file_path.read_text(encoding="utf-8")
            suffix = file_path.suffix.lower()
        else:
            text = path.read_text(encoding="utf-8")
            suffix = Path(getattr(path, "name", "")).suffix.lower()

        if not text.strip():
            return {}

        if suffix == ".json":
            loaded = json.loads(text)
        else:
            loaded = yaml.safe_load(text)

        if loaded is None:
            return {}
        if not isinstance(loaded, dict):
            raise ValueError("Config file must contain a mapping/object at top level")
        return loaded

    def _merge_custom_config(self, path: PathLike):
        custom_config = self._load_file(path)
        self._deep_update(self._config, custom_config)

    def _deep_update(self, base: dict[str, Any], update: Mapping[str, Any]):
        for key, value in update.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, Mapping)
            ):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return self._from_mapping(value)
            return value
        raise AttributeError(f"'{self.__class__.__name__}' has no attribute '{name}'")

    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split(".")
        value: Any = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def to_dict(self) -> dict[str, Any]:
        return dict(self._config)

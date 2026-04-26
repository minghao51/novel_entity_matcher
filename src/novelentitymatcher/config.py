import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Union

import yaml

from .config_registry import (
    BERT_DEFAULT_MODEL,
    DYNAMIC_MODEL_REGISTRY,
    LLM_PROVIDERS,
    MATCHER_MODE_REGISTRY,
    MODEL_REGISTRY,
    MODEL_SPECS,
    NOVEL_DETECTION_CONFIG,
    RERANKER_REGISTRY,
    RETRIEVAL_DEFAULT_MODEL,
    STATIC_MODEL_REGISTRY,
    TRAINING_DEFAULT_MODEL,
    get_bert_model_aliases,
    get_embedding_model_aliases,
    get_model_spec,
    get_training_model_aliases,
    is_bert_model,
    is_static_embedding_model,
    recommend_model,
    resolve_bert_model_alias,
    resolve_matcher_mode,
    resolve_model_alias,
    resolve_reranker_alias,
    resolve_training_model_alias,
    supports_training_model,
)

__all__ = [
    # Re-exported from config_registry
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
    # Local exports
    "Config",
    "get_bert_model_aliases",
    "get_embedding_model_aliases",
    "get_model_spec",
    "get_training_model_aliases",
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

PathLike = Union[str, Path]


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
            self._package_default_config(),
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

    def _package_default_config(self) -> Any | None:
        try:
            from importlib import resources

            return resources.files("novelentitymatcher").joinpath(
                "data/default_config.json"
            )
        except (ImportError, FileNotFoundError, ModuleNotFoundError, TypeError):
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

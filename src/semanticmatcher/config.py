import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import yaml

PathLike = Union[str, Path]


# Model registries for easy model selection.
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
RETRIEVAL_DEFAULT_MODEL = "potion-8m"
TRAINING_DEFAULT_MODEL = "mpnet"
MODEL_REGISTRY["default"] = MODEL_SPECS[RETRIEVAL_DEFAULT_MODEL]["name"]

RERANKER_REGISTRY = {
    "bge-m3": "BAAI/bge-reranker-v2-m3",
    "bge-large": "BAAI/bge-reranker-large",
    "ms-marco": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "default": "BAAI/bge-reranker-v2-m3",
}

# Matcher mode registry for unified Matcher class
MATCHER_MODE_REGISTRY = {
    "zero-shot": "EmbeddingMatcher",
    "head-only": "EntityMatcher",
    "full": "EntityMatcher",
    "hybrid": "HybridMatcher",
    "auto": "SmartSelection",
}


def resolve_model_alias(model_name: str) -> str:
    """Resolve model alias to full model name."""
    return MODEL_REGISTRY.get(model_name, model_name)


def get_model_spec(model_name: str) -> Optional[Dict[str, Any]]:
    """Return registry metadata for an alias or resolved model name."""
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
    """Return True when the model should use the static embedding backend."""
    spec = get_model_spec(model_name)
    if spec is not None:
        return spec["backend"] == "static"
    resolved_name = resolve_model_alias(model_name)
    return resolved_name.startswith(("RikkaBotan/", "minishlab/")) or (
        "static-" in resolved_name
    )


def supports_training_model(model_name: str) -> bool:
    """Return whether the model can be used as a SetFit/SentenceTransformer backbone."""
    spec = get_model_spec(model_name)
    if spec is not None:
        return bool(spec["supports_training"])
    return not is_static_embedding_model(model_name)


def resolve_training_model_alias(model_name: str) -> str:
    """
    Resolve the effective training backbone.

    The public default stays retrieval-first, but trained modes must use a
    SentenceTransformer-compatible backbone.
    """
    if model_name == "default":
        return resolve_model_alias(TRAINING_DEFAULT_MODEL)

    if supports_training_model(model_name):
        return resolve_model_alias(model_name)

    return resolve_model_alias(TRAINING_DEFAULT_MODEL)


def get_embedding_model_aliases() -> list[str]:
    """Return embedding aliases in a stable order for benchmarks."""
    return list(MODEL_SPECS.keys())


def get_training_model_aliases() -> list[str]:
    """Return aliases that support SetFit-based training."""
    return [alias for alias, spec in MODEL_SPECS.items() if spec["supports_training"]]


def resolve_reranker_alias(model_name: str) -> str:
    """Resolve reranker alias to full model name."""
    return RERANKER_REGISTRY.get(model_name, model_name)


def resolve_matcher_mode(mode: str) -> str:
    """
    Resolve matcher mode name to implementation class name.

    Args:
        mode: Mode name (e.g., 'zero-shot', 'head-only', 'full', 'auto')

    Returns:
        Implementation class name or the original mode if not found in registry.

    Example:
        resolve_matcher_mode('zero-shot')  # 'EmbeddingMatcher'
        resolve_matcher_mode('full')       # 'EntityMatcher'
        resolve_matcher_mode('auto')       # 'SmartSelection'
    """
    return MATCHER_MODE_REGISTRY.get(mode, mode)


def recommend_model(use_case: str = "general", language: str = "en") -> str:
    """
    Recommend appropriate model based on use case and language.

    Static embeddings are the default for retrieval-oriented use cases due to
    their speed improvement with modest quality tradeoffs.

    Args:
        use_case: Type of matching - "general", "fast", "multilingual", "accurate"
        language: Primary language - "en", "zh", "multilingual"

    Returns:
        Model alias or full model name
    """
    recommendations = {
        # Static embeddings as default for English
        ("general", "en"): "potion-8m",
        ("fast", "en"): "potion-8m",
        ("accurate", "en"): "potion-8m",
        # Static embeddings for multilingual (MRL model)
        ("general", "multilingual"): "mrl-multi",
        ("fast", "multilingual"): "mrl-multi",
        ("accurate", "multilingual"): "mrl-multi",
        # Dynamic models for trained or context-heavy workflows.
        ("dynamic", "en"): "bge-base",
        ("dynamic", "multilingual"): "bge-m3",
    }
    return recommendations.get((use_case, language), RETRIEVAL_DEFAULT_MODEL)


class Config:
    """Configuration loader with optional custom override merging."""

    def __init__(self, custom_path: Optional[PathLike] = None):
        self._config: Dict[str, Any] = self._load_default_config()
        if custom_path:
            self._merge_custom_config(custom_path)

    @classmethod
    def _from_mapping(cls, data: Mapping[str, Any]) -> "Config":
        obj = cls.__new__(cls)
        obj._config = dict(data)
        return obj

    def _load_default_config(self) -> Dict[str, Any]:
        for path in self._default_config_candidates():
            if path is None or not path.exists():
                continue
            loaded = self._safe_load_file(path)
            if loaded is not None:
                return loaded
        return {}

    def _default_config_candidates(self) -> list[Optional[Path]]:
        return [
            self._find_repo_root_config(),
            self._package_default_config(),
            self._cwd_config(),
        ]

    def _find_repo_root_config(self) -> Optional[Path]:
        current = Path(__file__).resolve().parent
        repo_markers = (".git", "setup.py", "pyproject.toml")

        for directory in (current, *current.parents):
            if any((directory / marker).exists() for marker in repo_markers):
                config_path = directory / "config.yaml"
                if config_path.exists():
                    return config_path
        return None

    def _package_default_config(self) -> Optional[Any]:
        try:
            from importlib import resources

            return resources.files("semanticmatcher").joinpath(
                "data/default_config.json"
            )
        except (ImportError, FileNotFoundError, ModuleNotFoundError, TypeError):
            return None

    def _cwd_config(self) -> Optional[Path]:
        cwd_path = Path.cwd() / "config.yaml"
        return cwd_path if cwd_path.exists() else None

    def _safe_load_file(self, path: Any) -> Optional[Dict[str, Any]]:
        try:
            return self._load_file(path)
        except (OSError, yaml.YAMLError, json.JSONDecodeError, ValueError):
            return None

    def _load_file(self, path: Any) -> Dict[str, Any]:
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

    def _deep_update(self, base: Dict[str, Any], update: Mapping[str, Any]):
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

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._config)

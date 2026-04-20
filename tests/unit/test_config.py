from novelentitymatcher.config import (
    BERT_DEFAULT_MODEL,
    Config,
    RETRIEVAL_DEFAULT_MODEL,
    TRAINING_DEFAULT_MODEL,
    get_bert_model_aliases,
    is_bert_model,
    resolve_bert_model_alias,
    get_embedding_model_aliases,
    get_training_model_aliases,
    is_static_embedding_model,
    resolve_matcher_mode,
    resolve_training_model_alias,
    supports_training_model,
)


def test_config_loads_default_and_nested_access(tmp_path, monkeypatch):
    default_path = tmp_path / "default.yaml"
    default_path.write_text(
        "default_model: base-model\ntraining:\n  num_epochs: 4\n  batch_size: 16\n"
    )

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [default_path],
    )

    cfg = Config()

    assert cfg.default_model == "base-model"
    assert cfg.training.num_epochs == 4
    assert cfg.get("training.batch_size") == 16
    assert cfg.get("missing.key", "fallback") == "fallback"


def test_config_merges_custom_overrides(tmp_path, monkeypatch):
    default_path = tmp_path / "default.yaml"
    custom_path = tmp_path / "custom.yaml"
    default_path.write_text("training:\n  num_epochs: 4\n  batch_size: 16\n")
    custom_path.write_text("training:\n  batch_size: 8\nembedding:\n  threshold: 0.9\n")

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [default_path],
    )

    cfg = Config(custom_path=custom_path)

    assert cfg.training.num_epochs == 4
    assert cfg.training.batch_size == 8
    assert cfg.embedding.threshold == 0.9


def test_config_instances_do_not_share_state(tmp_path, monkeypatch):
    first_default = tmp_path / "one.yaml"
    second_default = tmp_path / "two.yaml"
    first_default.write_text("default_model: first\n")
    second_default.write_text("default_model: second\n")

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [first_default],
    )
    first = Config()

    monkeypatch.setattr(
        Config,
        "_default_config_candidates",
        lambda self: [second_default],
    )
    second = Config()

    assert first.default_model == "first"
    assert second.default_model == "second"


def test_config_can_load_packaged_default_when_local_sources_absent(monkeypatch):
    monkeypatch.setattr(Config, "_find_repo_root_config", lambda self: None)
    monkeypatch.setattr(Config, "_cwd_config", lambda self: None)

    package_resource = Config()._package_default_config()
    assert package_resource is not None

    cfg = Config()

    assert isinstance(cfg.to_dict(), dict)


def test_resolve_matcher_mode_supported_values():
    assert resolve_matcher_mode("zero-shot") == "EmbeddingMatcher"
    assert resolve_matcher_mode("full") == "_EntityMatcher"
    assert resolve_matcher_mode("hybrid") == "HybridMatcher"
    assert resolve_matcher_mode("auto") == "SmartSelection"


def test_resolve_matcher_mode_unsupported_value_passthrough():
    assert resolve_matcher_mode("custom-mode") == "custom-mode"


def test_training_model_resolution_uses_training_default_for_public_default():
    assert resolve_training_model_alias("default").endswith("all-mpnet-base-v2")


def test_training_model_resolution_falls_back_for_bert_models():
    assert resolve_training_model_alias("distilbert").endswith("all-mpnet-base-v2")


def test_bert_model_aliases_are_discoverable():
    aliases = get_bert_model_aliases()
    assert BERT_DEFAULT_MODEL in aliases
    assert "mpnet" not in aliases


def test_bert_model_detection_accepts_alias_and_resolved_name():
    assert is_bert_model("distilbert") is True
    assert is_bert_model("distilbert-base-uncased") is True
    assert is_bert_model("mpnet") is False


def test_bert_model_resolution_uses_bert_default_for_public_default():
    assert resolve_bert_model_alias("default") == "distilbert-base-uncased"


def test_bert_model_resolution_preserves_bert_models():
    assert resolve_bert_model_alias("distilbert") == "distilbert-base-uncased"
    assert resolve_bert_model_alias("roberta-base") == "roberta-base"


def test_bert_model_resolution_falls_back_for_non_bert_models():
    assert resolve_bert_model_alias("mpnet") == "distilbert-base-uncased"
    assert resolve_bert_model_alias("potion-8m") == "distilbert-base-uncased"


def test_static_models_are_marked_retrieval_only():
    assert is_static_embedding_model(RETRIEVAL_DEFAULT_MODEL) is True
    assert supports_training_model(RETRIEVAL_DEFAULT_MODEL) is False


def test_dynamic_models_are_training_compatible():
    assert supports_training_model(TRAINING_DEFAULT_MODEL) is True
    assert supports_training_model("distilbert") is False


def test_training_model_aliases_exclude_static_models():
    aliases = get_training_model_aliases()
    assert "mpnet" in aliases
    assert "potion-8m" not in aliases
    assert "distilbert" not in aliases


def test_embedding_model_aliases_include_static_and_dynamic_entries():
    aliases = get_embedding_model_aliases()
    assert "potion-8m" in aliases
    assert "mpnet" in aliases
    assert "distilbert" not in aliases

import logging
from pathlib import Path

from novelentitymatcher.benchmarks.loader import DatasetLoader
from novelentitymatcher.benchmarks.registry import DatasetConfig


async def test_aload_dataset_warns_on_insecure_http_download_url(
    monkeypatch, caplog, tmp_path: Path
):
    config = DatasetConfig(
        name="insecure_dataset",
        hf_path="unused",
        task_type="entity_matching",
        download_url="http://example.com/deepmatcher",
        cache_dir=tmp_path,
    )
    loader = DatasetLoader(cache_dir=tmp_path)

    monkeypatch.setattr(
        "novelentitymatcher.benchmarks.loader.get_dataset_config",
        lambda _name: config,
    )
    monkeypatch.setattr(loader, "_is_cache_valid", lambda _config: True)
    monkeypatch.setattr(
        loader,
        "_load_from_cache",
        lambda _config: {"name": "insecure_dataset", "test": []},
    )

    with caplog.at_level(logging.WARNING):
        result = await loader.aload_dataset("insecure_dataset")

    assert result["name"] == "insecure_dataset"
    assert "insecure HTTP download URL" in caplog.text
    assert "Prefer an HTTPS mirror" in caplog.text


async def test_aload_dataset_does_not_warn_for_hf_native_dataset(
    monkeypatch, caplog, tmp_path: Path
):
    config = DatasetConfig(
        name="hf_dataset",
        hf_path="ag_news",
        task_type="classification",
        download_url=None,
        cache_dir=tmp_path,
    )
    loader = DatasetLoader(cache_dir=tmp_path)

    monkeypatch.setattr(
        "novelentitymatcher.benchmarks.loader.get_dataset_config",
        lambda _name: config,
    )
    monkeypatch.setattr(loader, "_is_cache_valid", lambda _config: True)
    monkeypatch.setattr(
        loader,
        "_load_from_cache",
        lambda _config: {"name": "hf_dataset", "test": []},
    )

    with caplog.at_level(logging.WARNING):
        result = await loader.aload_dataset("hf_dataset")

    assert result["name"] == "hf_dataset"
    assert "insecure HTTP download URL" not in caplog.text

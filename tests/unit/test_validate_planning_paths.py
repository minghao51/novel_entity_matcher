import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path("scripts/validate_planning_paths.py")
    spec = importlib.util.spec_from_file_location(
        "validate_planning_paths", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_extract_paths_filters_path_like_tokens():
    mod = _load_module()
    text = (
        "Use `src/novelentitymatcher/config.py` and `tests/unit/test_config.py`.\n"
        "Ignore `novelentitymatcher-ingest` and `PipelineConfig`."
    )

    paths = mod.extract_paths(text)

    assert "src/novelentitymatcher/config.py" in paths
    assert "tests/unit/test_config.py" in paths
    assert "novelentitymatcher-ingest" not in paths
    assert "PipelineConfig" not in paths


def test_validate_paths_reports_missing(tmp_path):
    mod = _load_module()
    doc = tmp_path / "doc.md"
    doc.write_text(
        "Exists: `src/app.py`\nMissing: `src/missing.py`\n",
        encoding="utf-8",
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")

    missing = mod.validate_paths(tmp_path, doc)

    assert missing == ["src/missing.py"]

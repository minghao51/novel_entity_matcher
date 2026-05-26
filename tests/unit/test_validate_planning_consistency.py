import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path("scripts/validate_planning_consistency.py")
    spec = importlib.util.spec_from_file_location(
        "validate_planning_consistency", module_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_consistency_happy_path(tmp_path):
    mod = _load_module()
    testing = tmp_path / "TESTING.md"
    ci = tmp_path / "ci.yml"
    testing.write_text(
        "Coverage is enforced via `--cov-fail-under=54`.\n",
        encoding="utf-8",
    )
    ci.write_text("run: uv run pytest --cov-fail-under=54\n", encoding="utf-8")

    errors = mod.validate_consistency(testing, ci)

    assert errors == []


def test_validate_consistency_flags_doc_drift(tmp_path):
    mod = _load_module()
    testing = tmp_path / "TESTING.md"
    ci = tmp_path / "ci.yml"
    testing.write_text(
        "Coverage is not enforced as a CI gate.\n",
        encoding="utf-8",
    )
    ci.write_text("run: uv run pytest --cov-fail-under=54\n", encoding="utf-8")

    errors = mod.validate_consistency(testing, ci)

    assert any("not enforced" in err for err in errors)

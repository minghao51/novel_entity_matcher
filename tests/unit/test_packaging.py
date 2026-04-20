from pathlib import Path


def test_optional_dependency_section_declares_expected_extras():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert "[project.optional-dependencies]" in content
    assert "\njupyter = [\n" in content
    assert "\ndev = [\n" in content
    assert "\nall = [\n" in content


def test_optional_dependency_section_has_no_self_references():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    content = pyproject.read_text(encoding="utf-8")

    assert "novel-entity-matcher[" not in content

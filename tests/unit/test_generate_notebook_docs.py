import importlib.util
from pathlib import Path


def _load_module():
    module_path = Path("scripts/generate_notebook_docs.py")
    spec = importlib.util.spec_from_file_location("generate_notebook_docs", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_qmd_reads_frontmatter(tmp_path):
    mod = _load_module()
    qmd = tmp_path / "01_sample.qmd"
    qmd.write_text(
        "---\n"
        "title: Notebook A\n"
        "description: Demo notebook\n"
        "---\n\n"
        "Body\n",
        encoding="utf-8",
    )

    meta = mod.parse_qmd(str(qmd))

    assert meta == {"title": "Notebook A", "description": "Demo notebook"}


def test_generate_stub_uses_relative_path_by_default(monkeypatch):
    mod = _load_module()
    monkeypatch.delenv("NOTEBOOK_DOCS_SITE_BASE_PATH", raising=False)

    out = mod.generate_stub({"title": "A", "description": "B"}, "01_demo")

    assert 'href="html/01_demo.html"' in out
    assert 'src="html/01_demo.html"' in out


def test_generate_stub_uses_configured_base_path(monkeypatch):
    mod = _load_module()
    monkeypatch.setenv("NOTEBOOK_DOCS_SITE_BASE_PATH", "novel_entity_matcher/")

    out = mod.generate_stub({"title": "A", "description": "B"}, "01_demo")

    assert 'href="/novel_entity_matcher/notebooks/html/01_demo.html"' in out
    assert 'src="/novel_entity_matcher/notebooks/html/01_demo.html"' in out


def test_generate_index_contains_artifact_policy():
    mod = _load_module()
    out = mod.generate_index([("A", "B", "01_demo")])

    assert "## Generated Artifacts Policy" in out
    assert "`docs/notebooks/html/*.html`" in out
    assert "[A](01_demo.md)" in out

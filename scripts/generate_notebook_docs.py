"""Generate MkDocs stub pages from .qmd notebook frontmatter.

Reads notebooks/*.qmd frontmatter, produces:
  - docs/notebooks/<slug>.md       — iframe embed + run-locally snippet
  - docs/notebooks/index.md        — notebook listing
"""

import os
import re

import yaml

NOTEBOOKS_DIR = "notebooks"
DOCS_OUT = "docs/notebooks"
QMD_PATTERN = re.compile(r"^\d+_.+\.qmd$")

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)


def parse_qmd(path: str) -> dict | None:
    with open(path) as f:
        content = f.read()
    m = FRONTMATTER_RE.match(content)
    if not m:
        return None
    meta = yaml.safe_load(m.group(1))
    return meta


def slug_from_filename(filename: str) -> str:
    return os.path.splitext(filename)[0]


def generate_stub(meta: dict, slug: str) -> str:
    title = meta.get("title", slug.replace("_", " ").title())
    description = meta.get("description", "")

    return f"""# {title}

{description}

<div style="margin: 0 -0.8rem">
  <iframe src="html/{slug}.html"
    style="width:100%; height:600px; border:1px solid var(--md-default-fg-color--lightest); border-radius:4px;"
    loading="lazy"></iframe>
</div>

## Run Locally

```bash
uv sync --extra docs
uv run marimo edit notebooks/{slug}.py
```
"""


def generate_index(entries: list[tuple[str, str, str]]) -> str:
    lines = [
        "# Notebooks",
        "",
        "Interactive marimo notebooks for entity matching, novelty detection, and benchmarking.",
        "",
        "**Source of truth:** [`notebooks/*.py`](https://github.com/minghao51/novel_entity_matcher/tree/main/notebooks)"
        " — Quarto `.qmd` versions are auto-rendered from the marimo sources in CI.",
        "",
        "| Notebook | Description |",
        "|----------|-------------|",
    ]
    for title, desc, slug in entries:
        lines.append(f"| [{title}]({slug}.md) | {desc} |")

    lines += [
        "",
        "## Run locally",
        "",
        "[![marimo](https://img.shields.io/badge/Run%20locally-marimo-2489F4?logo=python)]"
        "(https://github.com/minghao51/novel_entity_matcher)",
        "",
        "```bash",
        "uv run marimo edit notebooks/<name>.py",
        "```",
        "",
        "To rebuild docs pages locally after notebook changes:",
        "",
        "```bash",
        "uv run quarto render notebooks/",
        "uv run python scripts/generate_notebook_docs.py",
        "uv run mkdocs serve",
        "```",
    ]
    return "\n".join(lines)


def main():
    os.makedirs(DOCS_OUT, exist_ok=True)
    os.makedirs(f"{DOCS_OUT}/html", exist_ok=True)

    qmd_files = sorted(f for f in os.listdir(NOTEBOOKS_DIR) if QMD_PATTERN.match(f))

    entries: list[tuple[str, str, str]] = []

    for filename in qmd_files:
        path = os.path.join(NOTEBOOKS_DIR, filename)
        meta = parse_qmd(path)
        if not meta:
            continue

        slug = slug_from_filename(filename)
        title = meta.get("title", slug.replace("_", " ").title())
        description = meta.get("description", "")

        stub = generate_stub(meta, slug)
        stub_path = os.path.join(DOCS_OUT, f"{slug}.md")
        with open(stub_path, "w") as f:
            f.write(stub)
        print(f"  {stub_path}")

        entries.append((title, description, slug))

    index = generate_index(entries)
    index_path = os.path.join(DOCS_OUT, "index.md")
    with open(index_path, "w") as f:
        f.write(index)
    print(f"  {index_path}")


if __name__ == "__main__":
    main()

"""Validate repository paths referenced in planning docs."""

from __future__ import annotations

import re
import sys
from pathlib import Path

DEFAULT_DOC = Path(".planning/codebase/ARCHITECTURE.md")
PATH_TOKEN_RE = re.compile(r"`([^`]+)`")
PATH_PREFIXES = (
    "src/",
    "tests/",
    "scripts/",
    "docs/",
    ".github/",
    "notebooks/",
    "data/",
)


def _looks_like_repo_path(token: str) -> bool:
    cleaned = token.strip()
    if cleaned.endswith((",", ".", ";", ":")):
        cleaned = cleaned[:-1]
    return cleaned.startswith(PATH_PREFIXES)


def _normalize_repo_path(token: str) -> str:
    normalized = token.rstrip(".,;:")
    if ".py:" in normalized:
        normalized = normalized.split(".py:", maxsplit=1)[0] + ".py"
    return normalized


def extract_paths(doc_text: str) -> set[str]:
    paths: set[str] = set()
    for token in PATH_TOKEN_RE.findall(doc_text):
        if _looks_like_repo_path(token):
            paths.add(_normalize_repo_path(token))
    return paths


def validate_paths(repo_root: Path, doc_path: Path) -> list[str]:
    content = doc_path.read_text(encoding="utf-8")
    missing: list[str] = []
    for ref in sorted(extract_paths(content)):
        if not (repo_root / ref).exists():
            missing.append(ref)
    return missing


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    doc_path = Path(args[0]) if args else DEFAULT_DOC
    repo_root = Path.cwd()

    if not doc_path.exists():
        print(f"Planning doc not found: {doc_path}", file=sys.stderr)
        return 2

    missing = validate_paths(repo_root, doc_path)
    if missing:
        print(f"Missing paths referenced in {doc_path}:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1

    print(f"All referenced paths exist in {doc_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Validate high-signal consistency between planning docs and CI behavior."""

from __future__ import annotations

import sys
from pathlib import Path

TESTING_DOC = Path(".planning/codebase/TESTING.md")
CI_WORKFLOW = Path(".github/workflows/ci.yml")
REQUIRED_COVERAGE_FLAG = "--cov-fail-under=54"


def validate_consistency(testing_doc: Path, ci_workflow: Path) -> list[str]:
    errors: list[str] = []
    doc_text = testing_doc.read_text(encoding="utf-8")
    ci_text = ci_workflow.read_text(encoding="utf-8")

    if REQUIRED_COVERAGE_FLAG not in ci_text:
        errors.append(
            f"CI workflow is missing required coverage flag: {REQUIRED_COVERAGE_FLAG}"
        )

    if "Coverage is not enforced as a CI gate" in doc_text:
        errors.append(
            "Testing planning doc claims coverage is not enforced, but CI enforces it."
        )

    if REQUIRED_COVERAGE_FLAG not in doc_text:
        errors.append(
            "Testing planning doc does not mention the enforced coverage threshold."
        )

    return errors


def main(argv: list[str] | None = None) -> int:
    del argv
    if not TESTING_DOC.exists():
        print(f"Missing planning doc: {TESTING_DOC}", file=sys.stderr)
        return 2
    if not CI_WORKFLOW.exists():
        print(f"Missing CI workflow: {CI_WORKFLOW}", file=sys.stderr)
        return 2

    errors = validate_consistency(TESTING_DOC, CI_WORKFLOW)
    if errors:
        print("Planning/CI consistency check failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print("Planning/CI consistency check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

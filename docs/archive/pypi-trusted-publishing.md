# PyPI Trusted Publishing Setup

Archived note: this is a narrow release-operations reference rather than part of the main user or contributor docs path.

Use this for GitHub Actions OIDC-based publishing (no long-lived PyPI API token).

## One-Time Setup (PyPI)

1. Create or open the `novel-entity-matcher` project on PyPI.
2. Go to Publishing settings and add a Trusted Publisher.
3. Configure:
   - Owner: your GitHub org/user
   - Repository: this repo
   - Workflow file: `publish.yml`
   - Environment (optional but recommended): `pypi`

## Release Trigger

- Push a Git tag like `v0.1.0` to trigger `/Users/minghao/Desktop/personal/novel_entity_matcher/.github/workflows/publish.yml`.

## Notes

- The workflow builds both sdist and wheel and runs `twine check` before publish.
- If PyPI trusted publishing is not configured yet, the publish step will fail until the publisher mapping is added.

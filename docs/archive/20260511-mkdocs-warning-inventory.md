# MkDocs Warning Inventory (2026-05-11)

This inventory was captured from `uv run mkdocs build` and used to close docs warning debt.

## Categories addressed

- Broken asset links in top-level docs (`docs/classifier-routes-comparison.md`)
- Broken links to non-doc paths (`docs/examples.md`, `docs/configuration.md`)
- Broken benchmark links (`docs/models.md`, `docs/static-embeddings.md`)
- Malformed relative links (`docs/quickstart.md`, `docs/troubleshooting.md`, methodology index pages)
- Archive links pointing to wrong paths (`docs/archive/2026-04-25-novelty-detection-benchmark-root.md`)
- API docs warnings from missing type annotations in exposed signatures (core + novelty strategy modules)

## Result

- Current MkDocs warning count: `0`
- Remaining non-failing docs output:
  - `INFO` entries for docs files not included in nav (intentional archive/methodology policy)

## Policy decision

Archive/methodology content remains outside primary nav unless actively maintained as user-facing docs. Active docs should not depend on missing archive-only targets.

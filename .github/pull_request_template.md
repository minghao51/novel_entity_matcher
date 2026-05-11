## Summary

- What changed and why.

## Validation

- [ ] `uv run mkdocs build` (or `uv run mkdocs build --strict`)
- [ ] Relevant unit/integration tests passed

## Docs/Notebook checklist

- [ ] If any `.qmd` changed: ran `uv run quarto render notebooks/`
- [ ] If any notebook metadata changed: ran `uv run python scripts/generate_notebook_docs.py`
- [ ] Verified there is no docs drift (`git diff -- docs/notebooks/` is clean)
- [ ] Generated notebook artifacts (`docs/notebooks/html/`, `notebooks/_freeze/`) are included only when intentionally refreshed

## Scope control

- [ ] PR is scoped to one concern (no unrelated churn mixed in)

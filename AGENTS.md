## Workflow
- **Analyze first.** Read files before proposing. Never hallucinate.
- **Check skills.** Before any task, check + follow matching skill.
- **Approve changes.** Present plan before modifying code.
- **Minimal scope.** Change as little as possible. No new abstractions.
- **Verify.** Run `uv run ruff check` (lint) and `uv run mypy src` (type-check) after changes. Ask user for command if unsure.
- **No commits.** Never commit unless explicitly asked.

## Output Style
- sharp and concise, remove filler words, repetition and weak phrasing.
- be thorough in implementation, never leave TODOs or placeholders, complete all code changes fully
- No speculation about unread code.

## File Operations
- **Read before edit.** Always read a file before editing it.
- **Edit over Write.** Prefer Edit tool for surgical changes.
- **Edit existing over new.** Prefer editing existing files over creating new ones.

## Technical Stack
- **Python:** Package manager `uv`. Always `uv run <command>`. Never `python`. Sync via `uv sync`.
- **Frontend:** Run `npm run check` and `npm test` after changes.
- **Docs:** Update `ARCHITECTURE.md` on structure changes.
- **Files:** Markdown names use `YYYYMMDD-filename.md` format.
- **Project context:**
  - Architecture → @.planning/codebase/ARCHITECTURE.md
  - Stack → @.planning/codebase/STACK.md
  - Conventions → @.planning/codebase/CONVENTIONS.md
  - Integrations → @.planning/codebase/INTEGRATIONS.md
  - Testing → @.planning/codebase/TESTING.md
  - Concerns → @.planning/codebase/CONCERNS.md
  - Structure → @.planning/codebase/STRUCTURE.md

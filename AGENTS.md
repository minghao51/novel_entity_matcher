## Workflow
- **Analyze first.** Read files before proposing. Never hallucinate.
- **Approve changes.** Present plan before modifying code.
- **Minimal scope.** Change as little as possible. No new abstractions.

## Output Style
- sharp and concise, remove filler words, repetition and weak phrasing.
- be thorough in implementation, never leave TODOs or placeholders, complete all code changes fully
- No speculation about unread code.

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

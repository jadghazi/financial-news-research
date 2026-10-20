# Independent research projects

This repository is a collection of separate projects, not a shared pipeline.

- Read the root README, then work only in the requested project. Follow its own `AGENTS.md` and README.
- `projects/finin-study/` contains the paused FININ-inspired study and its evidence. Do not resume financial experiments merely to reorganize or document it.
- `projects/llm-news-study/` is reserved and intentionally empty. The user explicitly asked that nothing be added there yet. Do not create placeholders, instructions, dependencies, scraper code, prompts, or experiments without a subsequent request.
- Keep projects independent: no cross-project imports, data/artifact lookups, shared runtime configuration, automatic comparisons, or required environments. Do not inherit the FININ study's cohort, dates, models, or settings into future LLM work.
- Run commands from the relevant project root. Create new environments inside that project. Root-level ignored `.venv/` and `tmp/` are legacy local tooling only.
- Preserve scientific results and source snapshots unchanged. Earlier Git revisions use the old root-level layout; the tag `finin-before-project-split-2026-10-20` marks the state before relocation.
- Default to one main agent. Do not delegate unless explicitly requested by the user.
- Keep secrets, downloaded data, environments, caches, and large generated files out of Git. Project-specific rules belong in each project's own `.gitignore`.

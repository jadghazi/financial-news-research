# Financial news research

One repository for **independent research projects**. Sharing a repository is only for organization: each project owns its code, dependencies, data, configuration, documentation, and results. Projects do not import from one another or require one another to run.

| Project | Status | Start here |
|---|---|---|
| FININ-inspired study | Paused; implementation, completed experiments, negative findings, and meeting material preserved | [Project README](projects/finin-study/README.md), [code guide](projects/finin-study/docs/code_guide.md), [decision record](projects/finin-study/DECISIONS.md) |
| LLM news study | Reserved for future work; nothing added yet | No implementation or experiment has been created |

```text
projects/
  finin-study/       Existing study, with its own instructions and requirements
  llm-news-study/    Reserved empty folder; no files yet
```

Git does not track empty folders. `projects/llm-news-study/` exists locally but will appear in a clone only after work is added. At the user's request, it contains no placeholder, README, code, or configuration.

## Work within a project

For the existing study, change into `projects/finin-study` before using its documented commands. Read its [AGENTS.md](projects/finin-study/AGENTS.md), [runbook](projects/finin-study/docs/cmin_runbook.md), and [sharing instructions](projects/finin-study/docs/laptop_sync.md). Dataset/model acquisition requirements and historical replay limits are documented there. A clone includes selected evidence, not the full local training archive.

Future LLM work will be a standalone project. It must have its own environment and instructions when implemented; it must not use the FININ project's modules, artifacts, datasets, settings, or results as implicit dependencies. No comparison between projects is implemented or assumed.

## Preserved history

The repository was organized into project folders on 20 October 2026. The tag `finin-before-project-split-2026-10-20` preserves the FININ study immediately before relocation. Earlier commits use the original root-level layout. The relocation changes organization, not the saved scientific evidence or model calculations.

The local workspace may retain ignored workstation tooling (`.venv/` and `tmp/`) from the earlier layout. These are not shared project dependencies. New clones should create environments inside the relevant project.

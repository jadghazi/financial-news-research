# Sharing and saved evidence

This guide is for the independent FININ study. For an existing clone after pulling, change into `projects/finin-study/` before using project commands. Relative data, artifact, and documentation paths below refer to that directory. Environments are project-local for new setups; no sibling project is needed.

From an existing clone, run `git pull --ff-only origin main`. For a new clone, run:

```powershell
git clone https://github.com/jadghazi/financial-news-research.git
cd financial-news-research/projects/finin-study
```

Start with the links in the repository README, then [DECISIONS.md](../DECISIONS.md) for what was tried and why, and [the repository guide](repository_guide.md) for the file map. `output/pdf/` contains only the progress report and attention/MLP explainer, with their LaTeX sources and build records. Findings are under `reports/`, and experiment protocols under `docs/`. Earlier output documents and figures remain in Git history at `0be47b9`. Source, tests and configurations are included. The completed embedding/attention investigation is paused; no new LLM implementation has been added.

## Saved evidence included

A compact, explicitly selected subset of the otherwise ignored `artifacts/` directory is tracked: CMIN configurations, protocols, summary/analysis JSON, validation/training histories, selected evaluation prediction CSVs, and selected audit records. These preserve results for reading and checking without starting model training. [The 19 October manifest](../reports/provenance/research_sync_manifest_2026-10-19.json) lists the original selected artifact paths and SHA-256 hashes. [The 20 October additions manifest](../reports/provenance/research_share_manifest_2026-10-20.json) covers supporting records added to make linked evidence available, including the worked CMIN example and pinned acquisition inventory. Both manifests moved from `output/` to `reports/provenance/` without changing their bytes.

With Python and Git installed, run `python scripts/audit_repository.py` to check the tracked package without fetching datasets, loading checkpoints, or writing into study folders. The initial sharing audit is in `reports/repository_audit_2026-10-20.json`; the subsequent removal/refactor verification is in `reports/code_cleanup_2026-10-20.json`. The original 107-test result belongs to the pre-cleanup source version; use [the code guide](code_guide.md) for the current suite.

The timing-only implementation remains archived under `artifacts/cmin_timing_2026-10-19_v1/source_snapshot/`, together with its cleanup note. It has not been restored to active source or test folders. See that experiment's `ARCHIVED_CODE.md` before attempting to execute it.

## Files remaining local

Raw news/price downloads, full input arrays, headline feature caches, pretrained encoder weights, neural checkpoints, and local third-party source checkouts are excluded. This Git snapshot is sufficient for viewing the meeting material and inspecting the included results; it is not a complete copy of the workstation's execution environment. Some historical report links point to full local-only evidence. Completion/hash records may also reference omitted local files; their presence does not imply a complete checkpoint archive on the laptop.

For rebuilding experiments, use the relevant versioned protocol and [CMIN runbook](cmin_runbook.md). Reinstall dependencies and obtain the pinned sources/models first. Do not launch training simply to open a report. Existing completed experiments should retain their identities and results.

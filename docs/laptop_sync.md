# Research files available on the laptop

From an existing clone, run `git pull --ff-only origin main`. For a new clone, run:

```powershell
git clone https://github.com/jadghazi/finin-research-prototype.git
cd finin-research-prototype
```

Start with the links in the repository README. Meeting PDFs and available LaTeX sources are under `output/pdf/`, figures under `output/figures/`, findings under `reports/`, and experiment protocols under `docs/`. Source, tests and configurations are included. Historical reports and PDFs are retained with their original dates.

## Saved evidence included

A compact, explicitly selected subset of the otherwise ignored `artifacts/` directory is tracked: CMIN configurations, protocols, summary/analysis JSON, validation/training histories, selected evaluation prediction CSVs, and selected audit records. These preserve results for reading and checking without starting model training. `output/research_sync_manifest_2026-10-19.json` lists the selected artifact paths and SHA-256 hashes.

The timing-only implementation remains archived under `artifacts/cmin_timing_2026-10-19_v1/source_snapshot/`, together with its cleanup note. It has not been restored to active source or test folders. See that experiment's `ARCHIVED_CODE.md` before attempting to execute it.

## Files remaining local

Raw news/price downloads, full input arrays, headline feature caches, pretrained encoder weights, neural checkpoints, and local third-party source checkouts are excluded. This Git snapshot is sufficient for viewing the meeting material and inspecting the included results; it is not a complete copy of the workstation's execution environment. Some historical report links point to full local-only evidence. Completion/hash records may also reference omitted local files; their presence does not imply a complete checkpoint archive on the laptop.

For rebuilding experiments, use the relevant versioned protocol and [CMIN runbook](cmin_runbook.md). Reinstall dependencies and obtain the pinned sources/models first. Do not launch training simply to open a report. Existing completed experiments should retain their identities and results.

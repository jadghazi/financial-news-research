# Run the public FININ-inspired pipeline

**Status — 20 October 2026:** further financial experiments are paused. Start with the [README](../README.md), [decision record](../DECISIONS.md), and [repository guide](repository_guide.md). The frozen baseline study is in `configs/cmin_panel_v1.json`; the architecture is in [cmin_pipeline.md](cmin_pipeline.md), and statistical choices are in [the baseline protocol](cmin_panel_protocol_2026-10-14.md).

## Inspect a fresh Git clone

Reports, PDFs, JSON summaries, and selected prediction CSVs are included. Inspecting them does not require downloading a model or fitting anything. Run the standard-library audit from the repository root:

```powershell
python scripts/audit_repository.py
```

This verifies the tracked sharing package, manifests, PDF build hashes, and CMIN CSV metrics. It does not execute the numbered experiment scripts. The complete engineering suite needs Python 3.12, CPU PyTorch, and both requirements files:

```powershell
python -m venv .venv
& .venv/Scripts/python.exe -m pip install torch==2.5.1
& .venv/Scripts/python.exe -m pip install -r requirements.txt -r requirements-cmin.txt
& .venv/Scripts/python.exe -m unittest discover -s tests -q
```

No package installation is needed just to open a PDF. Tests include tiny synthetic models, not another financial experiment. Dependency versions above describe the recorded runtime; this documentation update did not reinstall packages or certify every operating system.

## Full reproduction prerequisites

The Git snapshot is **not a self-contained training archive**. Raw sources, arrays, encoder weights, selected logits arrays, checkpoints, and full per-epoch records remain local. Historical completion manifests reference these omitted files. Existing study directories in a clone contain only selected evidence; passing them to the original runner is not equivalent to having a complete original study.

Acquisition uses the pinned source inventory in [cmin_source_manifest.json](../artifacts/feasibility/public_reference_2026-10-14/cmin_source_manifest.json), commit `3f571a53b81d8f3ea7b3afb9877bdbdbeae54c97`. The recorded correction in `AMENDMENT_acquisition.json` precedes the original preparation and fitting. The baseline downloaded 24 cohort source files (about 158 MB), not the whole upstream repository.

Caching intentionally loads model snapshots offline through Hugging Face's local cache. The following pinned snapshots must first be obtained separately:

| Purpose | Model | Revision |
|---|---|---|
| Headline/company text | `FacebookAI/roberta-base` | `e2da8e2f811d1448a5b465c236feacd80ffbac7b` |
| Financial sentiment | `mikeysharma/finance-sentiment-analysis` | `4353ecd593de607cbb64147c8e2d21b82d842014` |

See `src/reproduction_v2_features.py` for tokenizer/pooling and model-loading checks, and each experiment's feature manifest for recorded provenance. Do not silently replace models or rebuild evidence under an old identity. A new reproduction needs separate output paths, the same audited source/data/selection rules, and a recorded protocol. Some original orchestration stages use fixed paths or refuse to overwrite completed runs; they require deliberate adaptation for a new namespace rather than blind copy-and-paste execution.

## Original workstation commands

The commands below document how the original baseline was executed with its full local data, pinned models, and preserved run directories. **They are not an inspection recipe for a partial Git clone.** They do not automatically test every professor question. Use the project runtime from the repository root only when the full prerequisites and a separately authorized experiment are in place:

```powershell
& .venv/Scripts/python.exe scripts/16_cmin_panel.py acquire
& .venv/Scripts/python.exe scripts/16_cmin_panel.py prepare
& .venv/Scripts/python.exe scripts/17_cmin_source_audit.py
& .venv/Scripts/python.exe scripts/16_cmin_panel.py cache
& .venv/Scripts/python.exe scripts/16_cmin_panel.py run
& .venv/Scripts/python.exe scripts/18_cmin_verify.py
& .venv/Scripts/python.exe scripts/16_cmin_panel.py inspect
```

The source audit is required before fitting. It replays the deterministic URL audit initially performed as an inline command during implementation, without replacing the existing audit. It adds no model fitting or selection. The encoders must already be present at their pinned local revisions; caching is offline. Current raw sources, features, predictions, checkpoints, traces, source snapshots and manifests are under `artifacts/cmin_panel_2026-10-14_v1/`. `inspect` reads the completed summary; `run` verifies completed-study and fit hashes before returning an existing result. An incomplete fit is retained for diagnosis, not automatically retrained.

Even `scripts/16_cmin_panel.py inspect` writes command-start/completion records. To read the summary without changing experiment folders, open its JSON directly or use the repository audit. Several verification stages add seals; these are not interchangeable with read-only inspection.

`18_cmin_verify.py` additionally recomputes training-only transforms and checks the saved fixed logistic reference. It seals their auxiliary hashes after verification and refuses later changes. It does not fit a model or rerun neural inference. The original completion manifest is retained.

For a later explicitly planned professor comparison, save a new JSON configuration and protocol, then reuse the audited data and cache:

```powershell
& .venv/Scripts/python.exe scripts/16_cmin_panel.py run --config PATH_TO_NEW_CONFIG --data-directory artifacts/cmin_panel_2026-10-14_v1 --study-directory PATH_TO_NEW_OUTPUT --protocol PATH_TO_NEW_PROTOCOL
```

The new configuration retains the source pin, cohort, five price sessions, cap, availability rule and development boundary. Change only the intended control, with explicit `arms`, `comparison_pairs`, seeds and fit/time ceilings. Optional `news_age_sessions` restricts news since arrival; `half_life_hours` is required for `pooled_decay`. Pair decay with `pooled_uniform` to preserve predictor geometry. Use `no_direct_market` for the final market block; `news_only` also removes query conditioning. These commands must not be used to tune repeatedly on the already inspected development periods. The reserved late-2021 outcomes remain outside this baseline data preparation.

Timing-only scripts 27/28 and their test/module are archived, not active. The [archive map](../artifacts/cmin_timing_2026-10-19_v1/ARCHIVED_CODE.md) explains the exact copies and original paths. Historical report commands require those files and the omitted full local run evidence; the documentation audit does not restore them or rerun that study.

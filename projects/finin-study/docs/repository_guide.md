# Repository guide for readers and agents

Updated 20 October 2026. Start with the [README](../README.md) and [decision record](../DECISIONS.md). This guide indexes only the standalone FININ-inspired project, now under `projects/finin-study/`. All relative execution/evidence paths refer to this project root. The current investigation is paused, not declared successful.

## Reading order

1. Read the README's questions/results table and the decision record's current handoff.
2. Use the [CMIN pipeline](cmin_pipeline.md) for the latest executed architecture and timing assumptions.
3. Read the relevant dated report, then its protocol, configuration, and saved summary/predictions.
4. Consult earlier phases only when investigating their decisions or implementation. Their next-step recommendations are historical.
5. Read [AGENTS.md](../AGENTS.md) before extending the work. Work with one main agent by default; optional specialist configurations do not authorize automatic delegation.

## Latest CMIN studies

All use the fixed 12-company cohort, two development evaluation periods, and seeds 17/42. Later results on inspected periods are exploratory. Reserved late-2021 outcomes were not used. “Log loss” in the reports averages losses from seed models, not probabilities from an ensemble.

| Study | Findings | Protocol/configuration | Saved evidence |
|---|---|---|---|
| Baseline; 12 neural fits; 160-input MLP | [Report](../reports/cmin_public_pipeline_2026-10-14.md) | [Protocol](cmin_panel_protocol_2026-10-14.md), [config](../configs/cmin_panel_v1.json) | [Summary](../artifacts/cmin_panel_2026-10-14_v1/summary.json) |
| News dependence; fixed-model probes | [Report](../reports/cmin_news_dependency_2026-10-14.md) | Outcome-independent validation-example selection described in report | [Summary](../artifacts/diagnostics/cmin_news_consumption_2026-10-14/summary.json), [examples](../artifacts/diagnostics/cmin_news_consumption_2026-10-14/examples.json) |
| 48-hour temporal decay; 8 fits; 96-input MLP | [Report](../reports/cmin_news_decay_2026-10-14.md) | [Protocol](cmin_decay_protocol_2026-10-14.md), [config](../configs/cmin_decay_48h_v1.json) | [Training summary](../artifacts/cmin_decay_48h_2026-10-14_v1/summary.json), [analysis](../artifacts/cmin_decay_48h_2026-10-14_v1/analysis/summary.json) |
| Current versus older news; 8 new/4 reused fits; 96-input MLP | [Report](../reports/cmin_older_news_2026-10-14.md) | [Protocol](cmin_older_news_protocol_2026-10-14.md), [config](../configs/cmin_older_news_v1.json) | [Analysis](../artifacts/cmin_older_news_2026-10-14_v1/analysis/summary.json) |
| No headline self-attention; 4 new fits; 160-input MLP | [Report](../reports/cmin_interactions_2026-10-14.md) | [Protocol](cmin_interaction_protocol_2026-10-14.md), [config](../configs/cmin_interaction_v1.json) | [Fit summary](../artifacts/cmin_interaction_2026-10-14_v1/summary.json), [trace inspection](../artifacts/diagnostics/cmin_interactions_meeting_2026-10-14_v1/summary.json) |
| Newer-news policies; 12 new fits; 160-input MLP | [Report](../reports/cmin_timing_2026-10-19.md) | [Protocol](cmin_timing_protocol_2026-10-19.md), [config](../artifacts/cmin_timing_2026-10-19_v1/config.json) | [Analysis](../artifacts/cmin_timing_2026-10-19_v1/analysis.json), [archive map](../artifacts/cmin_timing_2026-10-19_v1/ARCHIVED_CODE.md) |

Each tracked CMIN fit folder contains `history.json`, `summary.json`, and `predictions.csv`. CSV columns are ticker, forecast date, target date, cutoff UTC, binary target, and raw logit. The sigmoid of the logit is the UP probability; logit ≥ 0 predicts UP. Raw neural metrics differ from the validation-selected constant-fallback system where a fallback was selected. All-news fits reused by the older-news study live in the decay directory.

Age restrictions are half-open intervals: `news_age_sessions: [0, 1]` means only age zero. Arrival-session age differs from source-date age in hours and from true publication age. The retired decay proposal was superseded by the completed protocol/report and is recoverable from the pre-cleanup Git version.

## Code map and command side effects

The authoritative implementation map is [docs/code_guide.md](code_guide.md). It explains every retained source module, its main functions, input/output shapes, dependency flow, command side effects, and test coverage. Each retained production module and command links back to that guide in its docstring.

The active source tree contains 10 files: two package initializers, the CMIN data and experiment modules, frozen features, training, evaluation, shared runtime helpers, and the two model modules. Retired prototype/rebuild code, obsolete exporters, and their tests were removed during cleanup. The full pre-cleanup source is recoverable at Git commit `3b3f1f7`; source snapshots under `artifacts/` remain intact.

Use `scripts/audit_repository.py` to inspect tracked files and saved CSV metrics without model execution. Numbered CMIN commands can download, encode, fit, write reports, or seal audits. Their historical source/hash guards must not be bypassed to accept the cleaned implementation. [Execution prerequisites](cmin_runbook.md).

## Historical reports and their scope

| Phase | Reports and documentation |
|---|---|
| Original NIFTY/SPY preparation and results | [Data audit](../reports/data_audit.md), [preparation](../reports/data_preparation.md), [results](../reports/results.md); historical architecture, runbook and fixed plan at Git commit `3b3f1f7`, `reports/feature_benchmark.json`, `reports/prototype_audit.json` |
| 11 October diagnosis and alternative-source screen | [Gain diagnosis](../reports/v1_gain_diagnosis_2026-10-11.md), [FNSPID audit](../reports/fnspid_initial_audit_2026-10-11.md) |
| 12 October implementation/learning checks | [Readiness](../reports/pipeline_readiness_2026-10-12.md), [learning](../reports/learning_check_2026-10-12.md), [encoder comparison](../reports/encoder_comparison_2026-10-12.md), [headline coverage](../reports/headline_coverage_2026-10-12.md), [cap comparison](../reports/headline_cap_comparison_2026-10-12.md) |
| 13 October context, optimization, and temporal pooling | [History](../reports/history_comparison_2026-10-13.md), [pipeline diagnosis](../reports/pipeline_diagnosis_2026-10-13.md), [training/initial decay](../reports/ml_training_and_decay_2026-10-13.md) |
| StockNet tweet rebuild and SPNews adaptation | [StockNet report](../reports/reproduction_v2_2026-10-13.md), [SPNews report](../reports/public_headline_pipeline_2026-10-13.md) |
| Reference decisions and literature | [13 October feasibility](../reports/reference_feasibility_and_professor_tasks_2026-10-13.md), [superseding CMIN decision](../reports/public_reference_decision_2026-10-14.md), [news-age literature](../reports/news_age_literature_2026-10-14.md) |

Some historical detailed runs remain workstation-only. Linked support records are included where practical, but a cited directory name or checksum does not mean its full underlying dataset/checkpoints are in Git. Do not use old “current default” statements to override the dated decision sequence.

## Meeting material

| File under `output/pdf/` | Intended use |
|---|---|
| [FININ_meeting_progress_2026-10-14.pdf](../output/pdf/FININ_meeting_progress_2026-10-14.pdf) | Three-page CMIN progress report including dependence probes and attention investigation; predates timing study |
| [FININ_attention_and_MLP_explainer_2026-10-19.pdf](../output/pdf/FININ_attention_and_MLP_explainer_2026-10-19.pdf) | Two explanatory visuals; example headlines, scores and probabilities are illustrative |

Both retained PDFs have `.tex` sources and `.build.json` compilation/verification records beside them. At the user's request, all other output documents and standalone figures were removed from the current checkout on 20 October; they remain in Git history at `0be47b9`. Historical builder scripts were retired and remain recoverable at Git commit `3b3f1f7`; they are not current commands. `artifacts/meeting/` retains the earlier interactive walkthrough and selected predictions. Older build-source hash entries can describe earlier versions of subsequently edited guidance such as AGENTS.md.

## What is intentionally retained or excluded

- Keep dated protocols, reports, configurations, negative results, source snapshots, prediction CSVs, histories, and verification records. Duplicate source snapshots preserve the exact code used for a study; active code may subsequently evolve.
- Keep the archived timing implementation. Its removal from active code was a cleanup decision after a completed experiment, not withdrawal of its evidence.
- `AGENTS.md` and `.agents/skills/` retain the research workflow. Obsolete optional specialist configurations were removed; work uses one main agent by default.
- Exclude environments, bytecode, caches, raw corpora, checkpoints, large feature arrays, temporary downloads, and third-party checkouts. These remain ignored locally.
- [19 October evidence manifest](../reports/provenance/research_sync_manifest_2026-10-19.json) and [20 October additions manifest](../reports/provenance/research_share_manifest_2026-10-20.json) identify explicitly included artifacts. These metadata files moved unchanged out of `output/`. Earlier completion manifests intentionally retain references to omitted local files.

For inspection versus full reproduction, see the [CMIN runbook](cmin_runbook.md). For sharing checks, see the [original 20 October audit](../reports/repository_audit_2026-10-20.json), [subsequent code cleanup audit](../reports/code_cleanup_2026-10-20.json), and [project relocation checks](../reports/project_relocation_2026-10-20.json). The first two records precede relocation; their original project-relative paths and scientific evidence remain unchanged. The tag `finin-before-project-split-2026-10-20` preserves the standalone repository layout. No model arithmetic or saved results were changed by the move.

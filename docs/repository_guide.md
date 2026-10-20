# Repository guide for readers and agents

Updated 20 October 2026. Start with the [README](../README.md) and [decision record](../DECISIONS.md). This guide indexes the existing FININ-inspired work; the proposed direct-LLM project has not been implemented. The current investigation is paused, not declared successful.

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

Age restrictions are half-open intervals: `news_age_sessions: [0, 1]` means only age zero. Arrival-session age differs from source-date age in hours and from true publication age. The [decay proposal](cmin_decay_proposal_2026-10-14.md) predates execution; the protocol/report supersede its proposed actions.

## Code map and command side effects

| Area | Files | Purpose |
|---|---|---|
| Latest CMIN data | `src/cmin_panel_data.py` | Pinned acquisition, deterministic company/headline selection, source-date alignment, exchange sessions, labels, rolling plans |
| Latest CMIN experiment | `src/cmin_panel_experiment.py` | Tensor assembly, training-only transforms, model arms, fit/selection, paired uncertainty, retained evidence |
| Model implementations | `src/models/paper_finin.py`, `research_finin.py` | FININ-style fusion/attention and multi-session/pooled adaptations |
| Frozen encoders | `src/reproduction_v2_features.py` | Pinned BGE/RoBERTa and sentiment features; actual CMIN encoder is RoBERTa |
| Training/evaluation | `src/research_training.py`, `evaluation.py`; CMIN runner helpers | Controlled CPU training, stable logit scoring, diagnostics and model selection |
| CMIN stage commands | `scripts/16`–`18` | Acquire, prepare, encode, fit, inspect, and verify; some stages write logs/audit seals even when not fitting |
| CMIN follow-ups | `scripts/19`–`26` | Dependency probes, decay, older-news comparison, subgroup uncertainty, and trace/interaction analyses; inspect side effects before running |
| Archived timing-only code | `artifacts/cmin_timing_2026-10-19_v1/source_snapshot/` | Exact executed files; four removed active files map through `ARCHIVED_CODE.md` |
| Earlier pipeline | `src/data/`, `features/`, `training_data.py`, `training.py`, `models/finin.py`; `scripts/01`–`09` | Original SPY/NIFTY prototype; several commands overwrite fixed reports or fit models |
| Earlier diagnoses | Encoder, headline-cap, history, calibration, positive-control and temporal-study modules; `scripts/10`–`15` | Bounded historical experiments, not current CMIN defaults |
| Public-data rebuilds | `src/reproduction_v2_*`, `spnews_*`, `financial_headline_cache.py` | StockNet/SPNews acquisition, feature guards, probes and matched controls |
| Offline sharing audit | `scripts/audit_repository.py` | Standard-library inspection of tracked files, links, hashes and saved CSV metrics; no network or financial fitting |

`tests/` checks exchange/date alignment, shapes, masking, gradient flow, evaluation-state handling, cache provenance, and synthetic controls. It contains small synthetic fits, not a new market evaluation. Do not run acquisition, cache generation, model training, or report export to read existing findings.

## Historical reports and their scope

| Phase | Reports and documentation |
|---|---|
| Original NIFTY/SPY preparation and results | [Data audit](../reports/data_audit.md), [preparation](../reports/data_preparation.md), [results](../reports/results.md), [architecture](architecture.md), [runbook](runbook.md), [fixed plan](../REPRODUCTION_PLAN.md), `reports/feature_benchmark.json`, `reports/prototype_audit.json` |
| 11 October diagnosis and alternative-source screen | [Gain diagnosis](../reports/v1_gain_diagnosis_2026-10-11.md), [FNSPID audit](../reports/fnspid_initial_audit_2026-10-11.md) |
| 12 October implementation/learning checks | [Readiness](../reports/pipeline_readiness_2026-10-12.md), [learning](../reports/learning_check_2026-10-12.md), [encoder comparison](../reports/encoder_comparison_2026-10-12.md), [headline coverage](../reports/headline_coverage_2026-10-12.md), [cap comparison](../reports/headline_cap_comparison_2026-10-12.md) |
| 13 October context, optimization, and temporal pooling | [History](../reports/history_comparison_2026-10-13.md), [pipeline diagnosis](../reports/pipeline_diagnosis_2026-10-13.md), [training/initial decay](../reports/ml_training_and_decay_2026-10-13.md) |
| StockNet tweet rebuild and SPNews adaptation | [StockNet report](../reports/reproduction_v2_2026-10-13.md), [rebuild design](reproduction_v2.md), [SPNews report](../reports/public_headline_pipeline_2026-10-13.md) |
| Reference decisions and literature | [13 October feasibility](../reports/reference_feasibility_and_professor_tasks_2026-10-13.md), [superseding CMIN decision](../reports/public_reference_decision_2026-10-14.md), [news-age literature](../reports/news_age_literature_2026-10-14.md) |

Some historical detailed runs remain workstation-only. Linked support records are included where practical, but a cited directory name or checksum does not mean its full underlying dataset/checkpoints are in Git. Do not use old “current default” statements to override the dated decision sequence.

## Meeting material

| File under `output/pdf/` | Intended use |
|---|---|
| [FININ_meeting_progress_2026-10-14.pdf](../output/pdf/FININ_meeting_progress_2026-10-14.pdf) | Three-page CMIN progress report including dependence probes and attention investigation; predates timing study |
| [FININ_attention_and_MLP_explainer_2026-10-19.pdf](../output/pdf/FININ_attention_and_MLP_explainer_2026-10-19.pdf) | Two explanatory visuals; example headlines, scores and probabilities are illustrative |

Both retained PDFs have `.tex` sources and `.build.json` compilation/verification records beside them. At the user's request, all other output documents and standalone figures were removed from the current checkout on 20 October; they remain in Git history at `0be47b9`. Historical builder scripts remain as executed source and can recreate old outputs if explicitly run. `artifacts/meeting/` retains the earlier interactive walkthrough and selected predictions. Older build-source hash entries can describe earlier versions of subsequently edited guidance such as AGENTS.md.

## What is intentionally retained or excluded

- Keep dated protocols, reports, configurations, negative results, source snapshots, prediction CSVs, histories, and verification records. Duplicate source snapshots preserve the exact code used for a study; active code may subsequently evolve.
- Keep the archived timing implementation. Its removal from active code was a cleanup decision after a completed experiment, not withdrawal of its evidence.
- Optional `.codex/agents/` and `.agents/skills/` contain research workflow instructions, not experimental results. They do not launch agents automatically.
- Exclude environments, bytecode, caches, raw corpora, checkpoints, large feature arrays, temporary downloads, and third-party checkouts. These remain ignored locally.
- [19 October evidence manifest](../reports/provenance/research_sync_manifest_2026-10-19.json) and [20 October additions manifest](../reports/provenance/research_share_manifest_2026-10-20.json) identify explicitly included artifacts. These metadata files moved unchanged out of `output/`. Earlier completion manifests intentionally retain references to omitted local files.

For inspection versus full reproduction, see the [CMIN runbook](cmin_runbook.md). For sharing checks, see the [20 October audit](../reports/repository_audit_2026-10-20.json). The repository has not been renamed, relocated into a new project hierarchy, or mixed with a new LLM implementation in this documentation update.

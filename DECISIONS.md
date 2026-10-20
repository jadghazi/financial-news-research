# Research decisions and experiment history

Updated 20 October 2026. This is a retrospective, evidence-linked record, not a preregistration. Dated protocols and saved configurations document what was fixed before individual experiments. Reports are preserved as written; later decisions below supersede earlier next-step recommendations.

## Current handoff

**Pause further embedding/attention pipeline development.** Following the professor discussion, the next proposed task is an independent direct-LLM headline prediction experiment. No scraper, LLM prediction runner, API calls, or results for that direction have been created. Keep the completed work available for inspection and later matched comparisons.

The latest completed empirical result is the [19 October timing study](reports/cmin_timing_2026-10-19.md). The model responds to news, but reliable incremental forecasting value remains unproven. We have not identified one root cause. The original CMIN source-date-plus-two-calendar-day rule remains in code; timing-only changes are archived.

## 1. Establish a feasible independent prototype

**Decision:** implement FININ's core flow using accessible NIFTY headlines and SPY prices rather than claim replication of unavailable Reuters/TRNA experiments. Frozen BGE-small vectors and finance TinyBERT sentiment substituted for the original data/representations; one input day, at most 16 headlines, three seeds, one chronological split.

**Checks and result:** 2,107 examples (1,475/315/317); 18 trained runs. Full-model test accuracy averaged 58.4% versus 58.0% always UP, with balanced accuracy about 50%. These were near-majority predictions, not convincing useful news prediction. Preserve this as historical version 1.

Evidence: [data preparation](reports/data_preparation.md), [results](reports/results.md), [audit](reports/prototype_audit.json). The original fixed plan and architecture notes are recoverable at Git commit `3b3f1f7`; they were removed from active documentation during the user-requested cleanup.

## 2. Determine what produced the tiny apparent gain — 11 October

**Question:** was the small version-1 advantage evidence of news use?

**Checks and result:** replayed 18 saved checkpoints; the mean 0.315-percentage-point gain came from five decisions from one seed in March 2020. Fixed-model interventions preserved that gain when dated news summaries/query weights were replaced; removing the direct market representation removed it. This supported a market-branch explanation for those decisions, without establishing a general price mechanism.

**Decision:** diagnose learnability and input dependence before adding mechanisms. A small reported gain alone was insufficient.

Evidence: [gain diagnosis](reports/v1_gain_diagnosis_2026-10-11.md). FNSPID was [screened but not adopted](reports/fnspid_initial_audit_2026-10-11.md); that was a bounded feasibility decision, not proof of inferior data quality.

## 3. Test learning, representation, and news quantity — 12 October

**Checks:** small-batch memorization and longer learning curves; an actual pinned RoBERTa substitution; headline coverage; a 16-versus-64-headline comparison.

**Result:** the architecture could fit data and later checkpoints could depend on text, while validation probability quality often deteriorated. Neither the tested encoder substitution nor more headlines demonstrated a reliable rescue. The old recommendation to retain BGE applied to this prototype; the later CMIN implementation uses RoBERTa.

Evidence: [readiness](reports/pipeline_readiness_2026-10-12.md), [learning](reports/learning_check_2026-10-12.md), [encoder comparison](reports/encoder_comparison_2026-10-12.md), [coverage](reports/headline_coverage_2026-10-12.md), [headline cap](reports/headline_cap_comparison_2026-10-12.md).

## 4. Separate engineering health from financial signal — 13 October

**Checks:** unseen synthetic rules, shuffled labels, gradient/manual-forward checks, sampled feature re-encoding, and simpler chronological timing baselines. Synthetic success and headline-sensitive forecasts supported a functioning computation path, not predictive usefulness of real news.

**Engineering change:** the historical trainer could consume training randomness during evaluation. A new trainer isolates evaluation randomness/modes, checks diagnostic state mutation, and retains selected/final states separately. This repair is documented prospectively; it is not a demonstrated explanation for the negative historical findings.

**Further tests:** a bounded 129-neural-fit training study and 75-convex-fit initial temporal-pooling comparison found overfitting, near-majority decisions, and no convincing news benefit. This early three-day linear decay test differs from the later eight-fit CMIN neural experiment.

Evidence: [pipeline diagnosis](reports/pipeline_diagnosis_2026-10-13.md), [training and initial decay](reports/ml_training_and_decay_2026-10-13.md), [three-session context](reports/history_comparison_2026-10-13.md), [safe trainer](src/research_training.py).

## 5. Evaluate alternative public inputs — 13–14 October

| Phase | What was evaluated | Result and decision |
|---|---|---|
| StockNet rebuild | Eight companies, timestamped tweets, matched timing/no-news controls; actual frozen RoBERTa | Development evidence failed to persist in the fixed final quarter. Keep the negative benchmark. An initial cache incorrectly labeled RoBERTa contained BGE; the corrected RoBERTa study and correction were retained. [Report](reports/reproduction_v2_2026-10-13.md) |
| SPNews adaptation | Public company-linked headlines with S&P 500 prices | Computationally functional, but weak/unstable generalization and unresolved historical availability. [Report](reports/public_headline_pipeline_2026-10-13.md) |
| Reference/data feasibility | Bounded screen of public sources and original-paper access | The 13 October licensed-data-first priority was superseded on 14 October by an explicitly authorized bounded CMIN-US adaptation. [Earlier feasibility](reports/reference_feasibility_and_professor_tasks_2026-10-13.md), [CMIN decision](reports/public_reference_decision_2026-10-14.md) |

These are separate datasets and experiment versions. Their final-period scores are not CMIN development scores. The CMIN decision's “proposed, not executed” section describes its status at that time; the following stages subsequently executed it.

## 6. Build and freeze the CMIN-US baseline — 14 October

**Decision:** a shared model over 12 preselected companies, five price/news-arrival sessions, at most six headlines per bucket, frozen RoBERTa768 and TinyBERT sentiment. Predict next-session adjusted-close direction. Use source-date D+2 midnight UTC as a declared but unverified availability proxy.

**Checks:** labels, exchange calendar, train-only transforms, masks, source-date disagreements, cache identities, selected checkpoints, saved traces, and metrics. The 12 financial neural fits compare prices, mean headlines, and full attention across two seeds and two periods. A manifest-writing acquisition correction was recorded before preparation/fitting.

**Result:** raw full-model loss 0.690388 versus 0.690095 prices and 0.689820 training-prevalence probabilities. No convincing incremental news benefit. The final MLP receives 160 values, not raw headlines or a separate list of attention weights.

Evidence: [protocol](docs/cmin_panel_protocol_2026-10-14.md), [configuration](configs/cmin_panel_v1.json), [report](reports/cmin_public_pipeline_2026-10-14.md), [summary](artifacts/cmin_panel_2026-10-14_v1/summary.json), [worked example](artifacts/cmin_panel_2026-10-14_v1/walkthrough.md).

## 7. Prove computational news dependence — 14 October

**Checks:** hold trained weights and prices fixed; replace headline vectors with their training mean, bypass headline self-attention, or replace market-query weights with uniform weights. Examples were selected without outcomes.

**Result:** headline replacement changed 44/44 news-bearing model-cases, self-attention bypass 44/44, and query uniformization 16/44. Maximum probability changes were 6.76, 7.84, and 3.05 percentage points respectively. Two seeds can evaluate the same company/date; these are not 44 independent market dates.

**Decision:** the news path is used. Stop describing the pipeline as disconnected. These interventions do not establish that the original answer was more accurate or that attention preserved relevant meaning.

Evidence: [report](reports/cmin_news_dependency_2026-10-14.md), [saved diagnostic](artifacts/diagnostics/cmin_news_consumption_2026-10-14/summary.json).

## 8. Test explicit decay with an identifiable comparison — 14 October

**Decision:** pool daily news representations before prediction, retaining all five price representations. Multiplying separate concatenated day slots by fixed constants can be absorbed by MLP weights; therefore compare uniform versus exponential temporal pooling with the same **96-input** geometry. The 48-hour half-life was fixed before fitting, not estimated from results. Optimizer `weight_decay` is separate regularization.

**Result:** eight fits; decay-minus-uniform loss −0.000324, conditional interval [−0.001177, +0.000539]. Age-bucket removal effects were small and uncertain. No stable practical benefit or empirical diffusion curve was established; the chosen half-life is not an estimated market property.

Evidence: [protocol](docs/cmin_decay_protocol_2026-10-14.md), [report](reports/cmin_news_decay_2026-10-14.md), [analysis including age-sensitivity values](artifacts/cmin_decay_48h_2026-10-14_v1/analysis/summary.json). The earlier standalone figure was removed from the current output selection on 20 October and remains in Git history.

## 9. Isolate older news while holding prices fixed — 14 October

**Decision:** compare current-arrival-bucket news with all five buckets, retaining five price sessions and matched pooled geometry. Fit four current-news and four price-only models; reuse all four existing uniform all-news models without favorable selection.

**Clarification verified against code:** `news_age_sessions: [0, 1]` is a half-open interval, **0 ≤ age < 1**, not two included ages. Only the current arrival bucket survives. “Older” is ages 1–4 since proxy arrival, not verified publication age. This matches the report and the runner's mask checks; no correction to results is needed.

**Result:** all-news minus current-news loss +0.000560, interval [−0.000031, +0.001155]. No robust useful older-news addition. The current-only model's lowest point estimate also failed to establish improvement over prices/prevalence.

Evidence: [protocol](docs/cmin_older_news_protocol_2026-10-14.md), [saved current-news configuration](artifacts/cmin_older_news_2026-10-14_v1/recent/config.json), [age-mask implementation](src/cmin_panel_experiment.py), [report](reports/cmin_older_news_2026-10-14.md), [analysis](artifacts/cmin_older_news_2026-10-14_v1/analysis/summary.json).

## 10. Investigate headline interactions and preservation — 14 October

**Checks:** reconstruct attention and final-MLP inputs from saved traces; compare four new no-self-attention fits against matching original full-model fits, retaining market-query weighting and **160-input** geometry.

**Result:** removing self-attention increases pooled loss by 0.002999 (conditional interval [0.001058, 0.005092]), but the period effects reverse sign. Three of four selected model snapshots have almost direction-identical post-attention headline vectors. This is a diagnostic concern; magnitude, learned coordinate changes, and limited snapshots prevent concluding that relevant information was lost.

**Decision:** preserve the negative/unstable evidence. A systematic relevant-story retention study and a retrained query-only ablation remain unperformed.

Evidence: [protocol](docs/cmin_interaction_protocol_2026-10-14.md), [report](reports/cmin_interactions_2026-10-14.md), [trace analysis](artifacts/diagnostics/cmin_interactions_meeting_2026-10-14_v1/summary.json).

## 11. Test whether newer news fixes the result — 19 October

**Decision:** compare original D+2 availability with D+1, permissive same-date, and naive clocks interpreted as New York time. Preserve prices, rows, initial weights, shared transforms, and 160-input geometry. Exactly 12 new fits; no later tuning.

**Result:** original loss 0.690388; D+1 0.690410; same-date 0.690502; clock-based 0.690459. None beats the original or prices in average loss. News became substantially fresher, but the policies also changed headline selection/coverage. Same-date news can include after-close information. Source timezone remains unverified.

**Checks:** all 16 selected old/new news checkpoints reproduced complete saved logits exactly; sampled original encoder outputs were reproduced; 1,763 protected files remained unchanged. Training loss improved while validation loss often deteriorated, consistent with overfitting but not proof of a unique root cause.

**Cleanup:** at the user's request, four timing-only code/test files were removed from active folders after exact archive-copy checks. Keep their [archive/restore map](artifacts/cmin_timing_2026-10-19_v1/ARCHIVED_CODE.md). The active timing rule was not replaced.

Evidence: [protocol](docs/cmin_timing_protocol_2026-10-19.md), [report](reports/cmin_timing_2026-10-19.md), [analysis](artifacts/cmin_timing_2026-10-19_v1/analysis.json), [checkpoint replay](artifacts/cmin_timing_2026-10-19_v1/checkpoint_replay.json).

## 12. Preserve and share the completed work — 19–20 October

The 19 October Git snapshot `2a23808` added the CMIN studies, meeting PDFs, and selected experiment evidence. Larger raw data, caches, weights, checkpoints, and third-party checkouts remain local; a Git clone is not a full training environment. Source snapshots are intentional provenance copies, not redundant active implementations.

On 20 October, the user requested a professor-readable documentation and organization audit before any repository rename or new-project reorganization. The sharing update adds this record, a current README, a repository index, linked supporting records, and an offline audit. Historical experiment results and PDF bytes are preserved. [Audit result](reports/repository_audit_2026-10-20.json), [scope](docs/laptop_sync.md).

After that sharing audit, the user requested that `output/` retain only the attention/MLP explainer and `FININ_meeting_progress_2026-10-14`, including their sources/build records. Other output documents and figures were removed from the current checkout; they remain in Git history at `0be47b9`. Evidence-sharing manifests moved unchanged to `reports/provenance/`. [Cleanup record](reports/output_cleanup_2026-10-20.json). No model, dataset, experiment result, or retained PDF was changed; the earlier sharing audit describes the snapshot before this output selection.

The user subsequently requested a substantive source/folder audit, removal of the outdated reproduction plan and hardware instructions, and a code guide linked from the code. Retired NIFTY/StockNet/SPNews execution paths, their tests, old exporters, and optional specialist configurations were removed. Hashing, seeding, stable-logit metrics, and frozen-cache helpers were extracted so current training no longer imports retired experiments. Model arithmetic and saved scientific evidence were preserved. The current source has a new provenance identity; exact old commands require the full pre-cleanup checkout `3b3f1f7`, not patched checksum guards. [Code guide](docs/code_guide.md), [cleanup verification](reports/code_cleanup_2026-10-20.json).

## Claims and remaining questions

- **Supported:** the implementation can learn synthetic rules; selected financial predictions depend on news and attention; the declared comparisons completed and retain negative findings.
- **Not established:** repeatable incremental forecasting value, attention's semantic preservation, a measured economic news half-life, information diffusion, executable trading profit, or one confirmed cause for the disappointing results.
- **Unperformed:** systematic retained/discarded-story relevance measurement, separately retrained market-query ablation, original Reuters-data replication, and validated financial execution/evaluation.
- **Next direction:** direct LLM headline prediction, with matched information and explicit protection against future information in retrieval, prompts, and pretrained model knowledge. This remains a proposed experiment; do not treat literature discussed in the meeting as results produced here.

For an agent continuing the work: read [AGENTS.md](AGENTS.md), this record, and the relevant versioned protocol before modifying code. Use new experiment namespaces; do not rerun completed market studies, reopen reserved outcomes, or rewrite historical evidence as a documentation task.

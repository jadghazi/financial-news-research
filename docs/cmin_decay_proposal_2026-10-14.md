# Proposed bounded news-decay study — 2026-10-14

Status: design only; no decay models have been trained. Existing baseline artifacts remain unchanged.

Question: does an explicit relative-recency prior improve next-session company-direction probabilities compared with matched uniform temporal pooling in our CMIN adaptation?

## Connection to FININ

[FININ](https://aclanthology.org/2024.findings-emnlp.189.pdf), sections 4.2–4.3 (equations 6–7), learns within-day news interactions and market-conditioned weights, then sends multiple daily representations to a predictor. It does not specify an explicit age-decay function. Section 6.4 interprets longer-input results as delayed information diffusion and long memory. Our comparison is an independent extension, not a replication or a direct test of that economic mechanism.

Gradual information incorporation need not imply monotonically decreasing predictive usefulness. Older news might remain useful or show delayed effects. A fixed exponential curve is a modeling hypothesis, not a consequence established by information-diffusion theory. For context, [Lillo et al.'s author preprint](https://arxiv.org/pdf/1207.3300) distinguishes news, investor activity and persistence and cautions against causal interpretation of contemporaneous regressions; it does not establish a universal exponential decay law.

## Minimal comparison

- Reuse the pinned CMIN cohort, frozen headline/sentiment cache, selection cap, timing proxy, labels and chronological plans. Keep all five price sessions and the same eligible headlines in both arms.
- Train `pooled_uniform` and `pooled_decay` with a single predeclared 48-calendar-hour half-life. Every additional 48 hours halves a story's unnormalized recency multiplier. This is a design choice, not an estimated market half-life.
- Retain both FININ attention stages. Multiply the within-bucket attention weights by the recency factor and normalize over all active historical headlines. Uniform pooling gives equal total mass to each nonempty arrival bucket, not equal weight to every story.
- Normalization tests relative allocation among retained stories, not absolute fading of all news. A common increase in every story's age cancels; a lone same-aged bucket receives the same total news mass. The 48-hour clock is calendar time; diagnostic age bins below count arrival trading sessions.
- Both arms have the same pooled predictor geometry: 80 market values plus 16 news values. Do not compare decay pooling directly with the existing 160-value concatenation model as an isolated decay test. Fixed positive scaling of separate concatenated lag slots is absorbable by the MLP.
- Two seeds (17, 42), two existing development periods: at most eight fits; retain the existing 12-epoch optimization/validation rules. Proposed neural-training ceiling: 1,200 seconds, checked at epoch boundaries; retain partial failures without automatic retries. No half-life search or further dataset/model changes.
- Existing development outcomes have already been inspected. This study would be exploratory. Reserved late-2021 outcomes remain unopened; confirmation would need a separately frozen protocol and preparation audit.

## Measurements and report

Primary: paired evaluation log-loss difference, decay minus uniform. Keep the baseline practical-improvement criterion of 0.005, report period/seed effects and conditional date-block uncertainty; do not treat company-days or seeds as independent market histories. Report both raw neural forecasts and validation-selected constant fallbacks, with UP/DOWN counts and probability spread.

Retain training-prevalence probabilities and the existing price-only results as contextual references. The price-only concatenation model has different geometry; any incremental-news claim would need a matched price-only control. Decay beating uniform while failing the prevalence reference supports a relative modeling improvement, not a useful forecasting system.

Proposed plots:

1. Prescribed relative weight versus age, explicitly labeled an imposed curve.
2. Empirical age-sensitivity curve: for each selected pooled model, mask one arrival-session-age bucket at a time (ages 0–4), keep prices unchanged, and plot the paired change in evaluation log loss alongside changed-case/headline coverage and date-block uncertainty. Removal also renormalizes the remaining news. These are distribution-changing input perturbations without retraining, not isolated causal contributions or an estimated market diffusion curve. Report both model arms and periods; use the same dates and do not select favorable subgroups. Do not fit an economic exponential to this curve or choose the half-life from it.

Positive, negative and inconclusive outcomes are retained. Faster weighting helping may reflect regularization or suppression of stale noise. A stable older-news sensitivity pattern can motivate a later hypothesis but cannot identify delayed price incorporation. No improvement means no support for this particular recency prior in this setting, not a universal rejection of news value or diffusion.

## Timing limit

Availability is source date D + two calendar days at midnight UTC, then the first eligible exchange close. Decay ages currently use source-date UTC midnight; exact publication times, timezone and historical availability are unverified, with conflicting dates for some identical URLs. A common age offset cancels after exponential normalization, but varying timestamp errors do not. Arrival-age diagnostics measure age under our proxy. This study cannot recover the real publication-to-market-response curve or establish trading performance.

Source checks: FININ sections 2, 4.2–4.3, 6.4; `src/models/research_finin.py`; `src/cmin_panel_experiment.py`; `reports/cmin_public_pipeline_2026-10-14.md`. Bounded literature verification and independent design review use read-only conduct; runtime permissions do not enforce it.

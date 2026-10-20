# Frozen CMIN relative-recency experiment — 2026-10-14

The user authorized executing the bounded proposal on October 14, 2026. This protocol is saved before any new decay/uniform financial fit. The proposal and all prior artifacts remain preserved. Configuration: `configs/cmin_decay_48h_v1.json`; distinct output: `artifacts/cmin_decay_48h_2026-10-14_v1/`; audited data/cache: `artifacts/cmin_panel_2026-10-14_v1/`.

## Question and literature mapping

Does a fixed 48-calendar-hour relative-recency prior improve next-session company-direction probability forecasts compared with matched uniform temporal pooling? FININ sections 4.2–4.3, equations 6–7, use within-day interaction/query weighting and daily representations; section 6.4 interprets longer-input results as diffusion/long memory. FININ has no explicit exponential age-decay term. This is an independent extension, not empirical replication or proof of information diffusion. See the verified, independently reviewed `docs/cmin_decay_proposal_2026-10-14.md` for literature and design limits.

## Matched data, architecture and timing

Reuse the original pinned twelve-company cohort, five exchange-session price histories, same eligible hash-selected headlines and frozen RoBERTa/sentiment features. No new acquisition, encoding, label filtering, sample selection or unseen-period preparation. Source snapshots, source-date disagreement audit and existing purged train/validation/evaluation plans are retained. Training-only scalers must match the baseline exactly.

At exchange close d, use histories ending d to predict adjusted close q > adjusted close d for the immediately next exchange session q. Source-date D is provisionally eligible from D+2 calendar days at midnight UTC, entering its first eligible exchange-close bucket. Decay ages are cutoff minus source-date midnight in hours; exact publication timezone/time/availability remain unverified. Reserved late-2021 outcomes remain unopened.

`pooled_uniform`: retain both FININ attention mechanisms; equal total temporal mass per nonempty news bucket. `pooled_decay`: multiply the same within-bucket attention weights by 2^(-age_hours/48), then normalize over all active stories. Both arms retain the same 80 market values and one 16-value pooled news summary in the MLP. Same parameter geometry, seed initial state, masks, prices and optimization budget. Empty-news cases have zero pooled news. Normalization tests relative age allocation, not absolute disappearance of news influence. Common age shifts cancel; a single same-aged bucket retains total mass. Fixed scaling of separately concatenated day slots is not used.

## Fitting and budget

Two fixed chronological development periods from the baseline; two seeds 17/42; two arms: eight neural fits maximum. Twelve epochs, batch 64, AdamW learning rate 0.0003, weight decay 0.01, dropout 0.2, width 16, plain attention, gradient clip 1, four CPU threads. Select the minimum raw validation log-loss epoch; report the validation-selected prevalence fallback separately. Cumulative neural-training ceiling 1,200 seconds checked at epoch boundaries. Retain incomplete attempts and failures without automatic repeats. No half-life search or other tuning.

Both development evaluation periods were inspected in the prior baseline, so this is exploratory temporal validation. Each paired arm uses the same evaluation dates: May–August 2020 and January–April 2021. Preserve initial/selected/final states, completed epochs, predictions, source/environment identities and negative outcomes. Reuse the already fitted and verified fixed logistic references in the identical plans rather than fitting them again; record their source hashes.

## Analysis declared before fitting

Primary comparison: decay minus uniform mean raw seed-model evaluation log loss, not ensemble probabilities. Practical improvement criterion: decrease at least 0.005, inherited from the baseline. Report both period and seed effects, paired conditional 95% interval from 2,000 circular five-date blocks within each period, retaining stocks together. Intervals condition on fitted models and omit training uncertainty. Do not count stocks/seeds as independent market histories. Also report Brier, accuracy, probability spread, UP/DOWN counts, selected epochs, training/validation curves and fallback decisions.

Compare with training-prevalence probabilities, 50/50 probabilities and the saved price-only baseline as contextual references. Existing price-only has different 160-value geometry; do not claim an isolated incremental-news improvement from it. Improvement versus uniform without beating prevalence supports only relative modeling improvement. Separate selected-system behavior from raw neural sensitivity.

Diagnostic: selected raw checkpoints, evaluation only, no retraining. For each arrival-session-age bin 0–4, mask only that news bucket while leaving all prices and other stories unchanged. Recompute log loss/probabilities on all matched dates. Verify unperturbed inference reproduces stored evaluation logits within 1e-6. Retain per-case perturbed logits, changed-case/headline coverage, fitted-model/seed/period identity, and the paired conditional date-block interval for each mean seed loss difference. Do not select companies or regimes by the results. Empty-bin cases must remain unchanged. Compare periods/seeds rather than pooling away reversals. Curve removal renormalizes remaining news and changes input distributions: label it fixed-model loss sensitivity, not isolated causal contribution, true economic decay, estimated half-life or publication-time diffusion. No exponential fitting to this curve or selecting a new half-life from it.

Export a clearly labeled imposed weight curve separately from the empirical sensitivity curve, and a concise professor-facing report connecting the question, methods, outcome and limitations to FININ. Preserve machine-readable evidence. Independent skeptical review must directly inspect sources/artifacts and verify important positive or negative conclusions before finalization. Runtime permissions do not enforce read-only agents; require read-only conduct and disclose that limitation.

Stop after this comparison and its declared diagnostics. Positive, negative and inconclusive outcomes are all reportable; useful information cannot be promised before observing the evidence.

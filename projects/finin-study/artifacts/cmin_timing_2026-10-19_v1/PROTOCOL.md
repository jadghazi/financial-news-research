# Does the imposed news delay explain the missing forecasting benefit?

19 October 2026. User-authorized matched timing study. Single main agent; no delegation. All development periods and original results are already inspected; every new finding is exploratory. No reserved late-2021 outcomes or new markets.

## Fixed decisions before encoding or fitting

Reuse the pinned raw 12-company CMIN-US release, all9,984 development rows/labels, five actual exchange price sessions, chronological split/purge plans, company context, six-story hash cap per bucket, frozen RoBERTa/TinyBERT settings,160-input FININ, width16,12epochs, learning rate0.0003, AdamW weight decay0.01, dropout0.2, seeds17/42 and the two original periods. No hyperparameter/threshold search. Raw neural predictions are primary; validation-selected constant-fallback results are separate. Model losses are averaged across seeds, not probability ensembles.

Four timing policies:

1. Original `delay2`: source date D +2 calendar days, midnight UTC, assigned to first eligible exchange close. Reconstruct this from raw data and require exact agreement with original saved arrays. Reuse all four original full fits and four price-only fits after source/checkpoint/prediction verification.
2. `date0`: source-date midnight UTC, assigned to first eligible close. This is the paper-aligned daily-date formulation, not a certified market-close forecast. It can include source clocks after that close.
3. `delay1`: D+1 calendar day, midnight UTC, first eligible close. A less restrictive date-only scenario, not certified availability.
4. `clock_ny`: interpret naive source timestamps as America/New_York, convert to UTC and assign to first eligible exchange close. Post-close stories move to the next session. This is an assumed-timezone sensitivity, not verified historical availability. Also audit date0 cutoff violations under assumed UTC clocks without fitting another mode. Use actual exchange closes, including early closes and DST.

Parse/deduplicate raw titles identically before applying timing. Restrict source dates before2021-05-01. Keep source date/time/URL/proxy for every selected story. Selection is by the same content hash; moving bucket boundaries naturally changes cap selection and coverage, so this comparison estimates the timing-policy procedure, not the isolated causal effect of the same story's age. Measure overlaps, selected dates, coverage and clock-based cutoff violations to disclose this.

Preserve original cache indices and raw frozen values. Append missing selected titles to a new union cache, encoding only the additions with the original pinned offline models/max64tokens/batch16/no L2 normalization/first-token pooling. Verify a small reproduction sample against the original cache. Never edit the old cache or preparation functions that enforce the historical two-day rule.

Use the original training-only price/text transforms identically across timing policies. Their original source stories were available by each original training cutoff under the conservative rule. This fixed preprocessing isolates timing changes from refitting preprocessing and makes the original price-only comparator exactly reusable. Original item indices are preserved, including the ticker-description vectors. New timing policies may use additional train-available stories, but these do not refit shared transforms. This is a declared controlled choice, not an author FININ default.

## Primary assessment and budget

Exactly12 new full-FININ fits: three new policies x two periods x two seeds. Reuse originals; no new price-only or convex fits. Maximum1,800 neural seconds and600 additional-encoding seconds; retain failures and stop rather than sweep/retry. End after the frozen variants. Save initial/selected/final states, per-epoch validation evidence, per-case predictions, source/config/environment/data identities and commands.

Primary contrasts: each policy minus the original delayed full model, and each policy minus the same matched price-only model. Compare against training-prevalence probabilities as well. Use all2,004 matched evaluation company-days on167 dates, with2,000 paired circular five-date blocks within periods, retaining companies together. Report log loss, AUC, balanced accuracy, class counts, probability spread and seed-period effects. Intervals condition on saved fits and omit training uncertainty. Multiple timing contrasts are exploratory; none is a fresh confirmation.

A practically useful news contribution requires at least0.005 lower log loss versus price-only, interval below zero, both period means favoring news and at least three of four seed-period pairs favoring news. To support the delay as an explanatory factor, removing it must also improve versus the original news policy. Improvement only in date0, absent under the assumed-clock scenario, would raise cutoff/availability concerns rather than establish usable forecasting. If changes are tiny/inconsistent and news still fails against controls, the two-day delay is not demonstrated to be the principal explanation in this configuration. A null result does not prove timing irrelevant or dataset quality adequate.

## Verification and evidence

New artifacts: `artifacts/cmin_timing_2026-10-19_v1/`; new source/tests only. Hash all existing CMIN study files before work and verify preservation at completion. Reconstruct original arrays exactly; match row identities, prices, labels, old feature values, shared transforms and paired initial weights. Verify actual saved checkpoint selection, state/prediction hashes and score calculations. Include after-cutoff counts under both timezone assumptions and source-date conflicts; neither assumption repairs revised source vintages. Main-agent direct review only; no claim of independent-agent review or causal identification.

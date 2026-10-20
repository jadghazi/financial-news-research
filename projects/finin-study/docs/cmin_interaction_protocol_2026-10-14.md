# Headline interaction comparison for the meeting report

Date: 14 October 2026. Authorized by the user's request to investigate headline interactions and produce a concise progress report. Single main agent; no delegation.

## Scope and hypotheses

Exploratory: these two development periods and the baseline results were already inspected. A first read of saved traces found high post-attention directional similarity in three of four selected networks. This is a hypothesis-generating observation, not a preregistered discovery. Test whether retaining headline self-attention improves next-session probability quality compared with bypassing it while keeping market-query weighting. Separately inspect saved headline mixing and title-level relevance; neither cosine similarity nor a forecast gain establishes semantic preservation or causality.

## Fixed comparison

Reuse the pinned 12-company CMIN cohort and frozen feature cache from `artifacts/cmin_panel_2026-10-14_v1/`. Preserve five actual price sessions, up to six headlines per proxy-arrival bucket, date-only D+2 midnight UTC eligibility, next-session adjusted-close direction, exact chronological plans, training-only transforms and label-availability purging. No reserved late-2021 outcomes.

Train only `no_interaction`: two periods, seeds 17/42, 12 epochs, width16, dropout0.2, plain attention, AdamW learning rate0.0003 and weight decay0.01. The final MLP remains160 inputs. Self-attention parameters remain allocated but unused; active capacity differs. Query weighting, sentiment, price information and predictor remain. Exactly four new neural fits, maximum600 neural seconds, no tuning/retries or new convex fits. Copy the two verified existing logistic references solely to prevent the generic runner from refitting them.

Reuse every corresponding selected full-FININ fit from the baseline, without outcome selection. Verify exact initial-state equality, plans/scalers and dated labels, saved checkpoints and summaries. Report raw neural and validation-selected constant-fallback systems separately. The primary contrast is no-interaction minus full-FININ log loss, averaging seed losses rather than ensembling probabilities, on2,004 matched company-days/167 dates. Use2,000 paired circular five-date blocks within periods, retaining companies together. These intervals condition on saved fits and omit training uncertainty. Secondary contrasts: no-interaction versus matched baseline prices and the training-prevalence reference. Practical improvement criterion remains0.005, with consistency across period means and at least three of four seed-period pairs; no search for favorable subgroups or settings.

## Saved-tensor inspection

Inspect all four existing full-model traces, selected by the original runner as the first evaluation observation per company without consulting outcomes. For every bucket with at least two headlines, record mean pairwise cosine similarity before and after interaction, normalized self/query entropy and vector-length variation. Summarize medians over buckets per checkpoint. The traces cover two snapshot dates, not all evaluation dates. Reconstruct the plain self-attention output from saved weights and checkpoint value/output matrices; check masks, weight sums, daily summaries and final MLP concatenation. Annotate only the six AAPL titles in the saved April30 example as direct Apple mention/broad or other topic, before relating them to stored weights. This is a title-level descriptive audit, not an assessment of full articles, real financial relevance or discarded headlines. Treat weights as weights on contextual mixtures.

## Evidence and stop rule

New artifacts only in `artifacts/cmin_interaction_2026-10-14_v1/` and `artifacts/diagnostics/cmin_interactions_meeting_2026-10-14_v1/`; preserve all prior studies. Save configuration, protocol, code/environment/data identities, predictions, initial/selected/final states, histories, failures and analysis. Retain negative results. After four fits, verify and report; do not extend the search. Root-cause conclusions must distinguish measured blending/generalization behavior from unverified timing/relevance/representation explanations.

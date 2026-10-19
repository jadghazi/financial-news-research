# Headline interactions: saved-trace inspection and matched comparison

14 October 2026. Single main agent; no subagents. Exploratory investigation requested for the professor meeting report. Earlier development results and a first trace inspection were already visible; this is not a fresh confirmatory test.

## What was done

Inspected all four selected CMIN baseline full-model traces. The original runner selected the first evaluation observation per company without consulting outcomes: two snapshot dates, 30 April 2020 and 31 December 2020. Each model has34 buckets with at least two active headlines. The main agent verified masks/normalized weights, reconstructed the plain attention output from checkpoint value/output matrices (maximum discrepancy1.43e-6), daily weighted summaries and exact160-input MLP concatenation. No encoder or model inference was rerun for this inspection.

Median within-bucket mean pairwise cosine similarity:

| Period / seed | Before self-attention | After | After vector-length CV |
|---|---:|---:|---:|
| 2020 /17 |0.418302|0.999795|0.099380|
| 2020 /42 |0.530305|0.999706|0.008413|
| 2021 /17 |0.216291|0.366475|0.466094|
| 2021 /42 |0.834381|1.000000|0.000017|

Vectors become nearly identical in direction in three traces, raising a blending concern. This does not measure semantic preservation: magnitudes can retain information, representations are in different learned spaces, and one model behaves differently. The sample is two snapshot dates, not the entire dataset.

The saved AAPL example's six titles combine direct Apple mention with market-wide, chip-industry and other-company topics. The SAP/Deutsche Telekom title's contextual-vector weight is20.2%, versus15.6% for the Apple earnings title. These weights apply after mixing; interpreting them as independent raw-story causal importance is incorrect. Title-only topical annotation does not certify actual financial relevance. No full article bodies or discarded headlines were assessed.

## New controlled fitting

Four `no_interaction` fits completed in267.641 neural seconds: two periods, seeds17/42, same five-session information, chronological plans, scalers, initialization,160-input geometry and training budget as the retained full-model reference. Market-query attention remains. Seven existing model-integrity tests passed. No new acquisition/encoding/tuning or convex fitting; all655 baseline files were freshly verified unchanged. Self-attention parameters remain allocated but unused, so active capacity differs.

2,004 matched company-days/167 dates; mean raw seed losses, not an ensemble:

| Model | Log loss |
|---|---:|
| Full joint attention |0.690388|
| No headline interaction; query retained |0.693387|
| Existing matched baseline prices |0.690095|
| Global training-prevalence probabilities |0.689820|

No-interaction minus full loss is+0.002999, conditional95% interval[+0.001058,+0.005092]. Two thousand paired circular five-date blocks within periods keep companies together; uncertainty conditions on fitted models and omits training variability. Period means reverse sign: removal worsens loss by+0.006374 in2020 and improves it by-0.000499 in2021. Seed-period differences are+0.004563,+0.008184,-0.001238,+0.000240. The pooled decrease from retaining interactions is below the predeclared practical criterion0.005 and is not consistent across periods.

All four no-interaction fits selected neural forecasts rather than constant fallback. One original full fit used a validation-selected fallback; raw-network and selected-system contrasts are separately retained. The system contrast is+0.002997,[+0.001057,+0.005088], essentially the same.

## Supported conclusion

The pooled comparison favors retaining interactions over this bypass under the tested budget, but does not establish repeatable useful forecasting or semantic relevance preservation. Full still trails the matched price and prevalence references. Removing interactions did not rescue performance overall. Near-identical directions are an implementation diagnostic, not a demonstrated dominant root cause. Timing, relevance, noise, representation and generalization remain competing explanations.

Protocol: `docs/cmin_interaction_protocol_2026-10-14.md`. Saved tensors/title annotation/hashes: `artifacts/diagnostics/cmin_interactions_meeting_2026-10-14_v1/`. New predictions/checkpoints/history/configuration/source snapshots: `artifacts/cmin_interaction_2026-10-14_v1/`; `interaction_analysis.json` includes exact comparisons, fitted references, four verified initial-state pairs and655 preserved-file checks. No reserved-period evaluation occurred. Main-agent verification used actual saved artifacts; no claim of independent-agent review.

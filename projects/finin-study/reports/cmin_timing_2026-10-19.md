# Does newer news fix our forecasting results?

19 October 2026 · Completed CMIN-US timing comparison

**Removing our two-calendar-day delay did not improve the average result. None of the three newer-news policies beat the original news model or prices-only in average log loss. This experiment does not support the delay as the main explanation for the missing forecasting benefit. It does not establish the underlying cause.**

## What changed

We predict whether a company's next exchange-session adjusted close will exceed its current close. Prices are known through the current close. [FININ](https://aclanthology.org/2024.findings-emnlp.189/) describes day-d inputs predicting the next session (§3.1 and Eq. 7), but does not specify the exact intraday news cutoff or timezone. Our previous two-day availability rule was an independent assumption.

We tested three alternatives against that original rule:

| News policy | What enters the latest bucket for a Wednesday-close forecast of Thursday? |
|---|---|
| Original: source date + two calendar days | Monday-dated stories |
| Source date + one calendar day | Tuesday-dated stories |
| Same-date news | Wednesday-dated stories, including possible after-close stories |
| Clock-based scenario | Stories timestamped before Wednesday's close; later stories move to the next session |

The clock-based scenario **assumes the source clocks are New York time**. Their true timezone and original publication availability remain unverified. The Wednesday example was checked against the selected AAPL stories for 26 August 2020: the original newest bucket has six Monday stories; the same-date bucket has six Wednesday stories; the clock-based bucket has two Tuesday and four Wednesday stories.

All policies retain the same 12 companies, rows, targets, five price sessions, six-headline bucket cap, frozen encoders, 160-input model, training-only transforms, training budget and checkpoint-selection rule. This tests news availability, separately from the earlier 48-hour importance-decay study. Exactly **12 new fits** cover three policies, two chronological periods and two seeds. Original news and prices-only fits were reused after integrity checks. Evaluation covers **2,004 company-days on 167 dates**, in May–August 2020 and January–April 2021. These dates were already inspected, so this is exploratory evidence.

## Results

Lower log loss is better. Values average individual seed-model losses; they are not ensemble forecasts. “Prices-only” includes fixed company identity but no headlines or sentiment.

| Model / news policy | Log loss | Accuracy |
|---|---:|---:|
| Training-prevalence probability reference | 0.689820 | 54.39% |
| Prices-only | 0.690095 | 54.54% |
| Original two-day rule | 0.690388 | 54.07% |
| One-day rule | 0.690410 | 54.32% |
| Same-date news | 0.690502 | 54.04% |
| Clock-based scenario | 0.690459 | 53.84% |

The newer-news models differ from the original by only **+0.000022 to +0.000114** in loss: tiny changes, with slightly worse point estimates. They do not demonstrate useful improvement.

| New policy | Loss difference versus original [conditional 95% interval] | Loss difference versus prices-only [conditional 95% interval] |
|---|---:|---:|
| One-day rule | +0.000022 [−0.001385, +0.001428] | +0.000315 [−0.000102, +0.000745] |
| Same-date news | +0.000114 [−0.001839, +0.002191] | +0.000407 [−0.001388, +0.002245] |
| Clock-based scenario | +0.000071 [−0.001247, +0.001319] | +0.000364 [−0.000234, +0.000939] |

Negative differences would favor the new policy. Every interval includes zero. Intervals use 2,000 paired circular five-date block samples within each period, keeping companies together. They condition on these fits and omit training uncertainty; multiple exploratory comparisons are not independent confirmations. No policy meets the predeclared requirement of at least 0.005 improvement over prices, an interval below zero, favorable means in both periods and at least three favorable seed-period pairs. Same-date news loses to prices in both period means; one-day and clock-based news have opposite signs across periods. The latter two favor news in only two of four seed-period pairs.

## Did the experiment really make news more recent?

Yes. The median age of the freshest selected source date, among cases with news, falls from **69 hours to 20 hours** under same-date or clock-based selection, measured from source-date midnight UTC. This is an age proxy, not recovered publication time. Mean headline count rises from 13.85 to about 15 per forecast. Selection changes substantially: among nonempty new-policy cases, mean headline-set Jaccard overlap with the original is approximately 43% for same-date news and 48% for clock-based news. This compares timing-and-selection procedures; it is not a causal comparison holding every individual story fixed.

Same-date selection contains at least one timestamp after the forecast close in **627/2,004 cases (31.3%) if clocks are UTC**, or **1,097/2,004 (54.7%) if they are New York time**. Clock-based selection has no such violations under either of those assumptions, but this does not certify historical availability. Even the permissive same-date scenario fails to improve the average result.

## What is the working diagnosis?

**The measured problem is weak generalization, not a demonstrated failure to admit recent news.** In all 12 new fits, training loss falls between epochs 1 and 12; in 10, validation loss ends higher. For same-date news, mean training loss falls by 0.01980 while mean validation loss rises by 0.00995. These are post-result descriptive checks, consistent with overfitting. Reported evaluation predictions still use the best validation-selected checkpoints, not the final epochs.

We cannot yet attribute this to poor data, the frozen representations, headline selection, or information already reflected in closing prices. Those are competing explanations. The timing intervention did not resolve the observed problem. A useful next discrimination is a simple, matched news-summary model under the same cutoff, separating the available news signal from the attention architecture. Establishing the actual timestamp semantics remains necessary before claiming historical forecast availability.

For the meeting: **“We tested fresher news under three timing rules. It did not improve forecasting. The model fits training data better, but that learning does not carry over reliably to later dates. We have not identified a single underlying cause.”**

## Verification and evidence

The original delayed arrays were reconstructed exactly. Prices, targets, shared transforms, original cached vectors and paired starting weights were preserved. Eight original texts were re-encoded with the same pinned models; text differences were at most 0.00000191 and sentiment matched exactly. All 16 selected old/new news-model checkpoints reproduced their complete saved evaluation logits exactly. Scores, row identities, checkpoint choices and saved artifact hashes were checked. All **1,763 protected prior experiment files** remain unchanged. The 24 relevant tests passed. Additional feature encoding took 217 seconds and 12 new financial fits took 889 seconds. Reserved late-2021 outcomes were not used.

Protocol: [frozen design](../docs/cmin_timing_protocol_2026-10-19.md). Evidence: [analysis](../artifacts/cmin_timing_2026-10-19_v1/analysis.json), [checkpoint replay](../artifacts/cmin_timing_2026-10-19_v1/checkpoint_replay.json), [learning diagnostic](../artifacts/cmin_timing_2026-10-19_v1/learning_diagnostic.json), [worked timing example](../artifacts/cmin_timing_2026-10-19_v1/timing_example.json). Full configurations, source snapshots, cache provenance, epoch histories and predictions are retained in `artifacts/cmin_timing_2026-10-19_v1/`.

Executed stages, using `.venv/Scripts/python.exe`: `scripts/27_cmin_timing.py prepare`, `encode`, `preflight`, `run`, `analyze`; `scripts/28_verify_cmin_timing.py cache`, `checkpoints`. No tuning or additional fits followed the declared comparison.

# Explicit news decay: bounded CMIN experiment — 2026-10-14

**No stable, practically meaningful forecasting benefit from the fixed 48-hour decay was established.** This is a completed exploratory experiment, not a validated market-diffusion finding. No follow-up tuning was performed.

We executed the authorized comparison with the pinned CMIN data/cache: five price sessions, the same eligible headlines and sentiment, both FININ attention stages, and the same 96-value pooled predictor geometry (17,825 parameters). Only relative temporal weighting differed. Equal weighting gives equal temporal mass to nonempty news buckets; decay multiplies within-bucket attention weights by 2^(-source-date-proxy age hours / 48), then normalizes. The half-life was chosen before fitting, not estimated from outcomes.

Exactly eight neural fits (two arms × two seeds × two chronological periods) completed in 573.8 seconds, under the 1,200-second ceiling. No new encoding/downloads or convex fits; previously fitted logistic references were copied with verified hashes. Seven relevant existing model tests passed. Both arms started with identical paired weights, prices, splits and training-only transforms. All 655 protected baseline files remained unchanged.

## Forecast results

2,004 matched company-days on 167 dates. Losses average raw seed-model losses, not ensemble probabilities. Lower log loss is better.

| Model/reference | Log loss | Accuracy |
|---|---:|---:|
| Equal temporal weighting | 0.690210 | 54.27% |
| 48-hour decay | 0.689887 | 54.64% |
| Training-prevalence probabilities | 0.689820 | 54.39% |
| Existing price-only (different geometry; context only) | 0.690095 | 54.54% |

Decay-minus-uniform log loss: **-0.000324**, conditional 95% interval **[-0.001176, +0.000539]**. The predeclared practical decrease of 0.005 was unmet. Decay also failed to beat the prevalence reference on average log loss. Intervals use 2,000 paired circular five-date blocks within each period, retaining stocks together; they omit training uncertainty.

| Evaluation period | Seed | Decay-minus-uniform loss |
|---|---:|---:|
| May–August 2020 | 17 | +0.001125 |
| May–August 2020 | 42 | -0.000779 |
| January–April 2021 | 17 | -0.000780 |
| January–April 2021 | 42 | -0.000898 |

Both 2021-period decay networks predicted **UP on all 984 cases**; their probability standard deviations were 0.00401 and 0.00094. Neither uniform model was all-UP in that period (902 UP / 82 DOWN each). Validation selected the neural network rather than the constant fallback in all eight fits. A small accuracy increase does not establish useful forecasts.

## Measured age sensitivity

For each selected raw model, we separately removed each arrival-session-age bucket 0–4 while retaining all price inputs, with no retraining. Fresh unperturbed inference reproduced saved logits within 1e-6. Empty-bucket cases remained unchanged. Per-case logits, headline coverage, temporal masses and conditional pointwise intervals were retained.

Effects were tiny and non-monotone. Nine of ten pooled pointwise age intervals include zero. The one exception—equal-weighting age-3 removal—reduced loss by about 0.000016, largely driven by the first-period seed-17 fit. Across ten correlated diagnostic comparisons, this does not support a robust decay law. Each age bucket contained removable news in about 66.6–67.1% of matched cases. Removing news also renormalizes remaining weights and changes input distributions; these are fixed-model loss sensitivities, not independent causal contributions.

## What can be presented

> We tested an explicit recency prior in a FININ-inspired model while holding information and architecture fixed. A 48-hour half-life produced only a tiny, uncertain average improvement over uniform temporal pooling and did not beat constant prevalence probabilities. Age-removal diagnostics did not show a reliable monotonic pattern. We therefore found no support for this particular decay prior as a useful forecasting improvement in our public-data setting.

[FININ](https://aclanthology.org/2024.findings-emnlp.189.pdf), sections 4.2–4.3 (Eqs. 6–7), uses within-day interactions/query weights and multiple daily representations, with no explicit decay function. Section 6.4 interprets longer-input results as delayed diffusion and long memory. This study is an independent extension; it does not replicate or refute those economic claims. Gradual diffusion need not imply monotonically decreasing predictive usefulness.

Existing development outcomes were already inspected, so this is exploratory. Source dates/timezones and D+2 availability remain unverified proxies; conflicting source dates persist. Normalized exponential weights cancel common age offsets and test relative recency rather than absolute news fading. Reserved late-2021 outcomes remain unopened. No market half-life, causality or trading performance is established.

Evidence: [frozen protocol](../docs/cmin_decay_protocol_2026-10-14.md), [configuration](../configs/cmin_decay_48h_v1.json), `artifacts/cmin_decay_48h_2026-10-14_v1/` and its `analysis/` directory. The two preparation-wrapper representation errors were recorded before fitting and affected no financial fit or data. Independent final review freshly verified 1,055 hashes, four paired initial states, eight checkpoint selections/state matches, matched predictions and all ten sensitivity differences/intervals from saved logits. It found no material computation or reporting defect and supported the stated limits. Reviewer conduct was read-only; runtime permissions did not enforce it. The completed review is retained as `FINAL_REVIEW.md` in the study.

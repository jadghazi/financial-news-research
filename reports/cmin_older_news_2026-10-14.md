# Does older available news add forecasting value? — 2026-10-14

**No robust, practically useful older-news addition was established in this fixed predictive setup.**

We compared three otherwise matched 96-input pooled models: prices plus fixed company identity; the same prices plus current-arrival-bucket news; and the same prices plus current and four older arrival buckets. All retain five price sessions. Both news models use the same FININ-style headline/sentiment fusion, self-attention, market-query weighting and uniform temporal pooling. The no-news block is zero; its news parameters are unused, so total parameter equality does not imply equal active capacity.

Only eight new fits were run, taking 584.6 seconds under the 1,200-second ceiling. All four matching all-news fits from the preceding decay study were reused, without choosing favorable seeds or retraining. Identical initial states, hyperparameters, purged plans and training-only transforms were checked. No new acquisition, encoding or convex fitting; no tuning or automatic retries.

## Matched forecast results

2,004 company-days on 167 dates; two chronological periods and seeds 17/42. Lower log loss is better. Values average raw seed-model losses, not ensemble probabilities.

| Inputs | Log loss | Accuracy |
|---|---:|---:|
| Prices (matched pooled geometry) | 0.689771 | 54.59% |
| Prices + current arrival bucket | 0.689651 | 54.27% |
| Prices + current and four older buckets | 0.690210 | 54.27% |
| Constant training-prevalence probabilities | 0.689820 | 54.39% |

**Adding older news: log-loss difference +0.000560, conditional 95% interval [-0.000031, +0.001155].** The predeclared practical decrease was 0.005. A useful relative addition also required improvement in both period means and at least three of four seed-period pairs, with the interval below zero.

| Evaluation period | Seed | All-news minus current-news loss |
|---|---:|---:|
| May–August 2020 | 17 | -0.000171 |
| May–August 2020 | 42 | +0.001945 |
| January–April 2021 | 17 | -0.000043 |
| January–April 2021 | 42 | +0.000483 |

The interval uses 2,000 circular five-date blocks within periods, keeping stocks together. It conditions on saved fits and omits training uncertainty. Raw neural and validation-selected constant-fallback systems are retained separately in the analysis.

## Coverage checks declared before fitting

| Available-news group | Company-days | Dates | All-news minus current-news loss | Conditional 95% interval |
|---|---:|---:|---:|---|
| Both current and older | 1311 | 166 | +0.000922 | [+0.000039, +0.001789] |
| Older only | 402 | 144 | -0.000338 | [-0.001275, +0.000503] |
| Current only | 33 | 30 | -0.000712 | [-0.002217, +0.000840] |
| No available news | 258 | 167 | +0.000283 | [-0.000535, +0.001160] |

These groups use masks alone, not outcomes, and are descriptive. Their reported intervals use the full period trading-date calendars, with zero sums/counts on dates without group cases, as specified in a pre-group-results uncertainty amendment. Original five-observed-group-date intervals remain preserved as a descriptive sensitivity. Adding older buckets changes coverage, sentiment, pooling shares and fitted parameters. Group composition differs; comparisons do not isolate older headline semantics or a causal news effect. The current-only group has just 33 cases and is particularly uncertain.

Even on empty-news cases, the separately trained models can predict differently because they learned different shared parameters; this does not imply hidden news inputs on those rows.

## Class behavior

| Period | Inputs | Seed | UP / cases | Selected epoch | Constant fallback |
|---|---|---:|---:|---:|---|
| 2020 | price_only | 17 | 935 / 1020 | 11 | False |
| 2020 | price_only | 42 | 905 / 1020 | 9 | False |
| 2020 | recent_news | 17 | 1015 / 1020 | 10 | False |
| 2020 | recent_news | 42 | 936 / 1020 | 9 | False |
| 2020 | all_news | 17 | 935 / 1020 | 8 | False |
| 2020 | all_news | 42 | 892 / 1020 | 9 | False |
| 2021 | price_only | 17 | 984 / 984 | 4 | False |
| 2021 | price_only | 42 | 902 / 984 | 5 | False |
| 2021 | recent_news | 17 | 902 / 984 | 5 | False |
| 2021 | recent_news | 42 | 902 / 984 | 5 | False |
| 2021 | all_news | 17 | 902 / 984 | 4 | False |
| 2021 | all_news | 42 | 902 / 984 | 5 | False |

## Interpretation and stopping decision

No robust, practically useful older-news addition was established in this fixed predictive setup.

An improvement over a weak current-news model alone is insufficient to establish useful forecasting. All-news versus the newly matched price-only model and constant probabilities are secondary references; complete contrasts, Brier/AUC/MCC, probability spread and selected-system results are retained in the analysis. No favorable subgroup, additional seed or model was pursued.

The all-news model also had higher loss than matched price-only by +0.000439, conditional interval [+0.000061, +0.000814], a small secondary difference below the practical threshold. Current-bucket news improved on prices by only -0.000121 (interval [-0.000632, +0.000378]) and on constant probabilities by -0.000169 (interval [-0.001374, +0.001148]). Those intervals include zero and both contrasts reverse sign between periods; its lowest point-estimate loss does not establish a genuine forecasting gain.

"Current/newest" means the cutoff session’s proxy-arrival bucket, including empty buckets, not the latest nonempty bucket or verified newly published news. "Older" means 1–4 trading sessions since proxy arrival. Source-date D+2 availability/timezones and publication dates remain unverified. Development periods were previously inspected, so the entire comparison is exploratory; reserved late-2021 outcomes remain unopened.

[FININ](https://aclanthology.org/2024.findings-emnlp.189.pdf), sections 4.2–4.3 and Eq. 7, uses historical daily news representations. This experiment tests an independent pooled public-data adaptation and a narrower predictive question. It does not estimate publication-time market response, demonstrate information diffusion or identify an economic half-life.

Evidence: [frozen protocol](../docs/cmin_older_news_protocol_2026-10-14.md), [configuration](../configs/cmin_older_news_v1.json), `artifacts/cmin_older_news_2026-10-14_v1/analysis/summary.json`, paired per-case logits and original/new checkpoints. All 1,085 protected prior-study files were verified unchanged. Independent final audit verified 1,513 hashes, 12 actual paired initial-state matches, executed source/configuration/plan/scaler integrity, checkpoint selection, copied references and all primary/secondary and full-calendar subgroup contrasts. It found no material implementation or comparison defect and supported the reported negative conclusion. Review is retained as `FINAL_REVIEW.md`; conduct was read-only, though runtime permissions did not enforce it.

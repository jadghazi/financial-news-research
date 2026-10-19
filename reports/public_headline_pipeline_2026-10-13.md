# FININ public-headline pipeline: what is working?

The rebuilt pipeline is computationally functional and independently audited. It is suitable for controlled comparisons **conditional on the declared news-availability assumptions**. It has **not established a reliably useful forecasting model**.

We used the pinned [SPNews author release](https://github.com/FreddieNIU/Financial-Graph-Evaluation/tree/d19aa5e81cd1768d8b0958cc2a1ed505ba1e5b0a), accompanying [Niu et al. (2024)](https://arxiv.org/abs/2410.07216). Among the inspected public candidates, real financial headlines, company links, S&P 500 prices and 2022–23 coverage fit the intended input type better than StockNet tweets. This is a public-data adaptation, not Reuters/TRNA empirical replication.

**Executed:** 419 verified source files; 96,280 deduplicated first-version stories; 321 forecast observations; 17,663 selected headline texts plus one description encoded with the actual pinned frozen RoBERTa-base, and frozen finance sentiment. No language-model fine-tuning occurred; the projection, fusion, attention and prediction layers were trained. Historical experiments were retained.

Three chronological development blocks used 66/147/190 training observations, 19/21/19 selection observations and 42/41/43 evaluation observations. All arms used the same three-day inputs and eligible dates, training-only transforms, label-availability purges and fixed budgets. We completed **36 neural fits** with initial/selected/final states and learning curves, plus **nine convex fits** with fitted parameters and convergence records; all predictions were retained. The calendar check found no missing or extra sessions among 399 pre-2024 source dates against the [NYSE holiday schedule](https://ir.theice.com/press/news-details/2021/NYSE-Group-Announces-2022-2023-and-2024-Holiday-and-Early-Closings-Calendar/default.aspx). No 2024 numerical prices, labels or forecasts were evaluated.

## Is learning happening?

Yes. Training-only diagnostics learned both 16 balanced financial observations and a fabricated headline-dependent rule with identical non-text channels: 100% accuracy in each. These prove computational learnability, not forecasting or semantic understanding. In the actual financial fits, full-model final training loss fell to 0.0024–0.1941, while later-period performance remained weak: substantial fitting and overfitting are present.

All first-block full models predicted DOWN throughout. Later blocks predicted both directions; pooled full-model down counts were 103/92/82 of 126 dates. A retrospective selected-checkpoint audit replaced text with its training mean while retaining prices, sentiment and masks: it changed 102 decisions across 378 correlated seed-date outputs. Headlines affect the selected forecasts; this potentially out-of-distribution probe does not show that they help.

## Forecasting results

Neural entries average three raw seed models, not an ensemble. Lower log loss is better; there are 126 unique evaluation dates.

| Comparator | Log loss | Accuracy |
|---|---:|---:|
| Full FININ | 0.7539 | 47.4% |
| Mean news pooling | 0.7066 | 44.4% |
| Matched no-news | 0.7068 | 42.1% |
| Price-history MLP | 0.7339 | 42.1% |
| Linear prices | 0.7715 | 45.2% |
| Linear prices + coverage | 1.2738 | 47.6% |
| Linear prices + coverage + mean news | 3.4076 | 50.8% |
| Training-prevalence probabilities | 0.7060 | 47.6% |

Always-UP accuracy was 52.4%. An additional fixed 50/50 probability reference has log loss 0.6931; this is a post-run descriptive reference with no fitting or selection.

Full-minus-no-news loss was **+0.0470**, conditional five-date-block 95% interval **[+0.0088,+0.0831]**. Complete three-day collection histories retained 114 dates and gave **+0.0512 [+0.0085,+0.0931]**. Period differences were −0.0059/+0.0072/+0.1366. One full seed improved on its matched control, two worsened; an overconfident final-period run drives much of the aggregate harm. Full improves some relative accuracy/AUC indicators but fails the stronger overall forecast comparisons. Poor linear references do not validate FININ; their failure needs separate diagnosis.

**Supported conclusion:** no repeatable overall news benefit under this configuration. The intervals condition on saved fits, omit training/initialization uncertainty and do not establish universal news or attention failure.

## Research readiness and limits

The conservative proxy yields median headline publication age 76.75 hours. Its snapshot-date semantics and historical price/publication vintages remain unverified. Coverage-selected development windows, small selection blocks, generic sentiment and unknown historical constituents also limit claims. Fresh-news and delayed-news mechanisms cannot be equated.

Independent review recalculated 2,331 metric sets, 333 fit hashes, all checkpoint selections and the paired intervals, finding no implementation blocker. Reviewer conduct was read-only; runtime permissions did not enforce it.

The harness can now test interaction and query weighting separately, matched history changes and explicit decay with retained price histories. These remain conditional research tests, not evidence of a validated forecasting pipeline. Any regularization changes need a separately frozen study; do not retune these inspected evaluation periods or open the reserved period before the intended comparisons are fixed.

Evidence: `artifacts/rebuild/finin_headline_screen_2026-10-13/` (protocol, amendments, configurations, predictions, checkpoints and analysis); `spnews_learning_gate_2026-10-13/`; `spnews_selected_news_audit_2026-10-13/`; `spnews_cohort_2026-10-13/` (source provenance, acquisition/preparation logs, calendar audit and exact launch scripts).

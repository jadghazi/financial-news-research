# Public FININ-inspired pipeline: implementation and baseline

**The pipeline is implemented, exercised and independently audited for controlled research. This bounded study did not demonstrate a meaningful, repeatable forecasting benefit from news or joint attention.** No further fitting or tuning was performed after the declared study. This is an independent company-panel adaptation, with conditional date-based news availability, rather than a FININ/CMIN empirical reproduction.

The goal remains the professor's three questions: headline interactions, explicit temporal weighting, and contributions to the final predictor. [Architecture and controls](../docs/cmin_pipeline.md), [frozen protocol](../docs/cmin_panel_protocol_2026-10-14.md), [commands](../docs/cmin_runbook.md), and [worked example](../artifacts/cmin_panel_2026-10-14_v1/walkthrough.md).

## What was built and executed

The pinned [CMIN-US author release](https://github.com/BigRoddy/CMIN-Dataset/tree/3f571a53b81d8f3ea7b3afb9877bdbdbeae54c97/CMIN-US) supplies raw individual headlines and prices. Twelve companies were selected by a frozen hash rule before outcomes were examined. We acquired and verified 24 source files, totaling 158 MB, without downloading the entire repository. All companies have complete development price calendars. Eleven have news during that period; DIS has none and remains in the fixed cohort.

The dataset contains **9,984 eligible company-days**. Features use five actual exchange sessions and up to six headlines per arrival bucket. The target is next-session adjusted-close direction; no filtering by future return size. Price features and labels were independently reconstructed from raw files. Raw summaries and processed same-row return fields are not model inputs/targets.

**29,937 texts** were encoded once using actual pinned frozen RoBERTa and finance TinyBERT sentiment, offline. A shared model across companies retains FININ's shared text projection, separate numerical/fusion encoders, headline self-attention, market-query weighting and final MLP. Width is 16, with **18,849 total parameters**. The default final MLP input has 160 values: five market representations and five weighted news summaries. Sentiment enters earlier in headline fusion; weights shape summaries. These are declared departures from the original instrument-sentiment/index experiment.

Two chronological development periods, three trainable arms and two seeds produced exactly **12 financial neural fits**, plus two fixed logistic price/company-identity references. Neural fitting took **811 seconds**. All 107 repository tests passed; those are engineering checks, including synthetic tests, not forecasting evidence. Initial, selected and final checkpoints, completed-epoch evidence, dated predictions, example traces, configurations, source snapshots and provenance were retained. Completed-run verification returned the same result without fitting again.

## Forecasting evidence

There are **2,004 matched evaluation company-days on 167 distinct dates**. Neural values average raw seed-model losses, not ensemble probabilities. Lower log loss is better.

| Model/reference | Log loss |
|---|---:|
| Training-prevalence probabilities | 0.689820 |
| Prices plus fixed company identity | 0.690095 |
| FININ joint attention | 0.690388 |
| Prices plus mean headlines | 0.690569 |
| Fixed 50/50 probabilities | 0.693147 |
| Linear prices plus company identity | 0.695669 |

FININ-minus-price loss is **+0.000293**, with conditional 95% interval **[−0.000951, +0.001561]**. Both evaluation periods favor prices: +0.000158 and +0.000433. FININ-minus-mean loss is −0.000181, interval [−0.001525, +0.001120], with opposite signs across periods. The predeclared 0.005 improvement criterion is unmet. Intervals use paired five-date blocks within each period, retaining companies together; they condition on saved fits and omit training uncertainty.

FININ mean-seed accuracy is **54.07%**, versus **54.39% always UP**. It is not universally a constant predictor: three of four selected FININ networks predict both directions; the fourth predicts only UP and selects a constant fallback on validation. Raw network and selected-system outputs are reported separately. Some later news training losses improve while validation losses worsen, and selection often favors earlier epochs. Learning and some overfitting are present; additional data did not automatically make news useful. The weaker linear reference does not validate FININ.

## Research controls now available

| Professor question | Implemented comparison |
|---|---|
| Headline relationships and attention | Remove interactions or market-query weighting independently; retain other inputs and compare with uniform pooling |
| Explicit importance decay | Match pooled uniform/decay models, or restrict news-age bins while preserving the full price history |
| What enters the final MLP | Remove semantic headlines, sentiment or the direct market block separately; distinguish direct market removal from removing prices throughout the model |

These controls and their invariants were checked, but the separate professor-task experiments were **not automatically run**. Later explicit studies can reuse the cache in a new output folder with their own frozen protocol. Inspected development periods are exploratory evidence for those later studies. Late-2021 target outcomes and model evaluations remain reserved from this study's selection; raw source files include that publicly studied period.

## Limits and verification

Source times lack verified timezone/availability vintages. News dated D is provisionally eligible from D+2 midnight UTC. Fourteen identical-URL groups have differing source dates, and thirteen have differing arrival sessions. Updating or date disagreement remains unresolved. A future decay curve can measure **conditional predictive usefulness by proxy age**, not recovered publication-time diffusion. Retrospective large-cap selection, adjusted-price vintages, pretrained-corpus overlap, generic sentiment, uneven coverage and correlated company-days remain limitations. Attention maps do not establish causality or tradability.

Independent review recomputed all development labels, training transforms, checkpoint choices, epoch losses, saved trace relationships, pooled losses and bootstrap intervals; no material forecasting implementation bug was found. Read-only review conduct was followed, though runtime permissions did not enforce it. A supplementary audit freshly checked 594 artifact hashes, both scalers and both converged logistic references, then sealed seven auxiliary files without replacing the original completion manifest.

Evidence: `artifacts/cmin_panel_2026-10-14_v1/`, especially `summary.json`, `completion.json`, `auxiliary_audit_hashes.json`, `source_date_disagreement_audit.json`, and the per-fit predictions/histories. Existing experiments remain preserved. The acquisition manifest-writing error and correction were recorded before preparation and fitting; sources were reused and reverified.

**Supported next use:** a reproducible conditional testbed for the professor's controlled comparisons, with a documented negative baseline. Forecasting strength is unvalidated. Stop automatic dataset/model changes; improvements must come from a specific new hypothesis and a separately frozen comparison.

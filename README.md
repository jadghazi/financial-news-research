# Financial news prediction: FININ-inspired research

This repository records an independent investigation of whether financial headlines, headline interactions, and news recency improve next-session direction prediction beyond prices alone. It contains the implementation, completed experiments, negative findings, and meeting material. The reference is [Wang, Cohen and Ma's FININ paper (2024)](https://aclanthology.org/2024.findings-emnlp.189/); this is **not an exact replication of its Reuters-based experiments**.

**Status — 20 October 2026:** the embedding/attention investigation is paused following the professor discussion. The completed evidence is preserved. A separate direct-LLM headline experiment is the proposed next direction; it has not been implemented or evaluated here. Repository renaming and project-folder reorganization have not been performed.

## Start here

| To understand… | Read |
|---|---|
| The research path, decisions, tests, and reasons for stopping | [Decision record](DECISIONS.md) |
| The current model, data timeline, and final MLP inputs | [CMIN pipeline](docs/cmin_pipeline.md) |
| Where code, reports, protocols, and evidence belong | [Repository guide](docs/repository_guide.md) |
| A concise visual explanation for a meeting | [Progress report, 3 pages](output/pdf/FININ_meeting_progress_2026-10-14.pdf) and [attention/MLP explainer, 2 pages](output/pdf/FININ_attention_and_MLP_explainer_2026-10-19.pdf) |
| The last completed experiment | [News-timing findings, 19 October](reports/cmin_timing_2026-10-19.md) |
| What a Git clone includes and how to inspect it | [Reproducibility and commands](docs/cmin_runbook.md), [laptop sync](docs/laptop_sync.md) |

The PDFs are dated presentations. They precede some later findings; use the decision record and latest reports for current status. Older reports preserve the decisions made at their dates, including proposals subsequently completed or superseded.

## Latest completed setup

The CMIN-US adaptation uses **12 companies**, five exchange sessions of prices and news-arrival buckets, and up to six selected headlines per bucket (at most 30 per forecast). Frozen RoBERTa creates headline vectors; frozen financial TinyBERT produces sentiment probabilities. Learned fusion, headline self-attention, and market-query weighting form the inputs to a small prediction MLP.

At session **d's close**, the target is whether adjusted close at the **next exchange session** exceeds adjusted close at d. Five- or twenty-day histories in FININ refer to input lookbacks, not multi-day prediction horizons. Our default availability rule places source-date D news into its first exchange-close bucket after D+2 calendar days at midnight UTC. It is an unverified availability assumption, not an established publication timestamp.

The baseline uses **160 MLP inputs**: five 16-value market representations and five 16-value weighted news summaries. The temporal studies use a different, matched **96-input** geometry: the same 80 market values and one 16-value pooled news summary. Compare effects within their matched geometry.

There are **9,984 development company-days**, with **2,004 evaluation company-days on 167 dates** across May–August 2020 and January–April 2021; seeds 17 and 42. Later comparisons reuse inspected development periods and are exploratory. Reserved late-2021 outcomes were not used by these CMIN studies.

## What the experiments establish

| Question | Completed check | Supported result |
|---|---|---|
| Does the model use headlines? | Fixed-model headline replacement, gradients, and attention interventions | Replacing headline vectors changed 44/44 selected news-bearing model-cases, by up to 6.76 percentage points. This establishes computational dependence, not improved accuracy. [Report](reports/cmin_news_dependency_2026-10-14.md) |
| Does the news model beat prices? | Matched 160-input baseline comparison | Full attention log loss **0.690388**, prices **0.690095**, training-prevalence probabilities **0.689820**; lower is better. No reliable incremental benefit. [Report](reports/cmin_public_pipeline_2026-10-14.md) |
| Does explicit recency weighting help? | Uniform versus 48-hour half-life, matched 96-input models; 8 fits | Loss difference **−0.000324**, conditional 95% interval **[−0.001177, +0.000539]**. No stable practical benefit established. [Report](reports/cmin_news_decay_2026-10-14.md) |
| Does older news help with prices fixed? | Current arrival bucket versus all five buckets; 8 new fits, 4 reused | All-news minus current-news loss **+0.000560**, interval **[−0.000031, +0.001155]**. No robust useful older-news addition. [Report](reports/cmin_older_news_2026-10-14.md) |
| Does headline self-attention help? | Remove self-attention, retain market-query weighting; 4 new fits | Removing attention increases pooled loss by **0.002999**, but the effect reverses across periods. No stable forecasting advantage over prices established. [Report](reports/cmin_interactions_2026-10-14.md) |
| Was the news delay the explanation? | Three newer-news policies; 12 new fits | None improves average log loss over the original news model or prices. The underlying cause remains unresolved. [Report](reports/cmin_timing_2026-10-19.md) |

These loss values average raw seed-model losses; they are not probability ensembles. Uncertainty conditions on saved fits and does not include training uncertainty. The predeclared practical loss-improvement criterion was 0.005. The historical tests establish neither equivalence nor that news is universally useless.

Attention-vector similarity raised an excessive-mixing question, but does not demonstrate semantic information loss. Dataset quality, headline selection, representation, optimization, and information already reflected in prices remain competing explanations. No executable trading advantage or market-diffusion mechanism has been established.

## Repository contents

```text
README.md / DECISIONS.md     Overview and evidence-linked research history
AGENTS.md                   Instructions for an agent reading or extending the work
src/ / scripts/ / configs/  Implementations and versioned experiment settings
tests/                      Alignment, model, training, and synthetic integrity checks
docs/                       Architecture, protocols, runbooks, and repository guide
reports/                    Dated findings and verification records
artifacts/                  Selected tracked evidence; larger local runs are ignored
output/pdf/ / figures/      Meeting PDFs, available LaTeX sources, and figures
paper/                      Reference FININ paper
```

Earlier phases include the SPY/NIFTY prototype, StockNet tweet rebuild, and SPNews headline adaptation. They are retained as research history and are indexed in the [repository guide](docs/repository_guide.md). Their metrics, datasets, and timing policies must not be treated as the latest CMIN setup.

## Inspect and verify

Reading the PDFs, reports, JSON summaries, and prediction CSVs needs no model downloads. With Python 3.12 and Git, the repository audit uses only the standard library:

```powershell
python scripts/audit_repository.py
```

It checks tracked-file organization, documentation links, saved-evidence hashes, PDF build hashes, and CMIN prediction metrics without downloading data or running market experiments. The [20 October audit](reports/repository_audit_2026-10-20.json) records this sharing review.

For the full engineering test suite after installing the documented dependencies:

```powershell
& .venv/Scripts/python.exe -m unittest discover -s tests -q
```

Tests include small synthetic learning checks; passing them does not establish financial forecasting value. Do not run the numbered experiment/export scripts merely to inspect results: some fit models, download data, or write fixed output locations.

Git includes selected predictions, learning histories, configurations, source snapshots, and audits. **Raw corpora, feature arrays, pretrained weights, and neural checkpoints remain local.** The clone supports inspection and selected metric recalculation, not a complete replay of every experiment. [Evidence scope and prerequisites](docs/cmin_runbook.md).

This is academic research code. Dependencies retain their own licenses; the included [FININ paper](paper/Wang%20et%20al.%20-%202024%20-%20Modeling%20News%20Interactions%20and%20Influence%20for%20Financial%20Market%20Prediction.pdf) is distributed under [ACL's CC BY 4.0 terms](https://aclanthology.org/faq/copyright/). No separate license for this repository's code has been declared.

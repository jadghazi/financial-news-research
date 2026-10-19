# FININ Research Prototype

An independent, reduced-scale implementation of the core pipeline in [FININ: Modeling News Interactions and Influence for Financial Market Prediction](https://aclanthology.org/2024.findings-emnlp.189/) (Wang, Cohen, and Ma, 2024).

## Latest research and meeting material — 19 October 2026

The current public-data adaptation uses CMIN-US headlines for 12 companies, five-session price/news histories, frozen RoBERTa and generated financial sentiment. News changes selected predictions, but a reliable forecasting benefit over prices alone remains unproven. The completed timing comparison found no reliable improvement from newer-news policies; its experimental code is archived separately. The original timing rule remains active.

- [Meeting progress report — PDF](output/pdf/FININ_meeting_progress_2026-10-14.pdf)
- [Self-attention and final MLP explainer — PDF](output/pdf/FININ_attention_and_MLP_explainer_2026-10-19.pdf)
- [Current pipeline and controls](docs/cmin_pipeline.md)
- [News timing comparison — latest findings](reports/cmin_timing_2026-10-19.md)
- [Headline interaction findings](reports/cmin_interactions_2026-10-14.md)
- [News decay findings](reports/cmin_news_decay_2026-10-14.md)
- [Laptop sync and saved-evidence scope](docs/laptop_sync.md)

The Git repository includes code, protocols, reports, figures, PDFs with their available LaTeX sources, and a compact selection of CMIN experiment summaries and predictions. Large raw datasets, frozen feature caches and model checkpoints remain local. The dated PDFs describe the work at their publication dates; use the latest timing report for the subsequent timing result. The sections below describe the historical SPY/NIFTY prototype.

## Research objective

FININ studies whether relationships among financial news items, together with the current market state, help predict market direction. This project reproduces its main computational idea: encode news and market inputs, model interactions among headlines, use the market representation to weight the headlines, and predict whether the next trading session closes higher.

The prototype tests that architecture on accessible data. It is an independent study rather than an exact replication of the paper's Reuters-based experiments.

## Implemented pipeline

1. **Prepare examples:** align dated NIFTY headlines with SPY market data and next-session direction labels. Because the news source lacks publication times, each forecast uses headlines from the preceding trading session.
2. **Encode inputs:** create frozen BGE-small text representations and financial TinyBERT sentiment probabilities; transform numerical and text inputs with learned projection and fusion layers.
3. **Model news interactions:** apply self-attention across the selected headlines for each example.
4. **Condition on the market:** use the market representation to score headlines and form an attention-weighted news summary.
5. **Predict and evaluate:** combine market and news representations in an MLP classifier; compare against simple baselines and controlled ablations on held-out dates.

The [architecture and code map](docs/architecture.md) connects these stages to the implementation.

## Experimental scope and findings

| | Original FININ study | This prototype |
| --- | --- | --- |
| News | Reuters news and sentiment | NIFTY headlines; generated sentiment |
| Market | S&P 500 and NASDAQ 100 indices | SPY price series as an S&P 500 proxy |
| History per prediction | Up to 20 input days | One input day |
| Evaluation | Multiple time windows | One chronological split; three training seeds |

The prototype produced 2,107 dated examples, split into 1,475 training, 315 validation, and 317 test examples. On the test dates, reduced FININ averaged **58.4% accuracy** and **50.4% balanced accuracy** across three seeds. Always predicting up reached **58.0% accuracy** and **50.0% balanced accuracy**. The model predicted up on nearly every test date, so this experiment does not establish a useful directional gain from news attention.

See the [full results](reports/results.md) and [two-page research brief](output/pdf/FININ_meeting_brief.pdf) for the comparisons and limitations.

## Reproducibility

The [runbook](docs/runbook.md) covers environment setup, data preparation, feature caching, training, evaluation, and verification. The [reproduction plan](REPRODUCTION_PLAN.md) records the fixed data and model choices. Source manifests, generated data checks, and a [prototype audit](reports/prototype_audit.json) are tracked. Raw datasets, feature arrays, and checkpoints are excluded from Git; the runbook explains how to rebuild them.

The original [paper PDF](paper/Wang%20et%20al.%20-%202024%20-%20Modeling%20News%20Interactions%20and%20Influence%20for%20Financial%20Market%20Prediction.pdf) is included for reference. It is an ACL Anthology publication distributed under [CC BY 4.0](https://aclanthology.org/faq/copyright/).

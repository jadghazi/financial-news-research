# Are news and attention actually used?

**Yes, the saved neural models computationally depend on headline representations and attention. This has not established a repeatable forecasting improvement.** The check below was requested after the baseline study and is a retrospective dependency diagnostic, not another training experiment.

`scripts/19_cmin_news_consumption.py` loaded the four selected full-FININ checkpoints. It chose the first validation example per company with at least two active headlines across its five-session history, otherwise the first empty example. Selection did not use labels or prediction outcomes. There are 24 unique company/date examples across two periods, including two empty DIS examples; evaluating two seeds yields 48 model-cases, of which 44 contain news.

The original outputs agreed with saved selected validation logits. Prices, fixed company context, sentiment and masks were retained when replacing standardized headline vectors with zero, equivalent to the raw training-vector mean. All 44 active model-cases changed probability by more than 1e-6, up to **6.76 percentage points**. Gradients with respect to semantic headline input were nonzero in all 44. Replacing sentiment with its training mean changed 34/44, up to 0.12 percentage points.

With existing weights retained, bypassing headline interactions changed 44/44 probabilities, up to 7.84 points. Switching market-query weighting to uniform pooling changed 16/44, up to 3.05 points. Some daily buckets contain only one headline or nearly uniform weights, so unchanged uniform-pooling cases do not establish an unused query pathway. These transformations change the input/representation distribution; they are not retrained ablations, forecasts of actual alternative events, comprehension tests or causal importance estimates.

**Policy distinction:** this checks the raw selected neural networks. The baseline's second-period FININ seed42 selected a constant-probability fallback using validation, which ignores neural forecasts by design. Raw neural and selected-system results remain separately reported.

Independent read-only review checked hashes, case selection, original probabilities, active counts and all reported probe maxima/counts, finding no material diagnostic defect. No fitting, language-model inference, new outcome selection or reserved-period evaluation occurred. Main-agent inference/backpropagation covered only these small validation batches; the reviewer did not rerun inference. Runtime permissions did not enforce read-only reviewer access.

## What FININ actually reports

The [published FININ paper](https://aclanthology.org/2024.findings-emnlp.189.pdf), Tables2–3, reports S&P500 five-day-input accuracy .533 full, .506 without news and .529 without the influence quantifier, versus always-buy .520. At20 input days, no-news .537 exceeds full .522. Accuracy benefits are mixed; the paper emphasizes Sharpe. Its quantifier ablation does not isolate self-attention, and no separate sentiment-removal ablation is reported. Original predictions and financial-metric specifications remain unverified here.

## Why our adaptation might fail

Observed facts: the public setup delays date-only news to D+2 midnightUTC; some identical URLs have disagreeing dates; six headlines per arrival bucket are retained by hash rather than relevance; generic tone sentiment replaces instrument-linked TRNA sentiment; company prediction and fixed ticker descriptions replace index-level market descriptions/constituents. News-training losses can improve while validation losses worsen. These changes materially affect the information and fitting problem.

Plausible explanations include relevant information already entering observed prices before our forecast cutoff, noisy or weak company relevance, loss of useful stories through the cap, weak next-session signal, inappropriate representations, insufficient capacity/optimization, and overfitting. Their relative contribution has not been identified. Adding more inputs creates no guarantee of information about the future outcome. Attention learns weights from the forecast loss, not supervised labels saying which article truly matters.

The evidence rules out a wholly unused semantic-news branch in the neural models. It does not establish useful semantic understanding, explain the dominant failure mechanism, reproduce FININ or refute news prediction generally. No additional fits or tuning were performed.

Artifacts: `artifacts/diagnostics/cmin_news_consumption_2026-10-14/summary.json` and `examples.json`; baseline evidence remains unchanged under `artifacts/cmin_panel_2026-10-14_v1/`.

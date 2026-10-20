# FININ prototype: held-out results

## Research question

Does modeling interactions among headlines and weighting them against the market help predict the next trading session's SPY direction, compared with matched simpler models?

The experiment uses previous-trading-day NIFTY headlines, market data through forecast close, binary next-session direction, at most 16 headlines/example, frozen BGE-small text vectors, and financial TinyBERT sentiment scores.

Prepared rows: 1,475 train, 315 validation, 317 test. Models are selected by validation loss; test dates are not used for tuning.

## Comparison on the same 317 test dates

| Model | Accuracy | Balanced accuracy | Log loss |
| --- | ---: | ---: | ---: |
| Always up | 0.580 | 0.500 | 6.762 |
| Prices only | 0.583 ± 0.013 | 0.509 ± 0.015 | 0.689 ± 0.005 |
| Prices + average sentiment | 0.569 ± 0.005 | 0.499 ± 0.006 | 0.688 ± 0.005 |
| News mean pooling | 0.580 ± 0.000 | 0.500 ± 0.000 | 0.682 ± 0.001 |
| Market attention only | 0.580 ± 0.000 | 0.500 ± 0.000 | 0.682 ± 0.001 |
| FININ without sentiment | 0.580 ± 0.000 | 0.500 ± 0.000 | 0.681 ± 0.000 |
| Reduced FININ | 0.584 ± 0.005 | 0.504 ± 0.007 | 0.682 ± 0.001 |

Six trainable variants each ran with seeds 17, 42 and 73. Variation across seeds describes training stability on these particular dates; it is not a confidence interval over future market periods.

## Reading the result

The full model's mean accuracy is 0.584; mean pooling is 0.580; always predicting up is 0.580. The full-model seeds predicted up on 312–317 of 317 dates, while 184 dates actually rose. Their near-0.5 balanced accuracy shows that most of the apparent accuracy comes from the up-day majority. This experiment does not provide evidence that news attention improved directional classification over the simpler alternatives. It does not reproduce the paper's reported scores. An audit-added constant probability reference uses only the training up-rate (0.5471) and has test log loss 0.6824, versus 0.6816 for reduced FININ. This is a fairer probability comparison than assigning probability 1 to every day.

Accuracy is the share of correct directions. Balanced accuracy averages recall for up and down days, so it reveals models that mostly predict one class. Log loss also checks whether predicted probabilities are sensible; lower is better. The always-up baseline assigns a probability of 1.0, so its log loss sharply penalizes every down day; use its accuracy and balanced accuracy for the directional comparison.

## Limitations

NIFTY lacks individual publication times, so news is delayed one trading day. SPY substitutes for the S&P 500 index. Financial TinyBERT sentiment substitutes for Reuters instrument-related scores. Frozen text models released during or after some historical dates are used for a retrospective benchmark; attention weights are model diagnostics, not causal influence.

Complete per-seed results, learning histories, dated predictions and model checkpoints are saved in the ignored `artifacts/runs/` folder. The offline walkthrough and compact CSV are in `artifacts/meeting/`.

# FININ learning check — 12 October 2026

**The model can learn. Its original checkpoints learned little beyond the majority class; longer training increases fit, often with poor validation probability quality. One later checkpoint is a promising exception that depends on news, but its benefit is unconfirmed.**

## What we checked

Used unchanged version-1 data/features: 1,475 training and 315 validation dates. Ran two balanced 32-example memorization checks, then fixed 30-epoch curves for full FININ (seeds 17/42/73), price-only and mean pooling (seed 42). Original optimizer, learning rate, dropout and clipping were retained. Historical early-stopping trajectories and selected weights matched exactly. Training took 127 seconds on CPU. No new test evaluation, downloads or research implementation changes.

Validation was already used for selection. These are exploratory diagnostics, not independent evidence of improved forecasting. The protocol, dated probabilities, configurations and checkpoints are saved [here](../artifacts/diagnostics/learning_check_2026-10-12/PROTOCOL.md).

## Findings

**1. Computational learnability passes.** Both 32-example checks reached 100% training accuracy after 25 updates, with log loss approximately 0.048. One held numerical market inputs constant, leaving news/sentiment variation. This establishes that the news path can support memorization; it does not establish semantic understanding or predictive information.

**2. Majority-class behavior describes the original selected checkpoints.** All three full-model checkpoints predicted up on every validation date. The training-prevalence constant achieves 54.60% validation accuracy and 0.68891 log loss. Their losses were 0.68900, 0.68886 and 0.68893: essentially the constant reference. Previously saved validation news replacements/permutations changed no decisions at these checkpoints.

**3. Longer training exposes fitting and overconfidence.** Lower log loss is better; train/validation below use evaluation mode, unlike historical online training loss.

| Model/seed | Epoch-30 train loss | Epoch-30 validation loss | Best validation loss within 30 epochs |
|---|---:|---:|---:|
| Full/17 | 0.62673 | 0.72215 | 0.68900 (epoch 10) |
| Full/42 | 0.57373 | 0.74119 | 0.68886 (epoch 3) |
| Full/73 | 0.33681 | 1.14667 | 0.68349 (epoch 17) |
| Price-only/42 | 0.67949 | 0.69311 | 0.69066 (epoch 5) |
| Mean pooling/42 | 0.09320 | 2.53154 | 0.68685 (epoch 10) |

Poorer validation loss does not mean every ranking is random: epoch-30 AUC was 0.606 for full/73 and 0.601 for mean pooling. Their probabilities nevertheless became poor. Single-seed controls cannot establish architectural superiority.

**4. The later full/73 checkpoint materially changes with news.** Original patience would stop at epoch 9; epoch 17 reached 57.78% validation accuracy versus 54.60% constant (10 additional correct dates), balanced accuracy 54.32%, and AUC 0.549. This later improvement did not recur in the other full-model seeds. Mean pooling also found a later, smaller loss improvement.

A transparently post-result frozen check found:

- Fixing the news summary to its training mean changed 38 decisions, restoring all-up predictions and 54.60% accuracy.
- Three date permutations of news changed 60–67 decisions; accuracy fell to 51.11–54.92% and loss worsened.
- Uniform market-query weights changed six decisions and worsened loss, while accuracy stayed unchanged.

These support checkpoint-specific dependence on date-aligned news inputs. They do not isolate headline meaning from sentiment, coverage or market conditioning; replacing representations can create unnatural inputs. Direct-market replacement also retains market information in the news summary. These are interventions on frozen weights, not retrained ablations or causal tests.

## Decision

Keep the original gain description: **a small observed gain, with no convincing evidence that it is repeatable.** This diagnostic does not reclassify that test result. It removes total inability to learn as the leading explanation and flags early stopping/overconfidence for attention. Before the planned BGE–RoBERTa comparison, specify one matched training/checkpoint rule and evaluation plan; do not select longer training because one inspected seed improved. Dataset availability/provenance concerns remain unresolved. Independent source/artifact review is recorded in [REVIEW.md](../artifacts/diagnostics/learning_check_2026-10-12/REVIEW.md).

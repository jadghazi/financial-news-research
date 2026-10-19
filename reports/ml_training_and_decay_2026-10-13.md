# FININ: training quality, majority behavior and a first decay test

**The comparison pipeline works computationally. The tested news models still lack convincing forecasting value. Tuning reduced some losses, but did not solve that problem.**

## What we executed

129 neural fits: prices, mean pooling and full attention; three seeds; three chronological outer periods; baseline, weight-decay-only and a smaller regularized setup. Settings and common training duration were selected on an inner chronological split. Outer fits used fixed durations and never selected checkpoints from outer labels. Raw market scaling was refitted within every fitting partition. The safe trainer is now used by this runner.

The 990 outer dates were already explored historically, so these results remain retrospective. Scores below average three separate models' losses, not an ensemble.

| Model | Baseline loss ↓ | Tuned loss ↓ | Tuned balanced accuracy |
|---|---:|---:|---:|
| Prices | 0.69448 | 0.69154 | 49.70% |
| Mean pooling | 0.69131 | 0.68987 | 50.00% |
| Full attention | 0.69651 | 0.69197 | 49.63% |
| Training-prevalence constant | 0.68996 | — | 50.00% |

The mean-pooling loss advantage over constant is only 0.00010, with ranking near chance. Full attention still loses to constant. Some tuning-versus-baseline conditional intervals favor tuning, but that is improvement over a weak setup, not demonstrated news value.

## The always-up concern

Mean pooling predicts up on every outer date at all three seeds. Full attention makes 450/5/5 down calls across the respective seeds: one early model predicts down everywhere, while others predict up everywhere. Producing down calls therefore does not establish useful learning. Prices also produce both directions without beating the constant reference.

Learning itself is active: in the first inner window, baseline mean pooling reaches training loss 0.0033 while held-out loss reaches 4.5328 after 30 epochs. Full attention similarly reaches 0.0544/2.1941. These are strong overfitting patterns. Smaller capacity helps some curves, not consistently all architectures. At selected full checkpoints, text permutations change only four decisions across 2,970 seed/date evaluations and have tiny, mixed loss effects.

A post-result skeptical control calibrating only constant probabilities achieves loss 0.68932, better than every calibrated neural procedure. This adjustment uses recent inner class prevalence; it adds no news information and has chance discrimination within each window.

## Actual initial decay test

75 convex fits plus a synthetic solver check compared latest news, three-day uniform pooling and fixed exponential decay, with identical current prices, coverage controls and dates. Pooling occurs before prediction, so a linear predictor cannot undo separate day weights. No new encoder training was involved.

| News pooling on 988 matched dates | Loss ↓ |
|---|---:|
| Constant reference | 0.690113 |
| Latest day | 0.690220 |
| Three-day uniform | 0.689942 |
| Three-day decay | 0.690140 |

Decay assigns complete-day weights 1/7, 2/7, 4/7. It loses to uniform overall; the paired conditional interval includes zero. Every direction classifier is all-up. This is a simple frozen-feature baseline test, not a full neural FININ decay experiment or a refutation of diffusion theory.

## Research decision

[FININ Section 4.1](https://aclanthology.org/2024.findings-emnlp.189.pdf) explicitly freezes its language model. Our frozen encoders do not omit a mandatory fine-tuning step. Other verified papers support controlled tuning and representation comparisons, not guaranteed gains from longer inputs or more capacity.

Keep this chronological benchmark for subsequent tasks. Predictive learning remains insufficient; do not expand the final MLP to conceal it. Neural decay, task-specific representation adaptation and objective changes remain separate experiments. Publication-time availability, repeated headlines and pretrained overlap remain unresolved; no new historical test evaluation or trading claim was made.

The research skill and both specialist roles were used. Protocols, source snapshots, predictions and reviews are under `artifacts/experiments/ml_training_study_2026-10-13/` and `temporal_pooling_study_2026-10-13/`.

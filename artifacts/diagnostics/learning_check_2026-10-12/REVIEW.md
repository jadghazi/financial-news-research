# Independent skeptical review

The custom `skeptical_reviewer` role was actually delegated this audit in the current conversation as `/root/learning_review`, then recalled for the late-checkpoint supplement. It inspected original implementation, historical runs, the protocol and saved diagnostic predictions independently. It did not train, execute models or edit files. The inherited runtime granted write access despite the role's read-only configuration: read-only behavior was maintained, but sandbox enforcement was not verified.

## Evidence checks

- Independently recalculated the main CSV's 268,500 dated predictions across 300 epoch/split groups, including loss, accuracy, balanced accuracy, Brier score and AUC.
- Checked the historical trajectory/checkpoint comparisons and balanced tiny-subset construction. Original-history prefixes and selected weights matched; no implementation or metric defect was identified.
- Recalculated all 1,890 supplemental predictions and decision changes. Dates/labels matched epoch-17 validation records; original probabilities agreed within 5.96e-8, consistent with batch-size numerics. Checkpoint/helper hashes and training-only reference construction checked out.
- Confirmed all 127 preserved original files remained unchanged. No fresh test model evaluation occurred. During its initial source inspection the reviewer inadvertently read historical test outputs from an old run JSON; those values were excluded from this assessment.

## Findings, ranked by scientific importance

1. **Validation reuse limits conclusions.** Later checkpoint selection and its supplemental controls followed observed results. They support exploratory hypotheses, not a repeatable gain or a revised interpretation of the old test result.
2. **Learning exists, with mixed transfer.** Tiny memorization and stronger full-data training fit reject total computational inability to learn. Worsening validation loss supports probability overconfidence/overfitting, but later AUC improvements and full/73 epoch 17 contradict blanket claims that longer training never helps or all rankings are random.
3. **Late news dependence is verified.** News-summary replacement changes 38 decisions and removes the observed accuracy gain; shuffled daily news bundles change 60–67 decisions and worsen loss. This supports checkpoint-specific dependence on aligned inputs. It does not distinguish semantics from sentiment, coverage, date/regime correlates or disrupted input relationships.
4. **Controls have limited scope.** Direct-market replacement retains market-conditioned news. Uniform query weighting retains self-attention. Neither isolates a complete price ablation or headline-interaction benefit. News permutations preserve each daily bundle but not serial dependence; they are sensitivity checks, not statistical significance tests. One-seed baselines cannot establish superiority.

The next evidential step is a predeclared matched comparison across seeds and temporal periods with retrained controls. No further experiments were requested or performed by the reviewer.

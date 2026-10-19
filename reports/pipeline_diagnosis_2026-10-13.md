# FININ pipeline diagnosis — 13 October 2026

**The headlines work computationally, and some trained checkpoints materially use them. We found overfitting and weak evidence of useful market prediction, rather than a disconnected news path.**

| Question | Executed check | Finding |
|---|---|---|
| Can the architecture learn unseen signals? | Price, encoded-text and sentiment rules; three seeds; full model plus mean pooling | All 12 positive runs reached 100% balanced accuracy on separately generated holdouts. A shuffled-label control scored 51.6%. This tests learnability, not financial meaning. |
| Do real headlines reach the prediction? | Gradients, manual forward tracing and seven saved checkpoints | Text gradients were nonzero; padded inputs had zero gradients; traced outputs matched. Original selected checkpoints barely used text, but later checkpoints did. |
| Can cached features be reproduced? | Re-encode 16 actual headlines using pinned local models | BGE maximum absolute difference was 0.00000103; sentiment matched exactly. This is a sample check. |
| Does removing the news delay help? | Same-day versus previous-session news; matched dates and nested chronological linear baselines | No demonstrated rescue. Neither timing choice beat the constant-probability reference on pooled log loss. |

**What happened during learning.** At the saved full-model seed-73 epoch-17 checkpoint, accuracy was 57.8% on 315 validation dates. Permuting only text changed 58 decisions and worsened log loss from 0.6835 to 0.7099; permuting sentiment changed three decisions. Headlines therefore affect this model. The permutation disrupts natural associations and does not establish causal or unique news information.

Later training produced severe overconfidence: final full-model log loss was 1.1467, and final mean-pooling loss was 3.1010, despite accuracies of 57.1% and 58.4%. Lower log loss is better. Validation-loss selection consequently often preferred earlier, nearly all-up checkpoints. Improved accuracy at some later checkpoints is real as an observation, but does not establish repeatable forecasting value.

**The simpler check.** Four expanding temporal evaluations yielded 810 distinct outer predictions. The training-prevalence probability reference had log loss 0.68976; prices alone 0.69029; prices plus news 0.69293 with same-day news and 0.69027 with delayed news. All ten fitted baseline variants lost to the reference on pooled log loss. Timing-effect block intervals included zero. Normalizing BGE headline vectors made negligible difference. These are retrospective results on a restricted news-covered cohort, not a fresh independent test.

**Engineering repair delivered.** The historical trainer can consume training randomness during evaluation. The new CPU engine, `src/research_training.py`, isolates evaluation randomness and modes, rejects diagnostic model-state mutation, retains selected/final checkpoints separately, uses stable raw-logit losses, and exposes constant fallback separately. Fifteen regression/integrity tests pass. This prospective repair was not used in these market comparisons and is not an explanation for their disappointing results.

**Decision.** Computational pipeline health is supported. Useful generalization remains unproven. Exact historical headline availability is unresolved; date-only same-day news is not automatically usable at the forecast cutoff. Before further architecture additions, the priority is an availability-defined evaluation and a bounded generalization/overfitting study. More capacity or more headlines alone has no demonstrated justification from these checks.

The research skill, literature investigator and independent skeptical reviewer were used. Saved predictions, inputs, source snapshots and protocols are in `artifacts/diagnostics/pipeline_diagnosis_2026-10-13/`; the reviewer checked original evidence. Earlier research artifacts were preserved.

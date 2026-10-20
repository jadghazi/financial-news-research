# Does increasing news coverage help?

**Finding:** raising the cap from 16 to 64 did not resolve majority-class predictions or provide convincing useful improvement in this exploratory validation comparison. This tests headline quantity under our fixed pipeline; it does not establish that news is useless or 16 is universally optimal.

We trained six new BGE models: full FININ and simple news averaging, each with seeds 17/42/73 and the same 30-epoch budget. We reused six verified 16-headline BGE runs and three price-only references. The 1,475 training and 315 validation dates, labels, prices, frozen encoders, initialization and minibatch orders match. Only the headline cap changed. No new test evaluation was performed.

| Model | Cap | Validation loss ↓ | Accuracy | Balanced accuracy |
|---|---:|---:|---:|---:|
| Training-prevalence constant | — | 0.688906 | 54.60% | 50.00% |
| Full FININ | 16 | 0.689128 | 54.60% | 50.00% |
| Full FININ | 64 | 0.689018 | 54.60% | 50.00% |
| Simple news averaging | 16 | 0.687380 | 55.24% | 50.94% |
| Simple news averaging | 64 | 0.689098 | 54.60% | 50.00% |

Network rows average three individually selected models, not ensemble predictions. All six cap-64 networks made **zero down predictions**. Five also lost to the constant on validation loss; the sixth beat it by only 0.000009. Replacing that sixth model's news summary with a training average improved its loss slightly, so its tiny edge does not demonstrate useful date-specific news. The constant fallback is recorded separately and cannot explain the networks' all-up behavior.

Full FININ's mean loss improved by just 0.000110 against cap 16, with one seed worsening. Descriptive paired block intervals include zero. Averaging worsened in all three matched seeds; its small cap-16 accuracy gains disappeared. These are observations on reused, checkpoint-selecting validation dates, not independent generalization evidence. Conditional bootstrap intervals do not correct selection or previous searches; blocks contain observed forecast rows, with calendar gaps.

**The pipeline can learn.** The expanded model passed gradient, masking and checkpoint checks and achieved 100% accuracy on a balanced 32-example training batch, including with prices held fixed. Longer full-data fitting lowered training loss but worsened validation loss in every run. This supports learning and overfitting, not a dead computation path.

Replacing news summaries with their training average or moving news bundles between validation dates changed probabilities slightly but changed **no directional decisions** in any cap-64 model. This provides little evidence of useful news dependence in the selected models; it does not isolate headline meaning from sentiment or count.

**Decision:** keep cap 16 as the comparison reference and retain the cap-64 cache. Do not expand the cap further on this evidence. The next distinct question is whether matched longer news/price histories help, rather than continuing to search headline counts on this validation period. Date-only availability, coverage shifts and pretrained-data overlap remain limitations. No trading claim follows.

Protocol, predictions, analysis and independent review: `artifacts/experiments/headline_cap_2026-10-12/`. Existing research artifacts remain unchanged.

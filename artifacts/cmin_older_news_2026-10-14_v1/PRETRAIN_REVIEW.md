# Independent design review — 2026-10-14

Recorded by the main agent from the read-only skeptical_reviewer (/root/evidence_review). No reviewer file writes or experiments. Runtime permissions do not enforce read-only access; conduct was read-only.

Verdict: the design can answer whether adding older proxy-available news improves this model's forecasts beyond the current arrival bucket. No design blocker found, conditional on implementing reuse guards.

The current-bucket restriction [0,1] retains only the cutoff session's arrival bucket, including empty buckets, and keeps all five price sessions. Reuse all four earlier uniform-pooled fits, not selected favorable seeds. Verify source/hyperparameter/plan/transform and actual initial-state matching; newly train the recent model rather than using a fixed-weight masking diagnostic. The pooled price control provides matching 96-value geometry. Copy existing fitted logistic references into both namespaces to avoid implicit refitting. Four new fits/600 seconds per component and eight/1,200 seconds overall; retain failures without retries.

Adding old news changes pooling shares, sentiment and coverage as well as content. A gain supports the older-news information package under this model, not older semantic content, causal diffusion or intrinsic memory. Define coverage groups before predictions, keep all matched cases primary and report subgroup counts/dates/periods/seeds descriptively. Existing inspected periods remain exploratory; retain the 0.005 criterion and constant/price references.

The main preflight subsequently recorded eight matched initial-state constructions, unchanged prices/labels, current-bucket masking, frozen coverage groups, copied reference hashes and 1,085 protected prior files. Actual saved training initial states are checked again in analysis.

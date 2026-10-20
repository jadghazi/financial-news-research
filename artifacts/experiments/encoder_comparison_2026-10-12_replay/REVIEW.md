# Independent review and evidence map

Actually recalled the custom literature investigator `/root/encoder_evidence` and skeptical reviewer `/root/learning_review` from this conversation. The literature role verified pinned official model files, installed Transformers support, first-token extraction choices and primary-source caveats. Its accessible sources include the [FININ paper](https://aclanthology.org/2024.findings-emnlp.189.pdf), [pinned RoBERTa files](https://huggingface.co/FacebookAI/roberta-base/tree/e2da8e2f811d1448a5b465c236feacd80ffbac7b) and [official-hosted card](https://huggingface.co/FacebookAI/roberta-base). Original paper pooling/variant details are unspecified; pretraining story overlap is unknown.

The reviewer independently inspected original/current code, protocol, loaded checkpoint tensors and recalculated saved arrays/predictions. It did not run models, fits or tests, or edit files. Both roles maintained read-only behavior; runtime inheritance did not enforce their read-only sandbox configuration.

## What was independently verified

- 805,500 epoch predictions in 900 split/epoch groups: dates/labels, stable raw-logit BCE, Brier score, AUC, accuracy, balanced accuracy and confusions.
- All 15 strict earliest-minimum selections, saved checkpoint replay, explicit network-only and fallback predictions;12 constants selected,3 BGE-pooling networks retained.
- All 48 fixed news interventions and their metrics/decision changes. Train-only reference construction and all-arm controls checked against code.
- Matched minibatch-order hashes per seed and bit-exact paired initial downstream parameters; only the declared native projection differs by 24,576 weights. Active parameter counts verified.
- All 424 preserved original and first-attempt files unchanged; recorded artifacts/source hashes matched.
- Tiny 32-date prediction metrics recomputed: 100% accuracy and BCE 0.003046/0.003581; amended checkpoint weights exactly match the failed attempt. The sole failure was a one-ULP float32 replay difference. Exact restored weights, identical decisions and bounded absolute/relative 1e-6 tolerance justify the numerical amendment.
- On the same 100 headline IDs, RoBERTa/BGE mean coordinate deviation was 0.01769/0.32909 and mean pairwise cosine 0.99791/0.48379. Loading had no missing/mismatched encoder weights; unused MLM-head weights were expected.

## Verdict and limitations

Supported: current RoBERTa computational integrity/learnability; no convincing benefit from this frozen raw first-token substitution; pragmatic retention of BGE. Full-model average loss is marginally better for RoBERTa but mixed by seed and below the constant's probability quality. BGE pooling improves selected losses, but that is exploratory evidence on reused selection dates.

Unsupported: intrinsic RoBERTa inferiority, useless news, reliable future forecasting, causal headline interaction/diffusion, or executable trading benefit. One BGE-pooling news permutation increases accuracy 173→176 despite worse loss; do not claim aligned news uniformly causes the accuracy gain.

Unresolved: input geometry/optimization, projection capacity, sentiment versus text contribution, publication availability, revised data, recurring stories, pretraining overlap and independent temporal robustness.

Evidence: `PROTOCOL.md`, `cache_reference.json`, `execution_identity.json`, `gates.json`, `results.json`, `descriptive_summary.json`, `runs/*/{history.json,predictions.csv,selected_predictions.csv,fallback_predictions.csv,sensitivity.csv,summary.json}`, and `preservation_after_train.json`. The original namespace preserves cache/benchmark/source snapshots and the failed replay attempt. Use a new experiment version for future fits; the completed runner refuses overwriting results.

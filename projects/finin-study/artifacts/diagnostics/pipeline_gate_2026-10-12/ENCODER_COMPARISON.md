# Frozen encoder comparison protocol

**Decision:** compare BGE-small with frozen RoBERTa-base as a bounded exploratory representation check. This is not an exact paper replication or a confirmatory generalization test. No encoder comparison has been executed yet.

## Fixed comparison

- Keep current SPY/NIFTY dates, six price features, TinyBERT sentiment, 16 selected headlines, one input day, labels and splits. Forecast at session d's close using news from prior session n; predict close[q] > close[d] for the next session q. Retain availability/revised-price/pretraining caveats.
- BGE revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; RoBERTa `FacebookAI/roberta-base`, model/tokenizer revision `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. RoBERTa-base is our choice: the [FININ full paper](https://aclanthology.org/2024.findings-emnlp.189.pdf) does not recover variant, pooling, token cap or checkpoint rule.
- Use evaluation/inference mode, final-layer first token, raw float32 vectors, maximum 128 tokens. Re-encode the identical market description with each encoder. Preserve the existing BGE extraction; changing pooling, normalization and sentiment together would confound this first comparison. Report tokenizer-specific truncation and vector norms. RoBERTa first-token extraction is a declared choice, not a guaranteed sentence representation.
- Full FININ and mean pooling for each encoder, each with seeds 17/42/73; price-only once per seed, shared across encoders; training-prevalence probability and always-up references. This is 15 trained runs, not a seed search. Keep downstream hidden width 64, dropout .1, Adam .001, batch 16, clipping 1, CPU four threads. Dedicated seeded minibatch generators give paired orders; differing embedding widths do not yield identical initialized weights.

## Training and checkpoint rule

Every trained condition gets exactly 30 epochs, with no patience stopping, schedule, threshold tuning or hyperparameter sweep. Stop on nonfinite values, retain failed outputs, and do not silently change settings. Evaluate each epoch identically and preserve RNG during additional diagnostic evaluations. Use unrounded validation BCE to select the network checkpoint: strict numeric minimum, earliest epoch on exact ties. This intentionally differs from v1's 1e-5 improvement threshold.

For each condition/seed also compare the network's selected BCE with the training-prevalence constant. Use the constant if its validation BCE is lower or exactly tied. Save and report BOTH network-only and fallback-selected predictions, so fallback cannot disguise a model that fails to learn. Never select the best seed. Use a fixed .5 classification threshold; report loss, Brier score, AUC, balanced accuracy, up/down calls, training fit and news sensitivity alongside accuracy.

Primary encoder comparison: paired network-only validation loss differences within seed and architecture; also report fallback decisions. All seeds and failures must appear. Improvements confined to one seed, a narrow cluster of dates, probability scale or the full model alone require explanation; mean pooling is the simpler control. This phase does not claim a causal attention benefit. Confidence intervals on the same checkpoint-selection dates would not make the comparison confirmatory; seeds are not independent market histories.

## Requirements before execution

The current pipeline passes its BGE gate, but RoBERTa compatibility is unimplemented: its native width is 768, versus hardcoded 384 in the current loader/projection. Use a separate versioned runner/cache namespace; derive both headline and description widths from the selected immutable cache and verify them jointly. Bind cache identity to model/tokenizer revisions, pooling, normalization, token cap, feature dtype/shape, source IDs/text, description and generating-code hashes. Never overwrite `artifacts/features` or v1 runs. Reuse unchanged sentiment with verified ID mapping. Re-run finite-feature, masking, gradient/update, checkpoint and small-sample learning gates for the new path before full fits.

Report active parameter counts: native RoBERTa projection adds 24,576 learned weights, so this is an encoder-package comparison with a capacity difference. A claim of semantic superiority would require a further matched-capacity control.

The [official RoBERTa repository](https://huggingface.co/FacebookAI/roberta-base/commit/e2da8e2f811d1448a5b465c236feacd80ffbac7b) lists approximately 499 MB safetensors weights; download only necessary files. Benchmark 100 headlines first, record actual CPU time/storage, then cache once. The [model card](https://huggingface.co/FacebookAI/roberta-base) documents CC-News pretraining overlapping our historical period; exact story overlap remains unknown. Current training budget is informed by previously inspected validation. Treat this entire comparison as exploratory, keep the old test out of selection, and require independent temporal evidence before a reliable forecasting claim.

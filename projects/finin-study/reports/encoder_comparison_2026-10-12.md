# CPU RoBERTa comparison — 12 October 2026

**The RoBERTa path works and learns, but this frozen first-token substitution did not produce a convincing forecasting improvement. Keep BGE as the current working default; retain the RoBERTa cache for future controlled checks.**

## What ran

Downloaded pinned RoBERTa-base safetensors/tokenizer files, verifying the weights against the publisher SHA256. On CPU, 100 headlines took 1.73 seconds, with approximately 673 MB whole-process RAM. Cached 27,261 unique training/validation headlines and the same market description in 387 seconds; headline vectors occupy about 84 MB. No GPU setup or new libraries were needed.

Both balanced 32-example RoBERTa learning checks reached 100% accuracy, including prices fixed to their training mean: 100/175 updates. Gradients and parameter updates reached all ten core blocks; masking, padded-input gradients and checkpoint checks passed. One absolute-only replay check failed by 4.77e-7 despite exact restored weights. Its failure/source were preserved; a reviewed float32 tolerance amendment ran in a separate namespace, with unchanged training and bit-identical tiny checkpoints.

Completed 15 matched 30-epoch runs: full FININ and mean pooling for both encoders, three seeds each, plus shared price-only controls. Fitting took approximately 6.2 minutes. Data, sentiment, selected headlines, input lookback and downstream settings were fixed. Checkpoints used the same strict minimum validation loss rule; all seed outcomes and constant fallback outputs were retained. No new test predictions were made.

## Results

Network-only means across three seeds on the same 315 validation dates; these are not ensemble scores or independent market histories. Lower loss is better.

| Condition | Validation loss | Accuracy |
|---|---:|---:|
| Training-prevalence constant / always-up decisions |0.688906|54.60%|
| BGE, full FININ |0.689128|54.60%|
| RoBERTa, full FININ |0.689074|54.60%|
| BGE, mean pooling |0.687380|55.24%|
| RoBERTa, mean pooling |0.689095|54.60%|
| Price-only |0.689843|54.92%|

All six selected RoBERTa news models predicted up throughout validation and lost to the constant on loss. Full FININ did so with BGE too. RoBERTa's marginally better full-model average loss was mixed across seeds and still worse than the constant.

BGE pooling beat RoBERTa pooling in all three losses. Its two nonconstant seeds added five/one correct dates versus always-up. Replacing or permuting news worsened probability loss, supporting checkpoint-specific input dependence. Accuracy did not uniformly worsen: one permutation improved a seed's correct count 173→176. These frozen controls cannot isolate headline meaning from sentiment or prove causality.

## Interpretation and next step

Both encoders can fit training data. Stronger fit often overfits validation. RoBERTa's raw first-token vectors were much more alike than BGE's on 100 matched headlines; representation geometry, fixed optimization and a 24,576-weight larger projection remain alternative explanations. This tests our declared extraction, not every RoBERTa configuration or an exact original-paper replication.

Validation was reused for selection; availability, pretraining overlap, 14 recurring train/validation headline IDs and independent temporal generalization remain unresolved. The original small test-gain conclusion is unchanged. Next, investigate news coverage/lookback with simple pooling and matched price-history controls, retaining full attention as a comparator.

The research skill and both custom agents were used. The reviewer independently checked 805,500 dated epoch predictions, selection/fallbacks, all 48 news interventions and preservation of 424 prior files. [Protocol, results and review](../artifacts/experiments/encoder_comparison_2026-10-12_replay/REVIEW.md); [original cache and benchmark](../artifacts/experiments/encoder_comparison_2026-10-12/roberta_cache/manifest.json).

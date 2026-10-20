# FININ pipeline readiness — 12 October 2026

**The existing BGE pipeline is computationally working and capable of learning. No critical defect explaining a dead news pathway was found. Useful, repeatable forecasting remains unproven.**

## Fresh checks

- All seven existing alignment/model tests passed; [captured output](../artifacts/diagnostics/pipeline_gate_2026-10-12/unittest_execution.json) is retained.
- All 1,790 training/validation records and 28,632 headline slots matched frozen sources, selection, calendar offsets, independently recomputed targets, training-only scaling and cache mappings. Features were finite and varied across headlines.
- Eight training headlines and the market description were re-encoded offline using pinned BGE weights. Saved vectors agreed within 7.75e-7; sampled TinyBERT scores reproduced exactly and positive/negative/neutral controls behaved as expected. This is sampled regeneration, not regeneration of the entire cache.
- A small real-feature optimizer smoke check confirmed finite nonzero gradients and updates across all ten core blocks, including both attention components and the final predictor. Text, sentiment and market inputs received gradients; padded inputs received none. Masking and headline-order invariance passed.
- Checkpoint weights restored exactly. Logits agreed within 1.49e-8. An initial demand for bit-exact logits failed; that attempt was preserved and the gate transparently amended to require exact weights and a 1e-7 numerical tolerance.
- Both saved 32-example memorization checks still achieved 100% accuracy, including the condition with numerical market inputs fixed. These checkpoints were replayed, not retrained.

## Next comparison rule

The [reviewed protocol](../artifacts/diagnostics/pipeline_gate_2026-10-12/ENCODER_COMPARISON.md) gives BGE and RoBERTa the same 30-epoch opportunity, seeds and checkpoint-selection rule. It retains network-only outcomes alongside a constant-probability fallback and includes mean pooling and shared price-only controls. No best-seed selection. The budget is informed by inspected historical validation, so the comparison remains exploratory.

RoBERTa has **not** been downloaded or tested. Before execution, its different vector width requires a separate versioned cache/loader path and fresh integrity/learning gates. The literature agent verified that the paper leaves several encoder details unspecified; no improvement is presumed.

## Limits and harness

Fourteen exact headline IDs recur across training/validation: recorded for follow-up, not automatic proof of leakage. News publication availability, revised prices, pretraining overlap and generalization remain unresolved. Both custom agents actually contributed: literature verification and independent skeptical review. They maintained read-only behavior; inherited runtime permissions did not enforce their read-only configuration. All 172 preserved original files remain unchanged. No full-data training, new test evaluation or installation was performed in this step; the two smoke attempts used one optimizer update each.

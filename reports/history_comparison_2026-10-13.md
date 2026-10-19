# Does three-session context help?

**Finding:** longer price/news history did not provide convincing predictive benefit under this fixed pipeline. Keep the one-day input as the reference; this is not proof that longer histories cannot help.

We trained 18 fresh runs: six controls, seeds 17/42/73, 30 epochs each. All use the same 1,473 training and 313 validation forecasts; two warm-up dates per split were omitted to keep history within each source split. BGE/TinyBERT, cap 16 per news day and training-only scaling remain fixed. No downloads, new encodings or test evaluation. Market slots are exact consecutive SPY sessions; each news slot retains our one-session delay. Missing source-news observations stay empty.

| Network/input | Validation loss ↓ | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| Training-prevalence constant | 0.689445 | 54.31% | 50.00% |
| FININ: one price/news session (A) | 0.689358 | 54.21% | 50.88% |
| FININ: three prices, one news session (B) | 0.690122 | 54.31% | 50.00% |
| FININ: three price/news sessions (C) | 0.690609 | 54.10% | 49.80% |
| Daily averaging: three price/news sessions | 0.691018 | 54.31% | 50.00% |
| Numerical prices only: one session | 0.689598 | 53.67% | 49.52% |
| Numerical prices only: three sessions | 0.694470 | 52.93% | 49.17% |

Network rows average three separately selected models, not an ensemble or constant-fallback predictions.

**Main contrast:** adding older news with prices held fixed (C−B) worsened mean loss by 0.000487: two seeds worsened; one improved by only 0.000027. Adding price history (B−A) worsened loss in all three seeds. All C and averaging runs lost to the constant. C beats the weak three-session price-only model, which does not establish useful news prediction beyond the stronger constant reference.

The selected three-session news models remain nearly always-up: FININ made 0/2/0 down calls; averaging made none. The one-day reference made 0/107/0, so the pipeline can predict down. That varying seed did not improve accuracy over always-up.

Older-news interventions preserve current news, prices, availability and headline counts; permutations moved over 98% of available older bundles. All 24 controls changed zero directions and only slightly changed probabilities. Their loss changes were mixed, providing little evidence of useful dependence on correctly dated older content.

**Learning works, generalization remains weak.** Timeline/mapping/gradient/masking/replay gates passed, and both balanced batches of 32 examples reached 100%, including fixed prices. Those checks included disclosed padding perturbations. Longer fitting lowered training loss and worsened validation loss; late averaging runs nearly memorized training while validation loss exceeded 5.5.

All paired descriptive block intervals include zero. Reused validation/checkpoint selection prevents independent generalization claims. Historical inputs also increase active capacity and expose missingness patterns; this is a history-package comparison, not isolated semantics, diffusion or decay. Daily concatenation is our declared extension of [FININ Eq. 7](https://aclanthology.org/2024.findings-emnlp.189.pdf), not a recovered author implementation. Date-only availability, recurring headlines across splits and pretrained overlap remain unresolved.

Protocol, saved predictions and independent review: `artifacts/experiments/history_comparison_2026-10-12/`. All 694 preserved files remain unchanged.

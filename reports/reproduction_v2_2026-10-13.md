# FININ rebuild: what changed and what the evidence supports

**The rebuilt implementation can learn controlled rules, but this study did not establish a useful news forecasting model. The modest development lead failed its fixed final evaluation.**

Implemented the shared projection, separate numerical/fusion encoders, news self-attention, market-query aggregation and predictor from [FININ, Eqs. 1–7](https://aclanthology.org/2024.findings-emnlp.189.pdf). These parameters were trained; RoBERTa and sentiment encoding stayed frozen. FININ freezes its language model, so omitted fine-tuning is not a confirmed defect. This is an architectural adaptation, not exact author replication.

## Executed comparison

Eight prespecified stocks from the [StockNet author dataset](https://github.com/yumoxu/stocknet-dataset), with actual UTC timestamps: preprocessed **tweets**, not Reuters headlines. Compare post-close and overnight messages, cap 64, one input window, all-day next-session closing direction. Retain empty-news dates; avoid future-return filtering. Typical retained counts are three post-close messages and one overnight message.

Train: 2014–June 2015, with 3,000/3,008 eligible observations. Validation: July–September 2015. Final: October–December 2015, **512 observations across 64 shared sessions** per schedule. Preprocessing is training-only. Actual RoBERTa study: 36 neural runs, six matched no-news controls and linear probes. Freeze candidates before final inference. Primary: post-close linear text loss versus coverage and priors; neural comparisons are secondary.

## Final results

Post-close; neural rows average the three seeds' logits. Lower loss is better; AUC0.50 is chance ranking.

| Frozen model | Log loss | AUC |
|---|---:|---:|
| Training-prevalence constant | 0.69313 | 0.500 |
| Stock-specific prior | 0.69298 | 0.523 |
| Ticker + news-count control | 0.69305 | 0.512 |
| Linear text + sentiment + controls | 0.71050 | 0.501 |
| Full FININ, raw vectors | 0.69369 | 0.479 |
| Full FININ, PCA32 vectors | 0.69315 | 0.507 |
| Architecturally matched no-news | 0.69357 | 0.493 |

The primary linear model's validation AUC **0.557** did not persist. Final loss versus coverage is **+0.01745**, descriptive 95% date-block interval **[−0.00479, +0.04019]**. It failed qualification; uncertainty does not prove text universally harmful. Raw FININ is also slightly worse than matched no-news. Attention-versus-pooling comparisons remain inconclusive.

## Learning and news contribution

Raw financial training learns little under this budget. Post-close full/seed 17 ends at training loss **0.6923**. PCA improves training fit to **0.5597**, but validation deteriorates to **0.8073**: fitting without generalization. PCA changes conditioning and capacity together.

Unseen artificial rules using actual RoBERTa vectors reach **93–98%** balanced accuracy; shuffled labels reach **45%**. This establishes learnability under the control optimizer, not adequate financial optimization.

News is **not inert**: disruption changes **170/512** linear-model decisions, versus **2/512** raw-FININ decisions. Several overnight models have positive disruption-loss intervals; full PCA gives **+0.000318 [0.000055, 0.000556]**. These correlated, uncorrected diagnostics support some dependence on dated text, not baseline superiority or information diffusion.

A price/calibration lead remains: matched no-news has mean within-stock AUC **0.5832**, despite pooled AUC **0.4934**. A post-result date-block interval is **[0.5326, 0.6346]**, without search correction. Stock probability offsets may hide ranking, but calibration is an untested explanation. This does not rescue the news claim.

## Decision and audit limits

Preserve this negative benchmark. Next, investigate stock probability offsets with a simple price baseline and calibration on training/development only, before adding news. Assess richer timestamped financial news separately. Future confirmation needs fresh periods; do not tune on this final quarter.

Sparse tweets, one quarter, revised prices, pretrained overlap and unspecified author settings limit conclusions. Published FININ aggregates do not certify prediction behavior.

Correction: the first rebuild cache was mistakenly named RoBERTa while containing BGE384. It is preserved; the corrected study verifies pinned RoBERTa768. An earlier failed final attempt decoded source without predictions: final dates were previously unsealed but score-unused. The skill and both specialists were used. Independent review checked source, 82 metric sets and artifacts. Read-only behavior was observed but not sandbox-enforced. **57 tests pass; all 3,255 pre-existing files are unchanged.**

Protocol, correction, frozen choices, checkpoints, predictions, final results and checksums: `artifacts/rebuild/reproduction_v2_2026-10-13/corrected_roberta/`.

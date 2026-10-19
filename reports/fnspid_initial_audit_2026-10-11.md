# FNSPID initial audit: dataset decision

**Decision: do not switch to FNSPID or bulk-download it now. Keep NIFTY/SPY as the existing daily reference for the next controlled pipeline checks.** This is a practical adoption decision, not evidence that NIFTY is inherently cleaner or more predictive. FNSPID remains a candidate for a separately validated, source-filtered cohort.

## What was actually inspected

The official [Hugging Face release](https://huggingface.co/datasets/Zihan1004/FNSPID/tree/bf9189c41527198897d1af3e17b1a0095279fc45) was pinned to revision `bf9189c41527198897d1af3e17b1a0095279fc45`. Ten deterministic byte ranges across both news CSVs transferred **4,325,376 bytes (4.3 MB)**. Offline parsing recovered **6,838 records / 6,670 unique stripped URLs**. Raw bytes, hashes, protocol, script and source checks are retained in `artifacts/diagnostics/fnspid_initial_audit_2026-10-11/`.

These are convenience samples at file locations, not random article samples or complete chronological cohorts. Counts cannot estimate dataset-wide quality or usable coverage. The initial parser missed blank-index external records in the larger file; an offline v2 correction retains the original outputs and recovers them without downloading again.

## Findings that matter

| Observation in this sample | Research implication |
|---|---|
| 226 of 227 Nasdaq-source records have midnight times | Timestamp-shaped strings mostly do not supply publication-clock precision here. |
| 168 identical records occur in both sampled file tails | Do not concatenate the official files without deduplication. No cross-split leakage was demonstrated. |
| Mixed sources include Russian general news, repeated wire headlines and missing ticker fields | Source/relevance filtering and mapping need validation. Missing bodies alone are acceptable for headline-based FININ. |

Two direct checks strengthen the timing concern:

- A released Agilent record says **December 16, 2023, 23:00 UTC**; its [Nasdaq source](https://www.nasdaq.com/articles/interesting-a-put-and-call-options-for-august-2024) currently displays **December 19, 10:35 EST / 15:35 UTC**: a **64h35m discrepancy**.
- A released Benzinga record says **June 5, 2020, 06:30:54 UTC**; [explicit source metadata](https://www.benzinga.com/news/20/06/16190091/stocks-that-hit-52-week-highs-on-friday) says **14:30:54 UTC**: **eight hours later**. Both fall on the same date, so this alone does not invalidate daily use.

Current source pages are not historical archives; causes and original first availability remain unresolved. The author's [timezone converter](https://github.com/Zdong104/FNSPID_Financial_News_Dataset/blob/main/data_processor/preprocess.py) also uses reversed EST/EDT offsets. Do not apply a universal correction. A healthcare source check confirms the calendar date and genuine OCUL mention, demonstrating that not every investigated mapping/date is wrong.

## Why retain the current setup for now?

NIFTY already targets SPY and has pinned local provenance. Its date-only news, upstream filtering and coverage changes remain limitations; poor learning has not proved dataset inadequacy. Neither dataset reproduces FININ's Reuters/TRNA data. FNSPID adds useful URLs and potential coverage, but this audit has not established a cleaner, complete SPY cohort. Its prepared sentiment aggregates already introduce decay; adopting those would confound our later decay experiment.

Both independent research children recommend withholding adoption. Their review prevented treating sampled missingness as global rates or calling NIFTY superior. Official files are unpartitioned; author examples are small but not verified release subsets, and third-party Parquet conversion lineage remains unverified.

**Next:** bounded train/validation learning checks on the current pipeline, then individually controlled representation/cap/history comparisons. Revisit FNSPID only with a predeclared continuous cohort, defensible source-specific dates/cutoff, deduplication and measured coverage. No training, price ZIP download, installation or existing research changes occurred. Effective child read-only sandbox enforcement was unavailable; behavior was read-only.

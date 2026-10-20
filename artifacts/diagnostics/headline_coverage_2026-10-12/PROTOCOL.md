# Headline coverage audit

Task 1 only: quantify the information budget of the existing 16-headline cap and recommend one bounded larger-cap comparison. Do not train, encode/download features, change selection rules or overwrite existing data/code/results/reports. Use local pinned source news and retained processed dates; verify source/processed hashes and exact agreement of current selected IDs/text with `select_headlines`.

Count nonblank trimmed headline lines, exact within-day unique strings, duplicate removals, retained16 and excluded unique occurrences per date. Compare candidate32/64/128/all caps with nested deterministic hash selection. Report headline-weighted retention, mean per-day retention, distribution quantiles, fraction of dates affected and split/year changes; extreme-volume days must not be presented as typical. Raw source rows excluded by chronological preparation should be shown separately rather than silently mixed with actual experiment dates.

Summarize source test headline counts descriptively to expose coverage shift, without inspecting target labels, predictions or performance or using test results to tune a cap. Recommend a candidate from training coverage and validation feasibility, not accuracy. No gain can be inferred from counts.

Estimate required new unique headline IDs for each candidate on TRAIN/VALID ONLY, recognizing text/sentiment cache IDs reused across dates. Verify old BGE text and TinyBERT cache/index integrity; count missing inputs for both channels, do not generate them. Existing per-date market information/lookback remains unchanged in any later experiment. Record provenance, daily/yearly counts and independent reviewer verification. Preserve all old files checked; concise report only.

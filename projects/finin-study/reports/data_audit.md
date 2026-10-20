# NIFTY raw data audit

Source: [https://huggingface.co/datasets/raeidsaqur/NIFTY](https://huggingface.co/datasets/raeidsaqur/NIFTY)
Pinned revision: `9b8aef736cbaf6a7e9645ac20e8bd9b2344734d2`
Downloaded (UTC): `2026-09-30T09:41:55+00:00`

This audit describes the downloaded source files only. It does **not** establish that the published labels match FININ's binary target or that all headlines were available at a trading cutoff.

| Split | Rows | First date | Last date | Headlines total | Min/day | Median/day | Max/day |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| train | 1,477 | 2010-01-06 | 2017-06-27 | 411,237 | 11 | 78 | 1627 |
| valid | 317 | 2017-06-28 | 2019-02-12 | 17,034 | 18 | 54 | 93 |
| test | 317 | 2019-02-13 | 2020-09-21 | 8,299 | 7 | 26 | 46 |

## Data quality checks

### train

- Labels: `{"Fall": 307, "Neutral": 788, "Rise": 382}`
- Empty-news dates: 0
- Duplicate headline strings within a date: 18,932
- Context fields with only a header/one line: 1,477
- Market-history rows embedded in conversation prompt (min/median/max): 1/6/8
- Prompts with no embedded market-history rows: 0
- Missing `pct_change`: 0
- Weekend dates: 0

### valid

- Labels: `{"Fall": 53, "Neutral": 189, "Rise": 75}`
- Empty-news dates: 0
- Duplicate headline strings within a date: 242
- Context fields with only a header/one line: 317
- Market-history rows embedded in conversation prompt (min/median/max): 4/6/8
- Prompts with no embedded market-history rows: 0
- Missing `pct_change`: 0
- Weekend dates: 0

### test

- Labels: `{"Fall": 73, "Neutral": 143, "Rise": 101}`
- Empty-news dates: 0
- Duplicate headline strings within a date: 1
- Context fields with only a header/one line: 317
- Market-history rows embedded in conversation prompt (min/median/max): 5/6/8
- Prompts with no embedded market-history rows: 0
- Missing `pct_change`: 0
- Weekend dates: 0

## Interpretation for the next step

- Dates are ordered and non-overlapping across splits (the audit would fail otherwise).
- The revised plan specifies exact within-day deduplication and a deterministic cap of 16 headlines per example.
- The separate full-data price/target check runs in `scripts/03_prepare_data.py`; this raw audit does not rerun it.
- That check found all 2,111 supplied returns match next-trading-day SPY returns. Earlier documentation claiming current-day alignment was incorrect. The three-class label still differs from FININ's binary target.
- The revised plan derives binary targets from a separate verified SPY price snapshot and uses previous-trading-day news. Individual headline release-time availability remains uncertain.
- Every standalone `context` field contains only a header/one line. Market-history rows are instead embedded in the `conversations` prompt.
- Headline volume falls sharply from training to testing. The experiment must report this coverage shift and should compare models on identical dates.

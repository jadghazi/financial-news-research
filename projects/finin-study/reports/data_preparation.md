# Stage 1: prepared examples

The input is a dated set of NIFTY headlines plus SPY prices through the forecast day's close. The answer is whether the following trading session closes higher.

| Split | Examples | Selected headline slots | Up targets |
| --- | ---: | ---: | ---: |
| train | 1,475 | 23,592 | 807 |
| valid | 315 | 5,040 | 172 |
| test | 317 | 5,041 | 184 |

## What one example means

The headline text is selected by a fixed SHA-256 rule after whitespace trimming and exact within-day deduplication. The model never sees the future closing price, the NIFTY answer field, or the NIFTY prompt.

| News date | Forecast date: last available prices | Target date: answer observed | Close at forecast | Close at target | Target up? | Headlines selected |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 2010-01-06 | 2010-01-07 | 2010-01-08 | 114.19 | 114.57 | 1 | 16 |
| 2017-06-28 | 2017-06-29 | 2017-06-30 | 241.35 | 241.80 | 1 | 16 |
| 2019-02-13 | 2019-02-14 | 2019-02-15 | 274.38 | 277.37 | 1 | 16 |

## Checks

- All 2,111 supplied NIFTY returns were checked against the frozen SPY price calendar. They align with the next trading session. Binary targets were calculated independently for this delayed-news experiment.
- Forecast and target dates use the real trading calendar. Date ordering and every binary target were asserted for all retained examples.
- Four examples were excluded because their target dates cross into the next source split. The exclusions and file checksums are in `data/processed/manifest.json`.
- Six market features are standardized using training examples only. The mean and scale are in `data/processed/manifest.json`.
- NIFTY provides dates without individual headline release times. We use headlines dated the previous trading day, but cannot prove precise historical publication availability.

Next: the measured [feature benchmark](feature_benchmark.json) covers 100 headlines. All 32,286 unique selected headlines were then cached and used in the [held-out experiments](results.md).

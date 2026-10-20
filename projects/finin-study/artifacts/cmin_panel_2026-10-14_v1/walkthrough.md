# Worked pipeline example

Fixed example: AAPL, first fold, seed17, first evaluation target. Selection did not use the outcome or model score. Attention values are computational diagnostics, not causal importance.

Forecast cutoff: 2020-04-30T20:00:00+00:00. Target session: 2020-05-01. Stored neural UP probability: 0.4738.

Final MLP input has 160 values: 80 market values and 80 news-summary values. Sentiment is fused with headlines earlier; learned weights create the summaries.

| Input session | Headlines retained | News-summary width |
|---|---:|---:|
| 2020-04-24 | 6 | 16 |
| 2020-04-27 | 6 | 16 |
| 2020-04-28 | 5 | 16 |
| 2020-04-29 | 6 | 16 |
| 2020-04-30 | 6 | 16 |

Most recent news bucket, in retained source order:

- Weight 0.165; source-date proxy age 68.0h: Dow Jones Erases 375-Point Gain On Earnings; Tech Giants AMD, Alphabet Set To Report, As Apple Approaches Buy Point
- Weight 0.136; source-date proxy age 68.0h: Expert who called the 2008 crisis says the signal to sell stocks is coming soon
- Weight 0.181; source-date proxy age 68.0h: Autos Are Driving in the Chip Industry's Blind Spot
- Weight 0.159; source-date proxy age 68.0h: Dow Jones Gives Up Gains As Plunging Consumer Confidence Weighs On Stock Market
- Weight 0.202; source-date proxy age 68.0h: Germany Taps SAP, Deutsche Telekom for Contact Tracing App
- Weight 0.156; source-date proxy age 68.0h: Apple Earnings Preview: How Will the Coronavirus Impact iPhone Sales?

Self-attention matrices, market-query scores, projected/fused vectors, daily summaries and the exact MLP input are saved in `fold_0/finin_17/trace.npz`. Source IDs map through `daily_arrays.npz`, `items.json` and `selected_stories.jsonl`. Changing the direct market block, query weighting, sentiment or age pooling is a separate retrained comparison.

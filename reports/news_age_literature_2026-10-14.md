# News age and temporal weighting: bounded literature assessment

Date: 2026-10-14. Literature assessment only; no training, model inference, data acquisition or tuning. The financial-ml-research workflow and existing read-only specialist roles were used. Runtime permissions do not enforce their read-only configuration; review conduct was read-only.

## Direct comparison of exponential weighting

Axel Groß-Klußmann (2024), **Learning deep news sentiment representations for macro-finance**, *Digital Finance* 6, 341–377. [Publisher full text](https://link.springer.com/article/10.1007/s42521-024-00107-2); [version-of-record PDF](https://www.econstor.eu/bitstream/10419/316977/1/42521_2024_Article_107.pdf).

Inspected §§3.4 and 4.3, Eq. 7 and Tables 6–7. The study generates learned daily macro-news sentiment representations with rolling chronological training, then applies exponential moving averages with different half-lives to lagged trading signals. Table 6 compares half-lives from 1 to 90 days. For the equal-weighted supervised-autoencoder sentiment strategy, the reported information ratio is 0.54 at a two-day half-life and 0.15 at 90 days. These are performance ratios, not percentages or prediction accuracies. The authors report stronger sentiment-strategy results with short half-lives.

Limits: this weights aggregate sentiment representations, not individual headline attention. Trading assumes zero transaction costs. The highest-performing half-life summarizes a reported evaluation grid, rather than an independently validated optimum. It does not estimate the economic half-life of a news event, or isolate exponential weighting against an otherwise matched no-decay control.

## Indirect evidence of delayed and persistent responses

Steven L. Heston and Nitish R. Sinha (2016), **News versus Sentiment: Predicting Stock Returns from News Stories**, FEDS 2016-048, DOI 10.17016/FEDS.2016.048. [Federal Reserve record](https://www.federalreserve.gov/econres/feds/news-versus-sentiment-predicting-stock-returns-from-news-stories.htm); [inspected working-paper full text](https://www.federalreserve.gov/econresdata/feds/2016/files/2016048pap.pdf). Findings below refer to this working-paper version.

Inspected §§2–4, Table 5 and Figure 5. In their sample, daily news-based portfolio formation predicts returns for about 1–2 days, while weekly formation produces longer predictability, up to 13 weeks. Positive and negative sentiment have different temporal patterns: negative sentiment has more persistent predictability. Their analysis distinguishes news presence from sentiment, and examines earnings-announcement timing.

Limits: these are cross-sectional return/portfolio and lag-regression results, not FININ's next-session index-direction task. Weekly formation changes aggregation and selection; it is not an isolated intervention on the age of one fixed headline. Not every negative weekly lag is significant. The paper neither tests an exponential decay rule nor identifies a transferable half-life.

## Defensible meeting claim

> Published evidence makes temporal news weighting a reasonable research question. One study reports better results from short-half-life exponential smoothing of aggregate sentiment, while another reports delayed and more persistent responses to negative news. Together, these findings motivate testing context-dependent temporal weighting; they do not establish that all news loses predictive usefulness monotonically at one universal rate.

This is a literature synthesis, not our empirical discovery. Neither study establishes causality from our pipeline or supports choosing a particular decay rate for CMIN. FININ uses multiple daily market/news representations but does not explicitly measure an age-decay curve; changing its lookback changes both histories. Its long-memory interpretation therefore does not resolve this question.

Our completed public-data pilot remains separate: [48-hour weighting experiment](cmin_news_decay_2026-10-14.md) and [older-news addition experiment](cmin_older_news_2026-10-14.md) found no robust, practically useful benefit in their tested settings. These are not replications or refutations of the above papers.

Independent narrow review directly checked the exponential-weighting design and the sentiment-dependent lag evidence. It supported the limited synthesis and flagged grid selection, zero costs, aggregate-versus-headline weighting, and task differences. Saved predictions or author implementations were not reproduced during this assessment.

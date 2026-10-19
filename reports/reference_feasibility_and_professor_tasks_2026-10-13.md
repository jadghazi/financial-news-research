# Reference feasibility and professor tasks - 13 October 2026

**Decision:** freeze further dataset/model adaptations. No candidate in this bounded screen is verified as a suitable public headline-based next-session forecasting reproduction. This is a limited feasibility result, not a claim that no such reference exists. The next priority is establishing access and an auditable reference specification, before more financial fits.

## Existing evidence

The latest SPNews experiment is computationally functional but has unresolved historical news availability, small training/selection blocks, and unstable generalization. Independent read-only recalculation of 18 actual full/no-news evaluation prediction sets confirms mean log loss 0.753862 versus 0.706848. Full models initially predict only DOWN, then both directions; the current pipeline is not always UP. News-sensitive forecasts and low training loss do not establish useful forecasting. Existing evidence and code remain valuable infrastructure.

Sources: `reports/public_headline_pipeline_2026-10-13.md`; `artifacts/rebuild/finin_headline_screen_2026-10-13/{configuration.json,AMENDMENT_02_final_coverage_windows.md,fits/}`; `artifacts/rebuild/spnews_cohort_2026-10-13/primary/provenance.json`. Previously inspected evaluation periods remain exploratory.

## Bounded reference screen

| Reference | Accessible evidence | Adoption decision |
|---|---|---|
| [FININ, Wang/Cohen/Ma, Findings EMNLP 2024](https://aclanthology.org/2024.findings-emnlp.189.pdf), sections 3-5, Table 2, Appendix A/B | Published method; original Reuters/TRNA cohort, author implementation and per-date predictions were not verified accessible in this assessment. University access is unknown. | Preferred route if the professor can establish access and resolve implementation/evaluation details. Data access alone would not recover the exact author experiment. |
| [FNSPID, Dong/Fan/Peng, 2024](https://arxiv.org/pdf/2402.06698), sections 3-5/7; [author release](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) | Headlines and preprocessing code. Reference uses 50-day inputs and three-day future-price regression with MAE/MSE/R2. Released EST/EDT conversion routine reverses UTC offsets; hosted-data lineage is unresolved. Existing local source audit also finds timing discrepancies. | Potential data source requiring audit, not a ready direction-prediction baseline. Do not bulk-download or switch now. |
| [StockNet, Xu/Cohen, ACL 2018](https://aclanthology.org/P18-1183/), sections 3/6; [author code](https://github.com/yumoxu/stocknet-code), [data](https://github.com/yumoxu/stocknet-dataset) | Timestamped tweets, prices, chronological splits and original implementation. Legacy Python2.7/TF1.4; paper/code message-token limits disagree. Outcome-based filtering removes near-zero returns. | Conditional procedural reproduction reference if tweets are acceptable. It is a different population/task from all-session headline forecasting. Existing FININ-on-StockNet adaptation did not reproduce original StockNet. No new run recommended now. |
| [Astock, Zou et al., FinNLP 2022](https://aclanthology.org/2022.finnlp-1.24/), sections 3-4; [author release](https://github.com/JinanZou/Astock) | Stock-specific Chinese news and author release; random development split and return-ranked label selection; later 2021 evaluation. Full release completeness/checkpoints not verified. | Does not clear the temporal reference gate in this assessment. |

FININ Table 2 reports S&P accuracy 0.518-0.537 across lookbacks versus always-buy 0.520. Its principal claimed advantage is Sharpe, not consistently strong majority-class accuracy. Appendix B's correctness-based flag and daily risk-free-rate specification require clarification before economic replication. Published aggregates cannot establish author class-prediction behavior.

## Exact professor questions and future controlled comparisons

Page 1 of `output/pdf/FININ_meeting_brief.pdf` labels step 4 **Weight + summarize**, followed by step 5 **Predict direction**. In the current default code, headline vectors and sentiment are fused per item; self-attention refines these; market-query weights form daily news summaries. The final MLP receives concatenated market-history vectors and daily news-summary vectors. Sentiment is embedded upstream; attention weights act on news vectors and are not separate final input columns. See `src/models/research_finin.py:52-78` and `src/models/paper_finin.py`.

| Question | Proposed comparison | Evidence to report |
|---|---|---|
| Are headline relationships useful? | Self-attention on/off with query weighting retained; query weighting on/off separately. Match dates, information and selection budgets; disclose capacity differences. | Paired held-out loss differences across periods; selected headline-pair examples and attention maps as diagnostics. Current full-versus-mean removes both mechanisms together. |
| Does news importance decay? | Fixed, predeclared news-age bands/lags with unchanged price history and coverage controls; uniform versus exponential age pooling under the same pooled architecture. | Curve of incremental held-out probability loss against news age, with uncertainty and counts. Predeclare publication-age versus availability-age semantics. No expectation of monotonicity; information diffusion can produce delayed peaks. |
| What reaches and matters in the final predictor? | Trace market branch and weighted news summary; matched retrained text/sentiment/channel ablations. Use saved-checkpoint interventions only as model-dependence diagnostics. | Exact tensor/feature map and held-out contribution comparisons. Examine whether weighted summarization discards diversity before proposing one justified addition. |

Channel ablations must account for both paths: removing the direct market input to the predictor can leave market information in the query that weights news. Removing sentiment values also retains biased encoder layers, so the exact intervention must be documented.

Fixed decay applied to concatenated day slots can be absorbed by a trainable linear layer; the implemented pooled comparison avoids that particular confound. A plotted exponential weighting rule is an assumption, not a measured diffusion curve. Perturbations can leave the training distribution; attention maps do not identify causal market mechanisms.

## Gate and stopping rule

First establish one accessible reference with pinned data/code, reconstructable publication/availability timing, cutoff, labels, chronological splits, deduplication, sufficient temporal coverage, and declared implementation discrepancies. For FININ, the professor can first check institutional Reuters/LSEG access and author resources; no messages were sent on the user's behalf.

Only after this gate, propose one frozen, explicitly costed reproduction/comparison budget. Include training-prevalence probabilities, constant directions, price-only, and simple news pooling on matched dates; retain predictions and use dependence-aware paired uncertainty. Specify the untouched evaluation and stop rule before opening results. Do not automatically switch datasets or expand tuning after disappointing outcomes.

A valid research pipeline does not require a positive news effect. Positive evidence is required to claim forecasting improvement or support a mechanism. If resource/timing requirements cannot be satisfied, report that the desired empirical reproduction is unavailable and limit work to architecture analysis or explicitly conditional exploratory results.

Assessment used the financial-ml-research and PDF skills, a bounded literature researcher and an independent skeptical reviewer. Specialist behavior was read-only, but session permissions did not enforce that restriction. No training, model inference, new tests, large downloads, dependency installation, or historical-artifact edits were performed. One meeting-brief page was rendered for inspection; this new report records the decision.

---
name: financial-ml-research
description: Investigate financial ML research questions, verify literature, design controlled temporal experiments, and critically assess findings. Use for methodology, replication, baselines, ablations, robustness, and academic reporting; scale the workflow down for routine edits.
---

# Financial ML research workflow

Use this workflow alongside the repository's `AGENTS.md`. Its purpose is a defensible research answer, including an inconclusive or negative one. Adapt the depth to the question: a factual clarification may need one source check; a new empirical claim needs an explicit protocol and an evidence trail. Do not run training, broad searches or infrastructure setup merely because the skill is invoked.

## Frame the question

Identify the research decision, target population/market, forecast cutoff, horizon, and available evidence. Separate architectural reproduction, empirical replication, independent extension, and mechanism testing. Read relevant code, manifests and prior runs before assuming a documented method was executed. In this repository, use the root map to locate evidence; preserve version 1 as the historical reference.

State a falsifiable hypothesis with its comparator, primary metric, evaluation unit and a practically meaningful effect. Specify what would weaken or reject it. For example: market-query weighting reduces held-out log loss relative to matched mean pooling across temporal windows, rather than merely increasing accuracy on an up-heavy sample. Choose the effect criterion before inspecting new outcomes; do not adopt an arbitrary universal significance threshold.

List plausible competing explanations and the observation that would discriminate them. For FININ, distinguish delayed information diffusion from longer price history, representation/capacity differences, aggregation noise, coverage shifts and chance. Record missing professor requirements as unknown; do not invent assignments from recommendations in an old report.

## Verify literature and theory

Search and read relevant primary papers, appendices, author code and dataset/model documentation. Verify title, authors, venue/year, DOI or stable identifier, and version; a dataset card's citation block is not sufficient. Follow cited work when it materially affects the research choice. If access is incomplete, identify the unread material and limit the claim accordingly.

Keep a compact comparison of candidate methods: assumptions/mechanism, inputs and availability, evaluation protocol, reported evidence, reproducibility, and relevance to this question. Include credible contrary findings and simpler explanations. Separate a source's interpretation from its measured results and from our inference. Map implementation decisions to equations or code when replication matters; treat unspecified details as choices, not recovered author defaults. Inspect metric formulas and contradictory tables as critically as model descriptions.

## Design the experiment before running it

Write a short dated protocol in the experiment record: hypothesis, data/version, cutoff/horizon, exclusions, chronological splits/windows, fit/selection rules, primary metric, baselines/ablations, seeds, uncertainty method, compute budget and stopping conditions. Record later amendments and whether results were already visible. Existing inspected test dates are exploratory evidence; use a genuinely untouched period or disclose retrospective temporal validation for new claims.

Audit the information timeline end to end. Check timestamp/timezone semantics, actual trading sessions, missing dates, upstream filtering/truncation, duplicate or syndicated stories across splits, changing coverage, point-in-time constituents and adjusted-price vintages. Audit pretraining/fine-tuning overlap where feasible; absence of documentation is unresolved provenance, not proof of cleanliness. Draw feature and label intervals and apply justified purging/embargo where overlap requires it. Fit scaling, selection, calibration and thresholds inside training/validation only. Shuffling training minibatches is compatible with fixed chronological evaluation splits.

Use constant direction and training-prevalence probability references, a price-only model, and relevant simple news summaries. Add a lightweight linear model when it helps distinguish nonlinear capacity from information value. Match samples, preprocessing, information, optimization/selection budgets and evaluation across comparisons; disclose remaining capacity differences. In FININ, mean pooling removes both attention mechanisms; no-self-attention keeps market weighting; zeroed sentiment removes that information but retains biased encoder layers. Count active capacity rather than assuming equal parameter totals prove a controlled ablation.

Change one interpretable factor at a time when possible. Compare longer-news-history effects against longer-price-history controls; choose ablations that isolate the claimed mechanism. Estimate runtime/storage and begin with small integrity checks within the authorized scope. Do not substitute a broad hyperparameter sweep for a research question.

## Execute and retain reproducible evidence

Before running, assign a distinct experiment ID/output namespace and ensure commands will not overwrite previous artifacts. Record the Git revision and dirty diff, full configuration, dependency/hardware versions, data and encoder revisions/hashes, preprocessing/tokenizer/pooling settings, split/exclusion manifests, seeds, commands and timings. Check cache validity against both content and generating code/settings; a checksum verifies identity, not methodological correctness.

Check aligned labels, finite features, masking, gradient flow and basic learnability as relevant. These checks establish computational integrity, not predictive signal. Select checkpoints with the declared validation rule; retain learning histories, per-date predictions, checkpoints, failures and all attempted comparisons. Diagnose problems first on training/validation evidence. Label post-result diagnostics and audit-added baselines retrospectively rather than calling them preregistered. Verify reported tables against saved predictions.

## Assess evidence and challenge the result

Report effect sizes against matched baselines, class-specific behavior, probability quality and uncertainty. Separate variation from seeds from uncertainty across market periods. Use paired comparisons on identical forecast dates; choose block resampling or another justified dependence-aware method when serial/overlapping outcomes make independent-sample intervals invalid. Do not count seeds as independent market histories. Account for multiple comparisons, model searches and test reuse; qualify weak power and instability.

Probe relevant temporal windows/regimes, selection caps, news lags, representation/sentiment choices and reasonable parameter sensitivity in proportion to the claim. Investigate collapse, contradictory seeds/windows and failures rather than pooling them away. Challenge apparent news gains with informative negative controls, such as date-disrupted news within a permissible split or label randomization, explaining what each control diagnoses and preserving temporal boundaries. Select checks by their ability to distinguish explanations, not by their likelihood of producing a favorable result.

Treat attention inspection as a diagnostic. Prediction improvements alone cannot identify a causal news mechanism. For economic claims, specify executable positions, entry/exit timing, costs/slippage, return convention, risk-free-rate units and Sharpe annualization. Reconcile the FININ Appendix B formulas explicitly; classification accuracy and retrospectively available news do not establish tradable returns.

## Report and decide the next step

Deliver the question, verified method/source mapping, protocol deviations, results with comparators and uncertainty, strongest competing explanation, negative findings, limitations, and a prioritized next discriminating test. Link exact literature and local experiment evidence; distinguish newly executed checks from saved historical audits. Report unsupported or inconclusive conclusions plainly, without claiming either replication or refutation beyond the tested scope.

Maintain a compact research record, for example:

`ID/date | question/hypothesis | protocol/provenance | command/artifacts | result/uncertainty | alternatives/limitations | next test`

Use existing project records where suitable. Avoid mandatory new documents or full literature reviews for trivial changes. Finish with what the evidence supports, what remains unresolved, and what observation would change the research decision.

# Goal 1: isolated LLM evaluation

**2026-11-02 interface update:** release 0.5.0 connects a selected headline search directly to Predict and separate History/scoring. Exact inputs remain frozen; no tools/search/price labels were added. A real API call using an explicitly synthetic headline saved UP, validating mechanics only. Historical text/weight uncertainty and the unfinished held-out evaluation remain. See README/VALIDATION; older planning entries below are preserved.

**Status:** initial headline/date experiment interface and client-side no-tool runner implemented (0.4.0); full isolation and a held-out scientific evaluation remain incomplete. **Updated:** 2026-10-25. See the latest session record and VALIDATION.md; older planning entries below are preserved as history.

**Sequencing note, 2026-10-23:** scraper architecture is the current priority. See [its temporal/export contract](SCRAPER_ARCHITECTURE.md) and the [model-cutoff audit](SOURCE_AUDIT.md). Gemma 4 and a 90-day buffer were ideas in the supplied brief, not a completed model or decontamination decision. The collector remains independent of the future model and label definition; revisit this plan when the prediction stage begins.

## Objective and claim boundary

Build an inspectable way to give a general-purpose LLM a permitted news packet, deny external information during inference, record an up/down forecast, and evaluate it against a precisely defined later outcome.

The original aspiration, "make sure the model was not trained on the news, market, or ticker," needs qualification. We cannot erase or fully inspect the training history of an off-the-shelf model by sandboxing it. Familiarity with a company is also different from knowing the outcome of a particular event. The defensible goal is to control what reaches inference and minimize, test, and disclose potential knowledge of future outcomes.

| Risk | Proposed control to investigate | What it does not establish |
|---|---|---|
| External information retrieved during a run | Disable tools/retrieval; isolate context, files, and network access as appropriate; verify controls | Absence of information in model weights |
| Knowledge of historical outcomes | Frozen model established before evaluation events; prospective predictions; qualified post-cutoff historical tests | A vendor cutoff alone does not certify the complete training history |
| Future information in the news packet | Versioned source evidence and availability timestamps; reject later revisions | A current page's publication date does not establish its historical text |
| Leakage through development or scoring | Separate inference and labels; chronological splits; freeze protocol before test | A random split or a polished explanation does not establish validity |

## Questions to resolve first

- Are we measuring firm-level movement, an index, or another instrument? Which market, language, universe-selection rule, and independently chosen dates?
- Is the input one headline, a full article, or a bundle available at a fixed decision time? Are names/tickers/dates provided? What counts as "news only"?
- Is the target initial market reaction, or movement after the prediction can actually be made? Which return interval and zero-return rule? Choose one primary target before testing.
- Is a prospective collection period feasible? What budget, hardware, API access, acceptable run time, and institutional data access are available?
- Which interface exposes sufficient control and version evidence: a chat interface, an API, or locally hosted frozen weights? Evaluate the actual configuration rather than the product name. Codex as a development assistant is separate from the model being evaluated.

Resolve these through focused discussion and research during the next stage. They are not defaults inherited from the example papers or FININ.

## Research and design steps

- [ ] **1. Review relevant methods.** Search for news-to-return forecasting, financial LLM memorization, temporal contamination, prospective evaluation, and negative results. Critically review the papers that could influence the design, including their model versions, timing, baselines, and data access requirements. Produce an evidence comparison and identify which claims we can realistically test.
- [ ] **2. Specify the target.** Write the unit of prediction, entity mapping, eligible universe, permitted input fields, decision timestamp, return endpoints, price source and convention, calendar, zero-return policy, exclusions, and missing-data handling. Distinguish article-level and firm/day-level outcomes.
- [ ] **3. Compare isolation and contamination designs.** Assess prospective frozen-model evaluation, historical evaluation after an adequately evidenced model training window, and historical diagnostics with masking. Compare the available interfaces for actual context, tool, logging, network, and version controls. Record remaining uncertainty and select a bounded prototype.
- [ ] **4. Agree the collector/export contract.** With Goal 2, define immutable record and version IDs, availability evidence, eligible fields, and packet hashes. Specify how unreliable times and updated stories are excluded or quarantined. Do not assume the collector can reconstruct historical availability it never observed.
- [ ] **5. Prototype the runtime boundary.** Build the smallest runner that consumes a frozen packet, validates the output, records the request/response, and has no access to the scoring dataset. Check blocked network/tool/file paths and fresh-session behavior. Run adversarial news-text cases and deliberate leakage fixtures. Keep operational evidence of the boundary.
- [ ] **6. Freeze development and evaluation rules.** Select prompts and output schema on development data; set primary metrics, baselines, uncertainty method, sample-size rationale, collection/stopping rule, variant budget, and error handling before observing held-out outcomes. Seal the protocol and input manifests.
- [ ] **7. Run a small end-to-end pilot.** Verify timing and label calculations manually on a sample; inspect failures, cost, latency, and logging. Use the pilot to check feasibility, not to advertise predictive power. Changes informed by pilot outcomes need a fresh held-out evaluation.
- [ ] **8. Evaluate and hand over.** Run only the frozen protocol on the eligible evaluation sample; record every attempt and exclusion. Report qualified results with reproducible artifacts and a clean-environment runbook. Preserve the first predictions; later replays are distinct runs.

The steps are checkpoints for evidence, not automatic requests for user approval. Implement them when the user starts that stage; ask only where missing information changes the scientific claim or resource commitment.

## Candidate design: recommendation to test

Prefer a prospective test if its calendar and cost are feasible: establish an identifiable frozen model before collecting evaluation events; capture news as it arrives; record predictions before the target return interval; obtain labels only after they mature. This is our methodological proposal, not a literature result or an adopted provider choice.

Keep historical work useful for development and diagnostics, with its own contamination assessment. Identifier/date masking can test sensitivity to remembered context, but events may still be recognizable and masking can change their meaning. Synthetic news can test mechanics, not establish real-market forecasting ability. Do not call either technique proof of decontamination.

For hosted inference, blocking all network access would also block the model request. Investigate a narrowly permitted inference endpoint, no model tools/retrieval, an isolated client, and documented provider/version behavior. The client cannot directly audit everything happening inside a hosted service. For local inference, evaluate an already-downloaded frozen model with networking disabled and minimal readable files. Do not choose local hosting or a provider until the tradeoffs are measured.

## Timing and evidence contract to validate

Keep distinct timestamps for source publication, source update, first observation, retrieval completion, packet cutoff, request start, prediction completion, and label interval. Preserve source timezone and uncertainty; normalize comparable times to UTC.

For a prospective forecast, require every permitted input version to be acquired by the decision cutoff, the packet cutoff to precede the request, and prediction completion to precede the target return interval. Account for polling, extraction, inference latency, market holidays, daylight-saving changes, and after-hours news. A headline describing a price move that already occurred cannot be counted as forecasting that same move.

For historical tests, record the separate evidence supporting text availability at the simulated cutoff. Download time today is neither proof of historical availability nor a substitute for it. Exclude or separately label uncertain cases.

Each run should record model/provider identity and version evidence (or weight/tokenizer hashes), run mode, code revision including local changes, environment/configuration versions, full prompts and allowed parameters, input/version IDs and hashes, raw request/response, parsed prediction, timing, retries, errors, token usage, and cost where available. Keep audit metadata and labels outside the model's permitted fields. Separate development and evaluation sessions; preserve original outputs because hosted reruns may differ even with the same settings.

## Evaluation requirements to specify

- Use chronological development/validation/test periods or a predeclared forward test. Keep overlapping target intervals and near-duplicate/event families from leaking across development and evaluation boundaries; define boundary exclusions explicitly.
- Include simple no-news baselines, such as a direction chosen using development data only, and an interpretable text-only baseline. Add a matched no-news LLM control and consider shuffled/masked input controls where they answer a specific question. Do not require an embedding pipeline or import one from FININ.
- Choose one primary directional metric; report class balance, confusion counts, balanced accuracy and/or macro-F1 as appropriate. If eliciting probabilities, define their meaning and evaluate calibration/Brier or log loss. Report abstention coverage and failed-call rates using predeclared denominator rules.
- Justify sample size and confidence intervals using the dependence structure across time, firms, and repeated stories. Evaluate uncertainty in differences from baselines, not just an isolated accuracy estimate. Limit exploratory variants and label them honestly.
- Decide how to handle corporate actions, delistings, stale prices, missing news, and universe changes without using hindsight. Keep price data for scoring separate from inference.
- Do not equate sentiment with subsequent return direction, portfolio-day hit rate with individual-stock accuracy, or classification accuracy with tradable profit. Trading evaluation, if later requested, needs a feasible execution time and explicit costs, turnover, and exposure controls.

## Completion evidence

Goal 1 is ready for research use when a colleague can reproduce the setup and an offline example from this project alone; inspect the isolation checks and residual limitations; trace each forecast to its exact permitted packet and model configuration; and reproduce the scoring without exposing labels to inference. Include demonstrated rejection of future-dated inputs, later revisions, unapproved context, and label access. A functioning container without those checks does not complete this goal.

## Initial reference scan: leads, not selections

Sources checked on 2026-10-20. This scan establishes possible relevance only; no listed method has been adopted. Expand and update the search during actual work.

| Lead and reading depth | Possible relevance and required scrutiny |
|---|---|
| Lopez-Lira & Tang, [Can ChatGPT Forecast Stock Price Movements?](https://arxiv.org/abs/2304.07619v6), v6 (2025); supplied PDF opening and selected timing/data passages checked, PDF pp. 1-4, 13 | Distinguishes initial reaction from subsequent drift. Check full design and model snapshot evidence; headline performance numbers must not be transferred to our different target. |
| Glasserman & Lin, [Assessing Look-Ahead Bias in Stock Return Predictions Generated By GPT Sentiment Analysis](https://arxiv.org/abs/2309.17322), v1 (2023); abstract screened | Investigates historical overlap and identifier removal. Read methods before choosing masking controls or accepting its conclusions for other models. |
| Zhu et al., [From Knowing to Doing](https://arxiv.org/abs/2605.28359), preprint (2026); abstract screened | Recent lead on consistent masking and attribution. Trading-agent setting differs from a news-only classifier; masking is not proof of no memory. |
| White et al., [LiveBench: A Challenging, Contamination-Limited LLM Benchmark](https://arxiv.org/abs/2406.19314v2), v2 (2025); abstract screened | Lead on refreshed tests and objective scoring, not evidence of financial predictive power. |
| Li et al., [FinKario](https://aclanthology.org/2026.acl-long.446/), ACL 2026; supplied PDF opening and selected setup passage checked, PDF pp. 1-2, 6 | Optional reference on event organization. Its graph/retrieval and enriched context differ from our proposed restriction; do not adopt that architecture by association. |
| Zhao & Welsch, [Point-in-Time Financial RAG with Frozen LLMs and Market-Feedback Adaptive Retrieval](https://arxiv.org/abs/2605.31201v2), preprint v2 (2026); abstract screened | Lead for timing/provenance analysis. Its retrieval, market context, and feedback exceed the current news-only scope; inspect safeguards rather than importing its system. |

## Decisions and session record

| Date | Question / alternatives | Status and rationale | Revisit trigger |
|---|---|---|---|
| 2026-10-20 | Sandbox alone vs separate runtime, training, input-timing, and design controls | Adopted as a planning distinction; runtime restrictions cannot erase learned information | Refine controls after interface/model research |
| 2026-10-20 | Prospective vs qualified historical evaluation | Prospective preferred provisionally; feasibility and required claim remain unresolved | Discuss calendar, budget, and target; complete methods review |

**Completed this session:** project boundaries, initial plan, limited reference scan. No implementation, isolation verification, or performance evaluation.

**Next action:** define the intended forecast and feasible study calendar with the user, then critically review timing/contamination methods and compare candidate execution interfaces against that claim. Record the chosen design and alternatives here before building it.

**2026-10-25 methods follow-up:** at the user's request, checked relevant methods and appendix sections in both supplied PDFs, the Glasserman/Lin primary methods, and a version-pinned FinKario code spot check. See [methods review and evidence limits](LLM_EVALUATION_METHODS_REVIEW.md). Post-cutoff dates plus no external retrieval are useful controls, not sufficient evidence of an uncontaminated historical test. Lopez-Lira/Tang use fixed-model headline requests and RavenPack-guided timing/relevance; Online Appendix C explicitly acknowledges possible later fine-tuning exposure. Their portfolio hit rates do not transfer to individual BABA forecast accuracy. FinKario uses richer reports/graphs; its next-day execution lag is not a training-contamination control, and current public code presents a potential within-week future-document risk requiring further audit before drawing conclusions about published results. No architecture or checkpoint was adopted and no runner, collection, labels or forecasts were executed. The user's cleared collector stays empty. Next: define the return interval/decision cutoff and evaluate exact checkpoint training/release evidence before building the inference stage; a forward holdout is a proposed strengthening, not a replacement for the historical objective.

**2026-10-25 model-access follow-up:** checked official Gemma 4 hosted availability/free pricing, account-specific quota guidance, the January 2025 pretraining claim and separate instruction-tuning description. Documented the conflicting Search-grounding guide/pricing entries and the need for explicit tool exclusion. Shortlisted hosted Gemma 4 31B IT for convenience and original pre-event Gemma 3/2 checkpoints for a stronger historical chronology argument; no model/provider was adopted. Exact quotas, hosted version stability, checkpoint provenance and local feasibility remain unverified. Evidence and next checks are in the same methods review. No account/key access, inference calls, model download, new collection or experiments occurred.

**2026-10-25 experiment-interface discussion:** user requests a proposed clean interface connected to the collector, allowing pasted/collected headlines and 5/10/20-day comparisons. User explicitly chose **calendar days including weekends** and **prediction-date opening to closing price** (Friday morning to Friday evening), with news from before the prediction date. These replace the corresponding open preferences, but model/revision, dates, scopes, labels and scoring policies remain unselected. Created [Experiments interface proposal](EXPERIMENTS_INTERFACE_PLAN.md): one-prediction exploration plus frozen batch studies, separate model configuration/progress/scoring/reporting, exact permitted packet previews, and repeated-run stability distinct from date-level accuracy. No UI/runner was implemented or experiment executed in this planning follow-up. Suggested next bounded increment: packet preview and synthetic practice, followed by a one-packet provider check once model/access choices are settled; retain held-out dates for reported accuracy.

**2026-10-25 authorized implementation:** user approved the minimal interface, headline/date pairs only, Gemma 4 31B API and small tests; supplied a key and quota screenshot. Added Experiments with one-date prediction, batch 5/10/20-calendar-day comparisons, frozen packets, conservative sequential API budgeting, persistent pause/resume and separate manual/CSV opening/closing-price scoring. Actual requests omit all tools/search/context/history; responses containing tool calls/grounding are rejected. This is a client-side application boundary, not a completed OS/file sandbox or proof of provider-internal isolation. Strict observed and explicitly exploratory historical/manual availability remain distinct. The first valid output is retained; repeat accuracy uses try 1 with agreement separately, and report comparisons use shared dates. No actual historical stock-forecast performance was measured: synthetic fixtures and a small live synthetic API check validate mechanics only. No automatic price provider or trading workflow was added. User data remains empty; validation evidence lives in ignored tmp/. See VALIDATION.md for final counts and browser evidence. Next: user collects/imports their own dated headlines, previews packets, and chooses a development/held-out calendar before reporting accuracy. Goal 1 is not complete as a contamination-free evaluation environment.

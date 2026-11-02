# Experiments interface and evaluation proposal

Updated: 2026-10-25. The initial text below records the design proposal. The user's subsequent request authorized the scoped implementation in release 0.4.0: three experiment tasks inside the existing dashboard, headline/date input, Gemma 4 31B, conservative quotas, separate price entry/import and saved reports. See README.md for the actual workflow and VALIDATION.md for checks. Proposed features below are not automatically implemented; full process/file sandboxing, no-news LLM control, provider-internal isolation and automated price acquisition are not implemented. Existing active collections and studies remain cleared; test evidence is separate.

## Confirmed choices and remaining choices

Confirmed directly by the user on 2026-10-25:

- Start with BABA's NYSE-listed ADSs.
- Input windows: **5, 10 and 20 calendar days**, including weekend news.
- Target: the prediction date's **opening price to closing price**. For a Friday, UP means Friday's close exceeds Friday's open; DOWN means it is below. Thursday's closing price is not the starting price for this target.
- The explained starting workflow uses news from **before the prediction date**, not that morning's news. Make the boundary visible, not an implicit date conversion.

Still proposed/open: exact evaluation dates and model revision; genuine held-out development/test split; source scopes and historical availability policy; input size; sampling settings; repeated-run count and aggregation; zero-return handling; label source; account quota/credential handling. Gemma 4 31B IT hosted via the Gemini API is a candidate, not a verified connected model or adopted uncontaminated checkpoint. See [methods evidence](LLM_EVALUATION_METHODS_REVIEW.md).

## Interface recommendation

Add **Experiments** as a main sidebar section in the existing local Streamlit app. Reuse the design and selected ticker, but keep collection and experiments independently usable. Do not add prediction controls inside collection jobs or make collecting require a model account.

| Screen | What the user sees and does |
|---|---|
| Model connection | Provider, exact requested model ID, configuration/connection status and last checked time. A key is entered locally, masked and excluded from reports/logs. Show quota settings and whether they were entered manually or verified. Describe tools as disabled in the request, without claiming to audit all provider internals. A connection check must be explicit and contain no financial outcomes. |
| Try one prediction | Choose a target trading date, paste dated headlines or import a saved packet, or load saved collector news. Preview the exact permitted text, its source/timing flags and cutoff. Press **Run prediction** and inspect UP/DOWN plus a short reason. Untimestamped manual text is allowed as a clearly exploratory input, without claiming historical validity. |
| Set up a study | Select ticker, target dates, 5/10/20 calendar-day windows, headline/body mode and news scopes. Choose fixed protocol settings and see the call estimate, eligible dates, missing-news/ambiguous-time counts and any packet-size problems before freezing the study. Initially recommend headline inputs and one run per date/window. |
| Run progress | **Start study**, **Pause**, **Resume** and **Stop**; completed/total calls, current target/window, pending/error/quota states and last activity. Stop prevents new calls; an already submitted provider call may finish. Results already recorded remain saved. No automatic provider/model substitution. |
| Report | A row per window showing scored target dates, correct/incorrect predictions, accuracy, balanced accuracy, baseline comparison and an appropriate interval. Show shared-date comparisons, UP/DOWN label balance, coverage and errors. Open a date to see each packet and prediction. Repeated-run agreement, if enabled, gets its own panel. |
| Download | Results CSV, original response/attempt records, frozen permitted packets, protocol, model/version evidence and manifest, plus a concise readable report with limitations. No credentials or authorization headers. |

Use human-readable labels and help text. Do not present a fictitious connected model, invented quota, synthetic accuracy or an unverified 'leakage-free' badge. A synthetic practice mode must be visually distinct and cannot contribute to study metrics.

## Timing contract

For target date D, use XNYS and `America/New_York`. Proposed packet cutoff C is **00:00 at the start of D**. Window N includes local publication/availability evidence in `[C minus N local calendar dates, C)`. For a Friday, the input ends at the boundary between Thursday and Friday. Weekend news belongs in the interval; subtract calendar dates in the local timezone rather than N times 24 UTC hours around DST changes. Reject holidays/non-sessions as target dates; the session's real close covers early-close days.

Keep these separate:

- actual collection time and exact text-version evidence;
- simulated historical cutoff/target date;
- actual inference request/response times;
- actual future-market label interval, or historical interval being evaluated retrospectively.

Never backdate a request made today to a 2025 cutoff. A historical run remains retrospective. For forward runs, require prediction completion before the target session opens. A later completion is recorded as late/ineligible rather than silently counted.

The collector's `get_news(..., policy='observed'|'historical')` is an input selection boundary, not a ready-made fully safe inference prompt. `observed` requires records actually available to this collector by cutoff; today's backfill cannot pass that condition for 2025. `historical` remains explicitly exploratory where text-version evidence is absent. New packet code must enforce exact interval bounds in addition to date-level filters, handle date-only/timezone-uncertain sources explicitly, and allowlist fields before sending anything to the model.

## What a comparison means

The unit of accuracy is **one ticker and target date per window**. Three windows on 100 target dates produce 300 predictions, but only 100 distinct target outcomes. Multiple headlines on one date are inputs, not independent forecast observations. Different windows share labels and often most of their headlines; adjacent dates can share news too.

Recommended first pass:

1. Fix chronological development and held-out periods before examining held-out outcomes.
2. Freeze one prompt/model/settings set, one prediction per target/window and the source/text policy. Keep all three windows predeclared; do not repeatedly edit them after seeing held-out accuracy. If a primary window is required for a confirmatory claim, select it on development data; otherwise report the three comparisons together with multiplicity considerations.
3. Compare windows on **the same eligible target dates** for a paired comparison. Also report full coverage for each window so missing dates cannot disappear from the report. Empty or oversized packets need predeclared dispositions; they must not silently become no-news predictions or different truncation policies across windows.
4. Use a simple prespecified constant-direction baseline and a matched no-news LLM control to assess whether the supplied news helps. A development-majority baseline is possible when development labels are available. Do not assume 50% is the appropriate empirical baseline.
5. Keep every failure, invalid output and missing label visible. Report scored accuracy and coverage together; define exclusions/abstentions before the study. Proposed exact-zero returns are FLAT with a separate count, excluded from the binary metric; this remains a policy to freeze before scoring.
6. Report uncertainty using target-date dependence, such as an appropriately configured block bootstrap, and paired differences between windows. Do not make the sample size three times larger by treating overlapping windows or reruns as independent dates. The interval method/block choice needs to be fixed in the statistical protocol.

**Repeated tries:** repeated responses to an identical packet measure stability and sampling variation. If requested, configure a fixed repeat count before running; save all outputs and report agreement or a predeclared vote rule. Never keep rerunning until the direction is correct, select the best seed or count reruns as extra independent days. Retries after transport failures are attempt records, not new target observations. A one-run-per-window study is a sensible economical first increment, not a prohibition on a later repeated-run analysis.

## Packet, labels and execution architecture

Use one frontend and separate services inside this project:

`collector/manual import -> packet builder -> frozen study -> model worker -> immutable predictions`

`independent price-label import -> scorer reads predictions + labels -> report`

- **Packet builder:** load explicitly selected collector records/imported data, preserve provenance, choose eligible versions, apply a fixed source/dedup/order policy and render only permitted fields. Keep immutable packet IDs/hashes and exact sent text. News is untrusted data, not tool instructions. Dates/ticker and selected headline text can be permitted; source audit scores, outcome labels, later annotations and unrelated metadata cannot silently enter prompts. Do not create summaries with a separately informed assistant.
- **Study storage:** an experiment-specific database/directory under ignored project data, separate from collector jobs. Freeze the selected collector record contents instead of relying on IDs that might disappear after a collector reset. Manual changes create a new packet/study version. Preserve the originals and all attempts.
- **Model adapter:** explicit provider interface; initial hosted candidate may be Gemma 4 via Google. Fix the requested model, record returned version evidence, prompts and supported settings; explicitly omit all tools, search, URL context and external retrieval. Fresh request context per packet. No tools can be granted by source text. The model client must not have scoring-label access; the exact process/file/network boundary still needs design and tests. Hosted server internals cannot be fully audited by this client.
- **Output:** validate a small schema with UP/DOWN and an optional short reason. Invalid/refused responses are explicit states. A confidence/probability field, if later included, is an elicited estimate requiring separate calibration evaluation, not a verified probability or evidence of correctness. Do not ask for hidden chain-of-thought.
- **Queue:** bounded concurrency, configured RPM/TPM/RPD allowances, bounded retries/backoff and persistent checkpoints. Quota exhaustion waits/pauses visibly. Count packet tokens and reject an oversized single request rather than silently changing it or assuming waiting solves a per-request/input-limit problem. Record original and retry responses. An experimental rerun is distinct from resuming pending calls.
- **Labels:** begin with a separate validated CSV import (`ticker`, session date, open, close, source/provenance), rather than silently adopting a price provider. Check real XNYS sessions, positive finite prices, unique keys and consistent prices/security mapping. Labels are needed to compute accuracy but never enter the inference packet. Automating a public price source is a separate audited implementation choice. No price download was performed for this proposal.
- **Secrets:** do not include keys in snapshots, manifests, source control or exception strings; do not ask the user to paste a key in chat. Local key entry does not change the collector's account-free operation.

No model accuracy is assumed from general reasoning benchmarks. The original headline paper classifies individual headlines and builds portfolios; our date-bundle/window comparison is a distinct experiment. The primary report must not inherit the paper's portfolio hit rates as single-stock expected accuracy. See [methods review](LLM_EVALUATION_METHODS_REVIEW.md), and official [Gemma API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) / [quota guidance](https://ai.google.dev/gemini-api/docs/rate-limits), checked 2026-10-25. This is an engineering proposal based on those reviewed controls, not a claim that the cited paper validates this exact UI or protocol.

## Bounded implementation order and acceptance criteria

1. **Packet preview and synthetic practice:** validate window boundaries, DST/weekends/early closes, missing dates, manual provenance and an allowlisted prompt; show exact text and real empty-state behavior. Keep practice data outside real study metrics.
2. **One-packet provider adapter:** after model/credential selection, first validate request shape and no-tool configuration with a fake local HTTP service and adversarial fixtures. An explicitly initiated live check establishes availability/response handling, not complete server isolation. Save and validate outputs/errors, redact keys, record version uncertainty.
3. **Frozen batch study:** shared dates, persistent stop/resume, no duplicate completed calls after restart, bounded quotas, reproducible packet hashes and a fixed repeat policy. Do not add automatic model substitution.
4. **Separate label import and scoring:** hand-check synthetic known labels, flat/missing cases, denominators, paired comparisons and interval behavior. Verify the inference component cannot read the label file. A fake endpoint should see exactly the allowed packet and no outcomes.
5. **Professor-facing verification:** actual browser review of model status, input preview, start/stop/resume and downloads; meaningful automated regressions, clean-environment setup, a synthetic offline walkthrough and an honest live validation record. Use a small pilot to fix mechanics, then retain a separate held-out period for reported accuracy.

Next action: agree/choose remaining model, dates and label-source details, then implement the first bounded increment. This proposal introduces no live collection, API/model calls, labels or forecasts.

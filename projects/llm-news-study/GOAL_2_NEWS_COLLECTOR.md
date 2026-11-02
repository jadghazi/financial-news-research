# Goal 2: reliable, reusable news collection

**2026-11-02 handover:** [HANDOVER.md](HANDOVER.md) documents fresh-clone setup, first use, local state, references and the code map. Superseded dashboard modules and unreachable pages were removed; their two used helpers were retained. The 95-test suite passed after cleanup. Git excludes the user corpus, predictions, key, environments and local PDFs, which remain preserved on disk. The next research step remains a broader relevance/coverage review and a separately frozen evaluation protocol.

**2026-11-02 current update:** user-authorized headline-only product repair, release 0.5.0. Results now select one collection/publication range before deduplication; out-of-window evidence is retained. Added bounded CNBC/TechNode historical discovery and Alibaba Cloud announcements. Everyday UI is Headlines → Predict → History, source details separate. Only the supplied FININ paper's source description was checked; no FININ experiment resumed. Current validation/counts are in VALIDATION.md; older entries below are history, not the current UI.

**Status:** working initial collector/dashboard; bounded end-to-end validation completed. **Updated:** 2026-10-25. See [README.md](README.md) for the runnable workflow and [VALIDATION.md](VALIDATION.md) for evidence and limits. This is not a claim of exhaustive historical coverage or a completed statistical extraction/relevance benchmark.

The [current architecture](SCRAPER_ARCHITECTURE.md), [claim audit](SOURCE_AUDIT.md), and [decision log](DECISIONS.md) refine the initial plan below. Where an earlier open choice differs from the architecture, use the dated architecture/decision entry. This file tracks progress rather than duplicating those specifications.

## Objective

Build a collector the professor can use independently to discover, retrieve, inspect, and export reliable financial news. Support the LLM study through a versioned export contract, without requiring an LLM account or experiment runner for ordinary collection.

"Robust" means documented source coverage, reliable recovery, faithful text, explicit timestamp semantics, traceable origins, and an understandable user workflow. It does not mean a universal scraper that can access any site. "Modern" means we investigate current techniques and select them using evidence, not that every source must go through a new framework or an LLM.

## Questions to resolve first

- Which market, languages, companies, event types, and historical depth are needed? Is live collection, backfill, or both the priority?
- Which free sources permit the intended research, retention and professor reuse? Zero paid-service budget and a 16 GB Windows laptop are the current design constraints.
- Does the professor prefer a simple command, notebook, local dashboard, or another interface? Which operating systems must we support?
- Does the study need headlines only, full text, or both? Which retained material can legally be shared in the repository or with collaborators?

The supplied brief motivates an initial US-company pilot, with the first ticker still to be chosen. Keep source/company configuration extensible. The proposed starting point is one permitted SEC/IR path and a complementary source; the source audit makes Massive conditional on access/use evidence.

## Source research

Build a source register before promising coverage. Investigate each source for authority, relevance, publication behavior, language, historical access, revision behavior, timestamp precision, delivery latency, stable identifiers, extraction difficulty, access conditions, cost, and maintenance risk.

| Source family | Role worth testing | Risks and distinctions to preserve |
|---|---|---|
| Aggregators and news APIs/feeds | Broad discovery and potential archive access | Discovery time is not necessarily publication time; syndicated copies, summaries, coverage gaps, and usage limits vary |
| Company investor-relations/newsroom sites | Primary corporate announcements | Company-controlled narrative; inconsistent feeds, layouts, timezones, and backfill; may duplicate wire releases |
| Regulators and exchanges, including SEC candidates | Official filings, enforcement, and market announcements | SEC-authored press releases differ from issuer filings hosted by SEC; authority of hosting is not independent verification of issuer claims |
| Established publishers/newswires with permitted access | Independent reporting and broader context | Paywalls, licenses, archive limits, corrections, syndication, and redistribution restrictions |

For each concrete source, record a canonical homepage/endpoint, original publisher, route (API/feed/HTML/browser), identifier strategy, timestamp fields and their meanings, expected coverage, access-policy references and check date, storage/export permissions, and known gaps. Keep origin, discovery service, and retrieval service distinct. Measure bias toward companies that publish often or are easiest to scrape; multiple feeds do not automatically yield representative coverage.

## Research and implementation steps

- [ ] **1. Establish requirements and source shortlist.** Research scholarly dataset-construction methods, source documentation, and current acquisition/extraction tools. Identify credible alternatives and produce a source matrix with access, latency, history, licensing, and timestamp limitations.
- [ ] **2. Define a benchmark and export contract.** Select an allowed, representative sample across source types, page layouts, languages, updates, and failure cases. Manually establish reference text/metadata. Agree with Goal 1 on timing and version evidence before optimizing throughput.
- [ ] **3. Compare acquisition and extraction approaches.** Benchmark supported APIs/feeds, ordinary HTTP plus deterministic extraction, browser rendering where needed, and current crawler/LLM extraction candidates. Measure on the same eligible pages and document failures, operating cost, and maintenance. Choose a stack only after this comparison.
- [ ] **4. Implement one vertical slice.** Retrieve a small allowed source, preserve the raw response and provenance, normalize records, inspect quality, and export a documented dataset. Keep provider-specific logic in a source adapter with a stable internal contract. Verify the output independently before adding sources.
- [ ] **5. Add complementary source families.** Expand based on demonstrated research value. Handle entity mapping, syndication, updates, and availability evidence consistently across adapters. Produce source coverage and exclusion reports.
- [ ] **6. Demonstrate recovery and quality controls.** Test interruption/resume, repeated runs, throttling, malformed pages/feeds, changed layouts, missing times, unavailable sources, and broken redirects. Make source failures visible without losing successful records or silently declaring a partial run complete.
- [ ] **7. Add the chosen user interface.** Let a user configure a collection, preview source/coverage expectations, run it, inspect progress and problems, review samples, and export. Select the interface from the professor's needs and pilot evidence rather than building a dashboard by default.
- [ ] **8. Verify handover.** Test a clean installation, an offline example, and a small permitted live run. Document configuration, adding a source, export fields, common failures, recovery, costs, and coverage limitations. Validate a collector export with the Goal 1 consumer separately.

These steps are evidence checkpoints. The present documentation task does not start crawling, provision services, or install a stack.

## Candidate approaches to compare

The layers solve different jobs; a fair comparison may combine them rather than declare one library a universal winner.

| Layer | Candidates to investigate | Evidence needed to select |
|---|---|---|
| Discovery/acquisition | Official APIs, RSS/Atom, documented archives, permitted publisher access | Coverage, provenance, latency, backfill behavior, terms, and failure visibility |
| Crawl orchestration | Minimal HTTP/feed workflow versus a crawler such as [Crawlee Python](https://github.com/apify/crawlee-python) | Queue persistence, retry/rate controls, resumability, operational complexity on our sources |
| Article extraction | Explicit source parsers and an extractor such as [Trafilatura](https://trafilatura.readthedocs.io/en/latest/) | Faithfulness, boilerplate removal, headline/time preservation, languages, and layout variation |
| Rendered/AI-oriented acquisition | Browser-based adapters; a candidate such as [Crawl4AI](https://docs.crawl4ai.com/) | Incremental extraction gains, latency, memory, reproducibility, and source-policy fit |
| Optional LLM-assisted extraction | Constrained, schema-validated extraction from saved source evidence | Value beyond deterministic methods, invented-field rate, provenance, cost, and run variability |

Initial official documentation/repository pages were inspected on 2026-10-20 only to identify candidates. No comparative quality, compatibility, or current version claim is established. Recheck documentation, maintenance, dependencies, and licenses at selection time; add stronger alternatives found in subsequent research.

LLM-generated text must never silently replace the source wording. If used, preserve its input, model/configuration and output; keep derived summaries/entity links separate from raw and extracted text. The prediction model must receive only fields explicitly permitted by Goal 1.

## Draft record and availability contract

Validate this contract with the source pilot; do not implement it as an unexplained fixed schema.

| Group | Information to retain |
|---|---|
| Identity and origin | Stable record ID, version ID, source item/accession ID where provided, requested/final/canonical URLs, original publisher, discovery source, retrieval route, source family, language |
| Content and evidence | Source headline/body, extracted headline/body, content type, raw snapshot reference and hash, extracted-content hash, extraction method/version, schema version, response status and useful headers |
| Time | Raw timestamp strings; interpreted publication/update times with timezone, precision, and evidence; first-seen time; fetch start/completion; version observation time; uncertainty and availability policy |
| Entities and relationships | Candidate entity identifiers, matching evidence/confidence/method, effective-date considerations, exact duplicate/syndication/event-cluster relationships |
| Quality and access | Missing/ambiguous-field flags, extraction validation, eligibility/quarantine reason, source-policy reference/check date, retention/export restrictions |

An item, its revisions, and an event cluster are different things. Never overwrite the initial text when a source changes it. Keep first observation separate from later re-fetches; use hashes to detect changes and new versions to record them. Duplicate clustering may evolve, so version that mapping too.

Do not infer an exact publication time from a date-only page, filesystem time, a URL, or an LLM guess. Document parser rules and uncertain timezone assumptions. For prospective prediction packets, conservatively use evidence of when a specific content version was observed and ready for use; never backdate a later revision. Historical backfills require separate availability evidence or an explicit limitation.

Preserve distinct raw records for syndicated copies even when deduplicating an export. Distinguish exact duplicates, syndication, later updates, and separate reporting about the same event. Do not merge stories just because they share a ticker. Flag ambiguous tickers and name changes rather than forcing matches; choose the matching policy on validation examples.

## Reliability and quality requirements

- Persistent collection state and checkpoints; repeated runs are idempotent for unchanged versions; an interrupted run can resume without duplicate output or lost evidence.
- Per-source/domain limits, bounded concurrency, timeouts, capped retries with backoff/jitter, and handling of rate-limit responses. Use conditional requests/caching where supported and record cache provenance.
- Explicit handling of redirects, deleted pages, malformed feeds, browser failures, layout changes, empty responses, and partial runs. Quarantine suspicious output and retain a useful error reason; no silent fallbacks that change origin or text.
- Source adapters with regression fixtures and metadata/schema validation. Retain only permitted fixtures; use synthetic or shareable alternatives where necessary and label their limitations.
- A manual audit on representative records for text faithfulness, headline/body separation, timestamp evidence, entity links, and duplicate groups. Define sampling, metrics, and acceptance thresholds before choosing the winning approach.
- Source health and run summaries: discovered/fetched/retained/quarantined/failed counts, reasons, available vs requested dates, latency, retries, extraction completeness, and cost where applicable. Estimate discovery recall only against a meaningful reference set; fetched success rate is not coverage.
- Keep secrets out of logs/exports; protect snapshot parsers and URL/file handling. Treat remote text as data, including instructions embedded in articles. Respect robots/access policies and do not use access-control evasion as a reliability strategy.

## SEC starting leads, if relevant to the chosen market

Official pages checked on 2026-10-20: [EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces), [Developer Resources](https://www.sec.gov/about/developer-resources), and [RSS feeds](https://www.sec.gov/about/rss-feeds). These document submissions/XBRL APIs, filing feeds, and SEC materials. They are access-method leads, not a selected dataset or a substitute for measuring live coverage. An XBRL API is not a general news article feed.

The developer guidance states a maximum of 10 requests per second in aggregate across a user's machines. Recheck the policy at implementation, use a compliant identifying user agent and a conservative shared limiter, and measure dissemination/observation timing instead of treating access limits or typical update delays as a service guarantee. Preserve filing acceptance, source publication, and our observation as distinct times.

## Completion evidence and usability

The professor should be able to install the collector from this project's documented requirements, run a shareable offline example, configure a bounded collection without editing scraper internals, see which sources succeeded or failed, inspect original/extracted records and timestamp flags, resume an interrupted run, and export a documented interoperable format with a run manifest.

Select export formats after learning the workflow; a simple tabular export plus structured provenance is a candidate, not a requirement yet. Provide a source register, tested setup/run commands, configuration examples, output field definitions, troubleshooting, and known coverage/access limitations. Keep offline verification available when a live source changes or credentials are unavailable. Validate live behavior separately and date that evidence.

## Decisions and session record

| Date | Question / alternatives | Status and rationale | Revisit trigger |
|---|---|---|---|
| 2026-10-20 | Collector coupled to model vs independently usable collector | Adopted as a requirement: professor can use collection without an LLM; integrate through an explicit export | Refine contract after requirements and source pilot |
| 2026-10-20 | Framework-first vs source/benchmark-led selection | Adopted as process: investigate current tools and select using source-specific evidence; no framework selected | Complete comparative pilot |

**2026-10-20 completed:** source-family map, draft quality/provenance requirements, limited primary-documentation scan.

**2026-10-23 completed:** audited the supplied brief against primary sources; recorded GDELT recent-query success and an inconclusive historical 429; proposed the worker/storage/UI architecture, versioned temporal policies, export contract, tests and build order. Created a decision log. The user's later clarification permits revising the brief's proposals. No extraction benchmark, account entitlement check, complete backfill or implementation has run.

**2026-10-24 completed:** user authorized building/testing the collector only. Built project-local environment, package/CLI, persistent jobs and source observations, original-response archive, temporal text exports, dashboard and synthetic demo. Tested official feeds/pages for AAPL and NVDA and publisher feed bodies for MSFT; configured arbitrary ticker profiles and optional SEC/GDELT adapters. A separate clean install passed 44 tests; actual browser collection/download and export-manifest integrity were verified. Kept code/data independent of FININ and made no LLM or price requests. The wider architecture's deferred items are listed in VALIDATION.md.

**2026-10-24 source-quality clarification:** the user requests a diverse, relevant corpus rather than issuer feeds alone and selects **BABA's US-listed ADSs**, not Hong Kong shares, as the pilot. The intended later study uses historical prediction dates after a model's stated cutoff and alternative preceding-news windows (5/10/20 days given as examples). Exact dates, decision time and calendar/trading-day windows remain open. See [corpus methods and evidence](CORPUS_RESEARCH.md). Initial engineering tests do not establish corpus adequacy. Source-expansion work is in progress; no LLM work is authorized by this clarification.

**2026-10-24 follow-up completed:** implemented BABA's US ADS profile, configurable 6-K/20-F forms, first-page issuer archive, independent/context source profiles and frozen configurations, relevance/price-recap flags, balanced body budget and shared-article text variants. Added corpus diagnostics and a separate frozen accepted/rejected review sample. Live 2025 checks yielded six eligible issuer HTML bodies and two independent Global Voices bodies; PDF-only release notices were detected and quarantined, while actual PDF requests were disallowed by source robots rules. GDELT recent/historical probes timed out. Optional Fed/BLS feeds worked but do not remedy sparse independent BABA reporting. 62 tests passed in the separate test environment; a real browser download and shared-version IDs were verified. A process restart guard prevents code-fingerprint drift after edits. Details are in VALIDATION.md. No LLM or price work began.

**2026-10-24 next-action proposal, superseded by the no-account instruction below:** broaden free historical discovery with a bounded account/access check for the Media Cloud candidate described in CORPUS_RESEARCH.md, or another demonstrably better free path; measure incremental relevant independent coverage and exact time evidence before adopting it. Existing data is a pilot, not a sufficient 2025 study corpus. Human audit and historical-version evidence remain required. Verify SEC identification/access before enabling it. No paid source or university subscription may be assumed.


## Headline and usability follow-up, 2026-10-25

**Completed:** headline-only collection; expanded BABA brand/context rules; public multi-publisher aggregator, Qwen feed, SiliconANGLE/Rest of World profiles, expanded Global Voices searches, technology/trade-policy notices, optional SEC news and reviewed public Bluesky posts. A broad regulatory pilot was corrected with dated exclusions and a narrower replacement query. Five everyday dashboard sections now expose wrapped headlines, same-page progress and Stop/Resume, and clear downloads. Actual browser screenshots, start/stop/resume and download/mobile checks were performed. Exact results are in VALIDATION.md. No LLM or price-history/label work began.

**Current constraints:** free public sources with no accounts/API keys. Media Cloud is rejected as a dependency. Headlines are the priority; bodies optional. The user's trusted social-source suggestion permits reviewed public accounts but does not justify arbitrary posts or an unsupported X scraper.

**Next action:** review a frozen sample of direct and broader-context headlines, including rejected candidates; measure date/event coverage. Broader 2025 independent financial reporting and historical headline-version evidence remain incomplete. Continue account-free sources and document incremental value without inflating sample size through copies or unrelated notices. SEC live access needs a genuine contact identity. Working software and statistical corpus adequacy remain separate claims.

**2026-10-25 user-requested workspace reset:** the user wants to choose and run their own collections later. Moved the previous `data/` and `exports/` intact into ignored `backups/cleared-workspace-20261025-174631/`; preserved software, preset profiles and any local settings. Initialized a fresh active database and empty exports directory. Verified zero jobs, discoveries, documents, versions, events, source runs, annotations, audits and worker leases, SQLite integrity OK, unchanged archived database SHA256 and zero active export files. No collector process was running. Earlier validation evidence describes the archived pilot. Next action is the user's chosen ticker/date/source collection; do not start another collection automatically.


**2026-10-25 release 0.4.0 integration:** added Results → Experiments and a saved-news adapter producing date/headline pairs. The user-approved optional prediction workspace is documented in Goal 1/README; collection remains independently account-free and retains article options. A synthetic RSS fixture → saved collector → live synthetic Gemma → independent synthetic scoring → report download path was checked outside active user data. Active collection and experiment history remain empty, with no scheduled work; the user's model credential is configured locally. The next real collection/study is the user's choice. No new historical corpus or predictive-performance result was created.

# Newsroom: headlines → UP / DOWN

**Version 0.5.0 · 2 November 2026.** Find dated headlines for a ticker, give them to Gemma 4 31B, and save whether it predicts a close above or below that trading day's opening price. The pilot is **BABA's NYSE-listed ADSs**.

The everyday dashboard uses **headlines only**. Collection is free and needs no source account/key. Model calls optionally use the user's Google key. There is no automated price acquisition, trading or claim of predictive performance.

## Open the app

On this laptop, double-click **Launch.cmd**. Installation is already complete. Keep its console open while using the dashboard.

On another Windows computer, install Python 3.12 with PATH enabled, open **Setup.cmd** once, then **Launch.cmd**. The app runs locally at `http://127.0.0.1:8501`.

To install from GitHub, install Git and Python 3.12, then run:

```powershell
git clone https://github.com/jadghazi/financial-news-research.git
cd financial-news-research\projects\llm-news-study
.\Setup.cmd
.\Launch.cmd
```

If the repository is already cloned, run `git pull --ff-only` from that clone instead of cloning again. For later updates, close this app and its launcher, pull the changes, run Setup.cmd again, then Launch.cmd. Keep your own uncommitted code changes safe before updating.

Each laptop has its own local data and model key. Git transfers the application, documentation and default source/company profiles; it does not synchronize collected headlines, prediction history, locally edited profiles or credentials. Enter the key under **Settings → Model connection** on the new laptop. No GPU is required for Google-hosted predictions. Internet is needed for setup, live news collection and model calls.

Choose **Headlines / Predict / History / Settings** in the sidebar. Select your ticker there. **Offline demo** contains synthetic examples and disables real collection/model calls.

For a professor or new maintainer, start with [HANDOVER.md](HANDOVER.md): first run, saved files, troubleshooting, code map and references.

## 1. Find and review headlines

In **Headlines**, choose publication dates and press **Find headlines**. If results already exist, expand **Search again · change dates or sources**.

- Both boundary dates are included. Instant publication timestamps use **New York dates**; date-only source dates stay date-only.
- The list shows **only the selected search and its requested publication dates**. The latest search opens by default. Older runs remain available in **Saved search**.
- Each row shows **Published / Headline**, with publisher and company/context type underneath. Click a headline to open its source.
- Uncheck **Include related China / industry news** for company news only. **Filter these headlines** narrows the current selection by words/publisher. Relevance is rule-based, not a certified financial materiality score.
- **Stop search** preserves completed work. **Resume this search** continues unfinished work after a stop, failure or dead-worker detection, including a computer restart. An in-flight request may finish before stopping.
- **Clear view** hides the list without deleting evidence or predictions. Select the saved search to restore it.
- **Download headlines** saves only `date,headline` in CSV. **Source coverage & search details** offers original source/URL/timing metadata and a manifest separately.
- **Use these headlines →** opens Predict with this search and its currently filtered headline selection. No download/re-upload is needed.

The **publication date** is when a source says it published a headline. **Collected here** is when this tool retrieved it. Sources can return 2024/2026 items during a 2025 search: those observations/rejection reasons remain auditable, but do not appear in the selected search's dated results.

Public archives can take several minutes; results update automatically. Source failures, unavailable history and request caps are visible under source details. An empty date does not prove that no news existed.

## 2. Predict UP / DOWN

In **Predict**, choose a saved search, or paste/upload your own dated headlines. Choose a US trading date and the preceding **5, 10 or 20 calendar days**. Weekends count in news windows; only exchange sessions can be prediction targets.

Example historical practice run: collect **27 September–16 October 2025**, then predict **17 October 2025** with a 5-, 10- or 20-day window. Historical text/training uncertainty still applies.

Review the list and press **Predict UP / DOWN**. The exact request is under **Exact model request & timing details**. No target-day news is included.

- **UP:** predicts that date's closing price is above its opening price.
- **DOWN:** predicts that date's closing price is below its opening price.

Only **ticker/target instructions and date/headline pairs** enter the request. Prices, labels, URLs, article bodies, source notes and past conversations are excluded. Requests are fresh, with no configured tools/Search/URL context; tool/grounding responses are rejected and tools are never executed. This demonstrates an application-level boundary, not an audit of Google's internals or an OS sandbox.

The result appears on screen and saves automatically. **Pause / Resume / Stop prediction** control unfinished requests. Closing the browser does not stop the worker. Provider/quota errors pause visibly; the app never silently switches models or drops headlines to fit the token budget.

**Compare dates** runs selected windows over a trading-date range. Repeated tries are optional and measure consistency; primary accuracy uses the first try per date/window.

Paste one headline per line:

```text
2025-10-16 -- Alibaba announces a new partnership
15 October 2025 -- Another source headline
```

CSV uses `date,headline`; earlier structured exports with `published_at,headline` also work. Incomplete dates are rejected.

## 3. History and optional scoring

**History** shows saved predictions. **Download predictions & report** preserves exact inputs, outputs, attempts, model/version evidence and the report in a ZIP. Credentials are excluded.

To score predictions, expand **Check accuracy with actual prices**. Enter independently checked US-session opening/closing prices for one date, or download/fill/upload the `date,open,close` template for a batch. Labels are stored separately and never sent to the model. Flat days are excluded from binary accuracy and reported separately. Missing/failed calls remain recorded; repeated tries do not create extra independent market outcomes. Compare windows on shared scored dates.

## Sources

| Source | Contribution and limits |
|---|---|
| **CNBC public news archive** | Company/context headlines and explicit original publication metadata. Bounded sitemap/metadata requests; not exhaustive. No body extraction, Pro/Investing Club pages. |
| **TechNode dated archive** | English China-technology reporting with explicit archive publication timestamps. Bounded monthly pagination; no article-body requests. |
| **Alibaba Cloud press room** | Dated official AI/cloud announcements, including 2025. Issuer-originated, not independent reporting; public index is not guaranteed complete. |
| **Alibaba investor archive** | Official corporate/financial announcements from an incomplete public index. |
| **Qwen feed** | Retained official AI/product announcements, not every Alibaba release. |
| **Publisher feeds** | Global Voices, SiliconANGLE and Rest of World; current feeds and limited tested historical searches. |
| **Stock Analysis news cards** | Multiple-publisher discovery with original links; recent bounded coverage, no subscription archive. |

Optional Federal Register, SEC, macro feeds and reviewed Bluesky posts remain in source settings. They are not substitutes for independent company reporting. SEC needs a real identifying contact email, not an account/key. X is unimplemented. GDELT remains optional with historical availability unverified after failed probes. Denials, robots restrictions and request caps are recorded; controls are not bypassed.

Public access is not a blanket redistribution licence. Preserve attribution and check rights before sharing downloaded corpora. The professor can use collection independently of an LLM account.

## Research limits

FININ used **Thomson Reuters News Analytics**, not these public feeds. Its millions of items over a long period are not an expected single-ticker public-web count. See [the source/method review](CORPUS_RESEARCH.md).

Post-pretraining-cutoff dates and disabled search do not prove no training contamination. Historical headlines fetched today can have changed. These are exploratory historical inputs. The strict observed-by-cutoff option excludes news first collected after a simulated cutoff. Forward predictions recorded before the target session provide stronger timing evidence. Functional checks do not establish accuracy, complete coverage or a research-ready corpus.

## Settings and troubleshooting

- **Settings → Model connection:** save/check the Google key. This laptop already has a locally saved key. Model `gemma-4-31b-it` is fixed; no fallback. Free capacity can change; errors are visible. Hosted requests need internet to reach Google, and free-tier inputs may be used by Google to improve products.
- **Settings → News sources & connections:** source capabilities, contact configuration and worker controls.
- **Settings → Company profiles:** add/edit another ticker independently of BABA.
- **Settings → Run diagnostics:** detailed collection history and recovery.
- After a software update, close the old launcher and reopen **Launch.cmd**. Runtime fingerprints prevent mixing changed code with an old process.
- After a crash, reopen Launch.cmd and resume interrupted work. Use `doctor` below to check database integrity.
- News is in ignored `data/collector.sqlite3` and `data/artifacts/`; predictions in `data/experiments/`; scoring labels are separate. The key in `data/google-credentials.json` is local plaintext, ignored by Git and excluded from downloads.
- Validation fixtures live separately under ignored `tmp/`. Earlier cleared evidence remains under ignored `backups/`. Your own saved searches are preserved.

From this project directory, without activating an environment:

```powershell
.\.venv\Scripts\python.exe -m newscollector doctor
.\.venv\Scripts\python.exe -m newscollector verify
.\.venv\Scripts\python.exe -m newscollector sources
.\.venv\Scripts\python.exe -m newscollector backfill --tickers BABA --source news_sites,archives,cloud,issuer,feeds,publishers,aggregators --since 2025-09-27 --until 2025-10-16 --headlines-only
.\.venv\Scripts\python.exe -m newscollector demo
.\.venv\Scripts\python.exe -m pytest -q
```

Older article/export CLI options remain for compatibility; article controls are removed from the everyday dashboard. Collector backup and experiment ZIP export are separate. Use command `--help` for backup options. This project has its own environment and no runtime dependency on the paused FININ study.

[Validation](VALIDATION.md) · [Collector plan](GOAL_2_NEWS_COLLECTOR.md) · [Evaluation plan](GOAL_1_ISOLATION_AND_EVALUATION.md) · [Decisions](DECISIONS.md) · [Source audit](SOURCE_AUDIT.md) · [Working instructions](AGENTS.md)
Publicly exposed headline/date metadata may be retained when the linked article body is subscription-flagged. The app records that distinction and does not bypass access controls or extract the protected article text. Social posts and general macro-feed collection remain advanced CLI capabilities outside this headline-only screen.

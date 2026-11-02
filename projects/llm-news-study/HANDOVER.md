# Handover: headline collection and UP/DOWN experiments

This is a local research tool for **dated headlines → one stock-direction prediction → separate scoring**. The starting instrument is **BABA's NYSE-listed ADSs**, not its Hong Kong ordinary shares. The everyday interface uses headlines only. It does not acquire prices automatically or place trades.

## Install on another Windows laptop

Tested platform: Windows, Python 3.12. A GPU is unnecessary because predictions use Google's hosted API. Install [Git](https://git-scm.com/downloads/win) and [Python 3.12](https://www.python.org/downloads/windows/), enabling Python's PATH option.

```powershell
git clone https://github.com/jadghazi/financial-news-research.git
cd financial-news-research\projects\llm-news-study
.\Setup.cmd
.\Launch.cmd
```

Setup creates a project-local `.venv`, installs the pinned dependencies and runs a local health check. Keep the Launch console open. The dashboard is at **http://127.0.0.1:8501**. No connection to the paused FININ project is required.

**A new clone starts fresh:** no collected headlines, searches, predictions, scoring labels, API key or personal settings are stored in Git. Default company/source profiles and synthetic offline examples are included. Offline demo is explicitly separate from live collection.

Collection needs no source accounts or keys. For predictions, open **Settings → Model connection**, save your own Google key locally and use **Check connection**. This verifies access to the configured model record, not whether sufficient generation quota is available. Optional SEC collection needs a real identifying contact email under Settings → News sources & connections.

## First complete practice run

1. Select **Live collection**, ticker **BABA**, section **Headlines**.
2. Search publication dates **2025-09-27 through 2025-10-16** with the default sources. Press **Find headlines** once. Historical archives can take several minutes; the page refreshes itself.
3. Wait for a finished status. **Some coverage gaps** means collection ended with incomplete source coverage; inspect **Source coverage & search details** for the actual reasons. An empty date is not evidence that no news existed.
4. Review the dated list. Related China/industry news is included by default and labelled separately. Uncheck that option for company-only inputs. **Download headlines** saves a two-column `date,headline` CSV; it is optional.
5. Press **Use these headlines →**, select target **2025-10-17** and a **5-, 10- or 20-calendar-day** window. Review the exact input and press **Predict UP / DOWN**.
6. Open **History** to inspect the saved result or download the full experiment report. To score it, expand **Check accuracy with actual prices**, enter independently verified regular-session open/close prices, then save. Use matching unadjusted open/close values for this target; do not mix an unadjusted open with a dividend-adjusted close.

This is a historical practice example. The validated 2026-11-02 collection produced 59 headlines with related context enabled, but current source layouts/access can produce different counts. See [VALIDATION.md](VALIDATION.md); the fixtures and validation database are not shipped as a live corpus.

## What the prediction means

For a target Friday, **UP** means Friday's closing price is higher than Friday's opening price; **DOWN** means lower. The 5-day input covers the preceding Sunday through Thursday. Windows are calendar days, including weekend news, and exclude all target-day news. Exchange holidays/weekends cannot be target dates.

The API receives ticker, target instructions and chronological `date -- headline` pairs. It receives **no prices, labels, article bodies, URLs, publisher metadata, collection logs or previous conversation**. The exact instruction and input are saved with the result and are inspectable in the dashboard.

The current configuration is `gemma-4-31b-it`, temperature 0, minimal thinking, at most 512 output tokens, a fresh request and no configured tools/Search. The output is UP/DOWN plus one short reason. Temperature 0 does not guarantee identical reruns. Failed requests remain visible; the client does not silently change models or trim headlines. Repeat tries measure consistency; primary scoring uses try 1 for each date/window. Flat days are reported separately and excluded from binary accuracy.

Application controls do not audit Google's internals or erase learned knowledge. A reported pretraining cutoff does not certify every later training stage. Historical headlines retrieved today may also have changed since publication. This tool is functional; a clean held-out scientific evaluation and claims about predictive accuracy still require a separately frozen protocol. See [the methods review](LLM_EVALUATION_METHODS_REVIEW.md).

## Controls and saved files

| Action/location | Meaning |
|---|---|
| Stop search / Resume this search | Preserve completed observations; stop or continue unfinished collection. An in-flight request may finish. |
| Clear view | Hide the selected search; it does **not** delete results. Restore it from Saved search. |
| Close browser tab | The background worker can continue. Use the dashboard's Stop/Pause controls to stop work. |
| `data/collector.sqlite3`, `data/artifacts/` | Collection state, assessments and compressed original responses. |
| `data/experiments/` | Frozen prediction inputs, outputs, attempts and separately stored scoring labels. |
| `data/companies.json`, `data/sources.json` | Locally edited profiles; defaults are under `src/newscollector/resources/`. |
| `data/google-credentials.json` | Local plaintext key; ignored by Git and excluded from report downloads. |
| `exports/`, `backups/`, `tmp/` | Ignored local exports, backups and validation work. |

To move **history as well as code**, stop collection/prediction jobs and close the app/launcher before privately transferring the complete `data/` directory. A prediction ZIP or headline CSV alone is not a full workspace restore. Copying all of `data/` also transfers its key and any contact configuration; keep that transfer private. Never commit it. Keep active databases on a local disk, not a network/shared sync folder.

For routine updates, close the app/launcher, run `git pull --ff-only` in your clone, rerun Setup.cmd and relaunch. Do not copy a `.venv` between computers. Code fingerprints intentionally reject work from a process still using an older code version. Completed predictions remain preserved with their original provenance; unfinished runs made with changed code may require a new experiment.

## Troubleshooting

| Symptom | Next step |
|---|---|
| Nothing seems to happen after Find headlines | Check the saved search status and changing counts. Source requests are rate-limited. Open source details before starting another search. |
| Very few or zero old headlines | Check selected sources and coverage notes. Recent RSS/aggregator feeds cannot unlock a historical archive merely by filtering old dates. |
| Google quota/access error | Read the displayed error; check Model connection. Resume only when access/quota is available. No alternative model is selected automatically. |
| Code changed / restart required | Close the old launcher and reopen Launch.cmd. Original data is preserved. |
| Computer crashed | Relaunch, choose the saved search and use Resume if offered. Check Settings → Run diagnostics for persistent events. |
| Port 8501 is already in use | Check whether an existing launcher/app is still open before starting another instance. |
| Setup failed | Keep the console output. Verify Python 3.12 and internet access; rerun Setup.cmd. Do not use the FININ environment. |

Local checks, from this project directory:

```powershell
.\.venv\Scripts\python.exe -m newscollector doctor
.\.venv\Scripts\python.exe -m newscollector verify
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite uses isolated synthetic fixtures and mocked provider/source requests. It does not consume the real model key or establish financial performance. Live source probes and the separate synthetic real-API check are documented in VALIDATION.md.

## Maintainer map and references

| File | Responsibility |
|---|---|
| `src/newscollector/dashboard.py`, `workspace.py`, `model_settings.py` | Navigation, everyday workflow and local key configuration. |
| `core.py`, `resources/companies.toml`, `resources/sources.toml` | Validated profiles, timestamp meanings, calendar and entity/context rules. |
| `sources.py`, `archive_sources.py`, `broad_sources.py` | Discovery adapters. `fetch.py` enforces shared request/access controls and preserves raw responses. |
| `store.py`, `worker.py` | Local SQLite state, source checkpoints, collection and recovery. |
| `query.py` | Search/date eligibility, historical versus observed availability, deduplication and export manifests. |
| `experiments.py` | Frozen headline packets, run definitions, separate price-label scoring and reports. |
| `gemma.py`, `experiment_worker.py` | Allowlisted API payload, response validation, quotas and prediction checkpoints. |
| `quality.py`, `tests/` | Corpus diagnostics, frozen relevance reviews and regression checks. |
| `extract.py`, CLI article options | Retained, tested non-default body/PDF functionality; not used by the everyday headline-only flow. |

Source coverage is bounded, not exhaustive. Public access is not a redistribution licence. Respect robots, access terms and rate limits; retain attribution. A subscription-protected body is not fetched through a bypass. Source changes should be checked with a small permitted fixture/live probe before making coverage claims.

Primary implementation references: [Gemma API guide](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api), [exchange_calendars](https://github.com/gerrymanoim/exchange_calendars), [SQLite WAL](https://www.sqlite.org/wal.html), [Streamlit execution model](https://docs.streamlit.io/develop/concepts/architecture/session-state). These are references for specific design choices, not certification of this application.

Research/source reading depth, access dates and limitations are recorded in [CORPUS_RESEARCH.md](CORPUS_RESEARCH.md) and [SOURCE_AUDIT.md](SOURCE_AUDIT.md). Design changes belong in [DECISIONS.md](DECISIONS.md); scientific follow-up remains in the two goal plans. Earlier entries are historical records, not promises that every proposed capability exists. Supplied reference PDFs remain local and are not required to install or run the tool.

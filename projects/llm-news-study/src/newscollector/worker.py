"""Durable single-worker execution. The UI queues work; it never performs a crawl."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from urllib.parse import urlsplit
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta
from pathlib import Path

import psutil

from . import RUNTIME_CODE_SHA256, assert_runtime_current

from .core import Candidate, Company, Request, Settings, assess, digest, now, publication_day, utc
from .extract import extract, linked_pdf
from .fetch import Fetcher, Fetched
from .sources import ADAPTERS
from .store import Store


def enqueue(store: Store, request: Request) -> str:
    assert_runtime_current()
    request.checked()
    companies = store.settings.companies()
    unknown = [x for x in request.tickers if x not in companies]
    if unknown:
        raise ValueError("Configure company names and sources first: " + ", ".join(unknown))
    data = request.model_dump(mode="json")
    data["companies"] = {x: companies[x].model_dump() for x in request.tickers}
    data["source_profiles"] = {key: value.model_dump() for key, value in store.settings.source_profiles().items()}
    data["config_hash"] = digest(data)
    return store.create_job(data)


def discover(store: Store, job_id: str, source_run: int, ticker: str, company: Company, candidate: Candidate, seen: str | None = None) -> int:
    assert_runtime_current()
    seen = seen or now()
    candidate = candidate.model_copy(update={"metadata": {**candidate.metadata, "company_snapshot": company.model_dump()}})
    existing = store.rows("SELECT id FROM discoveries WHERE job_id=? AND source_run=? AND ticker=? AND url=?",
                          (job_id, source_run, ticker, candidate.url))
    if existing:
        store.event(job_id, "discover", "Repeated source URL retained in the original response; existing observation reused.",
                    duplicate_discovery=existing[0]["id"], url=candidate.url)
        return existing[0]["id"]
    assessment = assess(company, candidate, body=candidate.metadata.get("description") or "")
    good, reason = assessment["accepted"], assessment["reason"]
    candidate.metadata["assessment"] = assessment
    candidate.metadata["policy_hash"] = RUNTIME_CODE_SHA256["core.py"]
    if not candidate.title.strip() or candidate.title == "Untitled feed item":
        good, reason = False, "Missing usable source headline."
    try:
        link = urlsplit(candidate.url)
        if link.scheme not in ("http", "https") or not link.hostname or link.username or link.password or link.port not in (None, 80, 443):
            good, reason = False, "Missing or unsupported public HTTP/HTTPS source link."
    except ValueError:
        good, reason = False, "Malformed source link."
    pub = candidate.published.get("utc")
    if pub and utc(pub) > seen:
        good, reason = False, "Publisher timestamp is in the future relative to discovery."
    if candidate.published.get("precision") == "date" and (candidate.published.get("raw") or "")[:10] > seen[:10]:
        good, reason = False, "Publisher date is in the future relative to discovery; precise time is unknown."
    key = store.execute("INSERT OR IGNORE INTO discoveries(job_id,source_run,ticker,url,candidate,first_seen,headline_ok,headline_reason,assessed_at) VALUES(?,?,?,?,?,?,?,?,?)",
                        (job_id, source_run, ticker, candidate.url, candidate.model_dump_json(), seen, int(good), reason, seen))
    # lastrowid is not reliable after INSERT OR IGNORE; query the unique identity.
    return store.rows("SELECT id FROM discoveries WHERE job_id=? AND source_run=? AND ticker=? AND url=?",
                      (job_id, source_run, ticker, candidate.url))[0]["id"]


def quarantine(store, ticker, raw_hash, reason):
    store.execute("INSERT OR IGNORE INTO content_quarantines(ticker,raw_hash,at,reason) VALUES(?,?,?,?)",
                  (ticker, raw_hash, now(), reason))


def ingest(store: Store, discovery_id: int, company: Company, fetched: Fetched, *, extracted_at: str | None = None,
           source_evidence: dict | None = None) -> str:
    assert_runtime_current()
    row = store.rows("SELECT * FROM discoveries WHERE id=?", (discovery_id,))[0]
    candidate = Candidate.model_validate_json(row["candidate"])
    try:
        parsed = extract(fetched.body, fetched.url, fetched.content_type)
    except ValueError as exc:
        if "PDF-only release" in str(exc):
            quarantine(store, company.ticker, fetched.artifact, str(exc))
        raise
    at = extracted_at or now()
    title = parsed["title"] or candidate.title
    assessment = assess(company, candidate, title, parsed["body"])
    accepted, reason = assessment["accepted"], assessment["reason"]
    pub = (parsed["published"] if parsed["published"].get("utc") else candidate.published
           if candidate.published.get("utc") else parsed["published"]
           if parsed["published"].get("raw") else candidate.published)
    instants = [x.get("utc") for x in [parsed["published"], candidate.published] if x.get("utc")]
    # Exact timestamps that disagree cannot be silently resolved in a strict export.
    conflict = parsed["date_conflict"] or len({utc(x) for x in instants}) > 1
    future = any(utc(x) > fetched.at for x in instants)
    future = future or any(x.get("precision") == "date" and (x.get("raw") or "")[:10] > fetched.at[:10]
                           for x in [parsed["published"], candidate.published])
    if conflict:
        reason += " Conflicting publication timestamps; strict export quarantines this version."
    if future:
        reason += " Publication timestamp is later than fetch; strict export quarantines this version."
    metadata = {"source": candidate.metadata, "publisher_updated": parsed["updated"],
                "date_assertions": [*parsed["date_assertions"], candidate.published], "date_conflict": conflict,
                "future_publication": future, "source_item_id": candidate.source_item_id,
                "discovery_id": discovery_id, "discovery_artifact": candidate.discovery_artifact,
                "company_config": company.model_dump(), "config_hash": digest({"company": company.model_dump(), "source": candidate.metadata["source_profile"]}) if candidate.metadata.get("source_profile") else digest(company.model_dump()),
                "evidence": "local_observation", "policy_version": "observed-v1",
                "source_url": fetched.url, "assessment": assessment,
                "publisher_group": candidate.metadata.get("publisher_group", candidate.publisher),
                "authors": candidate.metadata.get("authors", []),
                "license_url": candidate.metadata.get("license_url", ""),
                "rights_note": candidate.metadata.get("rights_note", ""),
                "source_evidence": source_evidence or {}, "pdf_pages": parsed.get("pdf_pages"),
                "layout_note": parsed.get("layout_note")}
    metadata["content_scope"] = "publisher_feed_body" if candidate.metadata.get("embedded_html") else "publisher_page"
    if fetched.body.startswith(b"%PDF"):
        metadata["content_scope"] = "publisher_pdf"
    key = store.save_version({"ticker": company.ticker, "source": candidate.source,
        "publisher": candidate.publisher, "url": candidate.url, "title": title,
        "body": parsed["body"], "paragraphs": parsed["paragraphs"],
        "raw_hash": fetched.artifact, "text_hash": digest(parsed["body"]),
        "extracted_at": at, "fetched_at": fetched.at, "first_seen": row["first_seen"],
        "published": pub, "provider_time": candidate.provider_time,
        "extractor": parsed["extractor"] + "/rules-" + RUNTIME_CODE_SHA256["extract.py"][:12]
                     + "/policy-" + RUNTIME_CODE_SHA256["core.py"][:12],
        "kind": candidate.kind, "accepted": int(accepted), "reason": reason, "metadata": metadata})
    store.execute("UPDATE discoveries SET state=?,reason=? WHERE id=?",
                  ("accepted" if accepted else "rejected", reason, discovery_id))
    store.event(row["job_id"], "extract", ("Accepted: " if accepted else "Rejected: ") + title,
                version=key, url=candidate.url)
    return key


def run_job(store: Store, job_id: str, *, fetcher_factory=Fetcher):
    assert_runtime_current()
    job = store.job(job_id)
    if job["status"] == "completed":
        return
    raw_request = json.loads(job["request"])
    request = Request.model_validate(raw_request).checked()
    companies = {t: Company.model_validate(v) for t, v in raw_request.get("companies", {}).items()}
    store.execute("UPDATE jobs SET status='running',started_at=COALESCE(started_at,?),heartbeat=?,owner_pid=?,error=NULL WHERE id=?",
                  (now(), now(), os.getpid(), job_id))
    fetcher = fetcher_factory(store, job_id)
    interrupted = False
    try:
        # Resume uses persisted pending discoveries; finished source discovery is not repeated.
        for ticker in request.tickers:
            company = companies.get(ticker) or store.settings.companies()[ticker]
            for source in request.sources:
                assert_runtime_current()
                store.checkpoint(job_id)
                previous = store.rows("SELECT * FROM source_runs WHERE job_id=? AND ticker=? AND source=? ORDER BY id DESC",
                                      (job_id, ticker, source))
                if previous and previous[0]["status"] in ("completed", "partial"):
                    continue
                source_run = store.execute("INSERT INTO source_runs(job_id,ticker,source,started_at,status,coverage) VALUES(?,?,?,?,?,?)",
                                           (job_id, ticker, source, now(), "running", "unknown"))
                store.event(job_id, "discover", f"Checking {source} for {ticker}.")
                try:
                    # Commit discoveries per source response, before the next request can fail.
                    candidates, coverage, artifacts = ADAPTERS[source](fetcher, company, request,
                        emit=lambda candidate: discover(store, job_id, source_run, ticker, company, candidate))
                    source_status = "partial" if "INCOMPLETE" in coverage or "Feed failures:" in coverage else "completed"
                    store.execute("UPDATE source_runs SET status=?,finished_at=?,discovered=?,coverage=?,artifact=? WHERE id=?",
                                  (source_status, now(), len(candidates), coverage, json.dumps(artifacts), source_run))
                except InterruptedError:
                    raise
                except Exception as exc:
                    store.execute("UPDATE source_runs SET status='failed',finished_at=?,detail=?,discovered=(SELECT COUNT(*) FROM discoveries WHERE source_run=source_runs.id),coverage='unknown: source failed; any returned discoveries retained' WHERE id=?",
                                  (now(), f"{type(exc).__name__}: {exc}", source_run))
                    store.event(job_id, "source", f"{source} / {ticker}: {exc}", "error")
        pending = store.rows("SELECT * FROM discoveries WHERE job_id=? AND state IN ('pending','fetching') ORDER BY id", (job_id,))
        # Round-robin source/ticker groups: a busy issuer feed must not consume all bodies.
        groups = defaultdict(deque)
        for row in sorted(pending, key=lambda r: (not r["headline_ok"], r["id"])):
            metadata = json.loads(row["candidate"]).get("metadata", {})
            groups[(row["ticker"], row["source_run"], metadata.get("profile_id"))].append(row)
        candidates = []
        while any(groups.values()):
            for group in groups.values():
                if group:
                    candidates.append(group.popleft())
        attempted = store.rows("SELECT COUNT(*) n FROM events WHERE job_id=? AND stage='fetch_attempt'", (job_id,))[0]["n"]
        legacy_attempts = store.rows("SELECT COUNT(*) n FROM events WHERE job_id=? AND stage='fetch' AND message LIKE 'Fetching %'", (job_id,))[0]["n"]
        attempted = max(attempted, legacy_attempts)
        fetched_urls = {r["url"] for r in store.rows("SELECT DISTINCT url FROM discoveries WHERE job_id=? AND state='accepted'", (job_id,))}
        for row in candidates:
            store.checkpoint(job_id)
            candidate = Candidate.model_validate_json(row["candidate"])
            company = companies.get(row["ticker"]) or store.settings.companies()[row["ticker"]]
            pub = candidate.published.get("utc")
            pub_date = publication_day(candidate.published)
            reason = None
            if pub_date and len(pub_date) == 10 and (pub_date < str(request.since) or pub_date > str(request.until)):
                reason = "Outside the requested publication-date interval."
            elif not request.collect_bodies or candidate.metadata.get("body_mode") == "metadata_only":
                state = "headline_only" if row["headline_ok"] else "rejected"
                reason = "Headline saved; full article was not requested." if row["headline_ok"] else row["headline_reason"]
                store.execute("UPDATE discoveries SET state=?,reason=? WHERE id=?", (state, reason, row["id"]))
                continue
            elif attempted >= request.max_articles:
                reason = "Per-job article fetch limit reached; discovered item retained."
            elif not candidate.url.startswith(("https://", "http://")):
                reason = "Missing or unsupported source URL."
            elif candidate.metadata.get("body_mode") == "feed_only" and not candidate.metadata.get("embedded_html"):
                reason = "Source profile uses publisher feed bodies only; no substantial body supplied."
            elif candidate.url in fetched_urls:
                reason = "Repeated URL in this run; existing article representation retained."
            if reason:
                store.execute("UPDATE discoveries SET state='rejected',reason=? WHERE id=?", (reason, row["id"]))
                continue
            attempted += 1
            store.event(job_id, "fetch_attempt", "Article-processing budget used.", discovery_id=row["id"])
            store.execute("UPDATE discoveries SET state='fetching' WHERE id=?", (row["id"],))
            try:
                source_evidence = {}
                store.event(job_id, "fetch", f"Fetching {candidate.title}", url=candidate.url)
                if candidate.metadata.get("embedded_html"):
                    # Publisher-supplied RSS content is a distinct, explicit representation.
                    # Preserve the full original feed via discovery_artifact as well as this entry.
                    body = candidate.metadata["embedded_html"].encode("utf-8")
                    artifact = store.archive(body)
                    fetched = Fetched(candidate.metadata["feed_url"], body, artifact,
                                      candidate.metadata["feed_fetched_at"], "text/html", 200)
                    store.event(job_id, "fetch", "Using body supplied by publisher RSS; original feed retained.", url=candidate.url)
                else:
                    fetched = fetcher.get(candidate.url)
                    pdf = linked_pdf(fetched.body, fetched.url) if not fetched.body.startswith(b"%PDF") else None
                    if pdf:
                        quarantine(store, company.ticker, fetched.artifact, "PDF-only release landing page is not an article body; collect the linked official PDF.")
                        source_evidence = {"landing_artifact": fetched.artifact, "landing_url": fetched.url,
                                           "landing_fetched_at": fetched.at, "linked_pdf_url": pdf}
                        store.event(job_id, "fetch", "Following the issuer's explicit PDF-only release link.", url=pdf)
                        fetched = fetcher.get(pdf, delay=2)
                        if not fetched.body.startswith(b"%PDF"):
                            raise ValueError("Official release link did not return PDF bytes; no full body exported.")
                ingest(store, row["id"], company, fetched, source_evidence=source_evidence)
                fetched_urls.add(candidate.url)
            except InterruptedError:
                raise
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                store.execute("UPDATE discoveries SET state='rejected',reason=? WHERE id=?", (message, row["id"]))
                store.event(job_id, "fetch", f"Could not extract {candidate.title}: {message}", "warning")
        states = store.rows("SELECT state,COUNT(*) n FROM discoveries WHERE job_id=? GROUP BY state", (job_id,))
        sources = store.rows("SELECT status FROM source_runs WHERE job_id=?", (job_id,))
        rejected = store.rows("SELECT reason FROM discoveries WHERE job_id=? AND state='rejected'", (job_id,))
        partial = any(s["status"] != "completed" for s in sources) or any(
            "fetch limit" in (x["reason"] or "") or (x["reason"] or "").startswith(("FetchError:", "ValueError:", "OSError:")) for x in rejected)
        status = "partial" if partial else "completed"
        summary = {x["state"]: x["n"] for x in states}
        summary["coverage"] = "Bounded source coverage; inspect Source health for unavailable periods."
        store.execute("UPDATE jobs SET status=?,finished_at=?,heartbeat=?,summary=? WHERE id=?",
                      (status, now(), now(), json.dumps(summary), job_id))
        store.event(job_id, "finish", f"Run {status}. All discovered items have a recorded disposition.", **summary)
    except InterruptedError as exc:
        interrupted = True
        store.execute("UPDATE source_runs SET status='interrupted',finished_at=? WHERE job_id=? AND status='running'", (now(), job_id))
        store.execute("UPDATE jobs SET status='cancelled',finished_at=?,error=? WHERE id=?", (now(), str(exc), job_id))
        store.event(job_id, "cancel", str(exc), "warning")
    except BaseException as exc:
        store.execute("UPDATE jobs SET status='failed',finished_at=?,error=? WHERE id=?", (now(), str(exc), job_id))
        store.event(job_id, "failure", f"{type(exc).__name__}: {exc}", "error")
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        fetcher.close()
    if request.mode == "watch" and not interrupted and not store.job(job_id)["cancel"]:
        due = (datetime.now(UTC) + timedelta(minutes=request.interval_minutes)).isoformat(timespec="microseconds")
        store.execute("UPDATE jobs SET next_run=? WHERE id=?", (due, job_id))


def lease(store: Store) -> str | None:
    token = uuid.uuid4().hex
    with store.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        row = db.execute("SELECT * FROM worker_lease WHERE name='collector'").fetchone()
        if row and psutil.pid_exists(row["pid"]):
            return None
        db.execute("INSERT OR REPLACE INTO worker_lease VALUES('collector',?,?,?)", (os.getpid(), token, now()))
    # A dead worker left in-progress jobs and source records; reset only these.
    store.execute("UPDATE jobs SET status='queued',owner_pid=NULL WHERE status='running'")
    store.execute("UPDATE source_runs SET status='interrupted',finished_at=? WHERE status='running'", (now(),))
    return token


def work(store: Store, *, once: bool = False):
    (store.settings.root / "stop-worker").unlink(missing_ok=True)
    token = lease(store)
    if not token:
        return
    try:
        while True:
            if (store.settings.root / "stop-worker").exists():
                break
            queued = store.rows("SELECT id FROM jobs WHERE status='queued' AND cancel=0 ORDER BY created_at LIMIT 1")
            if queued:
                run_job(store, queued[0]["id"])
            # Durable watch schedule, explicitly restarted after a sleeping laptop.
            due = store.rows("SELECT * FROM jobs WHERE next_run IS NOT NULL AND next_run<=? AND cancel=0", (now(),))
            for parent in due:
                spec = json.loads(parent["request"])
                spec["until"] = str(datetime.now(UTC).date())
                # A rolling seven-day reconciliation includes delayed provider items.
                spec["since"] = str(datetime.now(UTC).date() - timedelta(days=7))
                new_job = store.create_job(spec)
                store.execute("UPDATE jobs SET next_run=NULL WHERE id=?", (parent["id"],))
                store.event(new_job, "watch", "Scheduled catch-up: first-seen times are the actual receipt times.", parent=parent["id"])
            store.execute("UPDATE worker_lease SET heartbeat=? WHERE token=?", (now(), token))
            if once and not queued:
                break
            if not queued:
                time.sleep(2)
    finally:
        store.execute("DELETE FROM worker_lease WHERE token=?", (token,))


def start_worker(settings: Settings, *, once: bool = False) -> int:
    (settings.root / "stop-worker").unlink(missing_ok=True)
    args = [sys.executable, "-m", "newscollector.worker", "--home", str(settings.root)]
    if once:
        args.append("--once")
    log_path = settings.root / "worker.log"
    kwargs = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {"start_new_session": True}
    with log_path.open("ab") as log:
        process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=log, stderr=log, **kwargs)
    return process.pid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    work(Store(Settings(args.home)), once=args.once)

"""Synthetic, redistributable workflow example. Never presented as market evidence."""
import json

from .core import Candidate, Company, Request, Settings, timestamp, utc
from .fetch import Fetched
from .store import Store
from .worker import discover, enqueue, ingest


def load_demo(store: Store) -> str:
    company = Company(ticker="DEMO", name="Example Robotics", aliases=["Example Robotics"], feeds=[])
    store.settings.save_company(company)
    previous = store.rows("SELECT id FROM jobs WHERE json_extract(request,'$.demo_loaded')=1")
    if previous:
        return previous[0]["id"]
    request = Request(tickers=["DEMO"], since="2026-10-20", until="2026-10-24")
    job = enqueue(store, request)
    source_run = store.execute("INSERT INTO source_runs(job_id,ticker,source,started_at,finished_at,status,coverage) VALUES(?,?,?,?,?,?,?)",
                              (job, "DEMO", "demo", utc("2026-10-23T13:00:00Z"), utc("2026-10-23T13:10:00Z"), "completed", "Synthetic offline fixtures only; no actual company or market news."))
    examples = [
        ("Example Robotics opens a research facility", "Example Robotics opened a new research facility on Friday. The facility will support engineering and testing of industrial robots. Management described the staffing plan and stated that the project would be completed in phases. The announcement did not disclose any stock price movement.", "facility"),
        ("Example Robotics announces quarterly results", "Example Robotics reported quarterly revenue of 120 million dollars. Operating expenses increased as the company expanded its engineering team. Management said customer orders remained uncertain and offered no guarantee about the following quarter. This synthetic report is only an extraction demonstration.", "results"),
        ("Example Robotics announces quarterly results", "Example Robotics reported quarterly revenue of 120 million dollars. Operating expenses increased as the company expanded its engineering team. Management said customer orders remained uncertain and offered no guarantee about the following quarter. This synthetic report is only an extraction demonstration.", "syndicated-results"),
        ("Another Company launches a bicycle", "Another Company announced a bicycle for city commuters. The company described the wheel design, delivery schedule and distribution partners. This fictional item is unrelated to the configured company, demonstrating a recorded relevance rejection rather than silently discarding the article.", "irrelevant"),
    ]
    for i, (title, body, slug) in enumerate(examples):
        page = f'<html><head><meta property="article:published_time" content="2026-10-23T12:00:00Z"></head><body><article><h1>{title}</h1><p>{body}</p></article></body></html>'.encode()
        artifact = store.archive(page)
        c = Candidate(url=f"https://example.org/demo/{slug}", title=title, publisher="Synthetic Example Newsroom",
                      source="demo", published=timestamp("2026-10-23T12:00:00Z"), discovery_artifact=artifact,
                      metadata={"synthetic": True, "sentiment": "not an input"})
        discovery_id = discover(store, job, source_run, "DEMO", company, c, utc(f"2026-10-23T13:0{i}:00Z"))
        ingest(store, discovery_id, company, Fetched(c.url, page, artifact, utc(f"2026-10-23T13:0{i}:20Z"), "text/html", 200),
               extracted_at=utc(f"2026-10-23T13:0{i}:30Z"))
    failed = Candidate(url="https://example.org/demo/blocked", title="Example Robotics release unavailable",
                       publisher="Synthetic Example Newsroom", source="demo",
                       published=timestamp("2026-10-23T12:30:00Z"), metadata={"synthetic": True})
    did = discover(store, job, source_run, "DEMO", company, failed, utc("2026-10-23T13:05:00Z"))
    store.execute("UPDATE discoveries SET state='rejected',reason='Synthetic HTTP 403 example: access unavailable; no bypass attempted.' WHERE id=?", (did,))
    store.execute("UPDATE jobs SET status='completed',finished_at=?,summary=? WHERE id=?",
                  (utc("2026-10-23T13:10:00Z"), json.dumps({"accepted": 3, "rejected": 2, "synthetic": True}), job))
    store.event(job, "demo", "Synthetic demo loaded. These records are not real financial news.")
    # Stable marker separate from random run IDs for repeated dashboard/demo commands.
    store.execute("UPDATE jobs SET request=json_set(request,'$.demo_loaded',1) WHERE id=?", (job,))
    return job

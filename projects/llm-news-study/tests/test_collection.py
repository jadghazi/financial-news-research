import json
from pathlib import Path

import httpx
import pytest

from newscollector.core import Candidate, Request, digest, relevance, timestamp
from newscollector.fetch import FetchError, Fetcher, public_url
from newscollector.sources import sec_candidates
from newscollector.worker import enqueue, lease, run_job


def fixture_fetcher(handler, waits=None):
    def factory(store, job):
        return Fetcher(store, job, client=httpx.Client(transport=httpx.MockTransport(handler)),
                       validate=lambda _: None, sleep=(waits.append if waits is not None else lambda _: None))
    return factory


def test_end_to_end_feed_failure_and_limit_accounting(store, job, company):
    entries = "".join(f'<item><title>Test Robotics announcement {i}</title><link>https://news.example.com/article/{i}</link><pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate></item>' for i in range(5))
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path == "/feed":
            return httpx.Response(200, text=f"<rss><channel><title>Fixture newsroom</title>{entries}</channel></rss>", headers={"content-type": "application/rss+xml"})
        if request.url.path.endswith("/1"):
            return httpx.Response(403, text="Access denied")
        return httpx.Response(200, text="<html><article><h1>Test Robotics announcement</h1><p>Test Robotics announced an expansion of its engineering facility. The company said the project will take several quarters and depends on market conditions. This is a synthetic fixture used to verify article extraction and failure reporting.</p></article></html>", headers={"content-type": "text/html"})
    run_job(store, job, fetcher_factory=fixture_fetcher(handler))
    rows = store.rows("SELECT state,reason FROM discoveries WHERE job_id=?", (job,))
    assert len(rows) == 5
    assert all(x["state"] in ("accepted", "rejected") for x in rows)
    assert sum(x["state"] == "accepted" for x in rows) == 2
    assert any("HTTP 403" in (x["reason"] or "") for x in rows)
    assert sum("fetch limit" in (x["reason"] or "") for x in rows) == 2
    assert store.job(job)["status"] == "partial"
    assert store.rows("SELECT COUNT(*) n FROM versions")[0]["n"] == 2


def test_retry_after_and_robots_denial(store, job):
    waits = []
    attempts = []
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
        attempts.append(str(request.url))
        if len(attempts) == 1:
            return httpx.Response(429, text="wait", headers={"retry-after": "4"})
        return httpx.Response(200, text="saved article")
    f = fixture_fetcher(handler, waits)(store, job)
    assert f.get("https://news.example.com/allowed").body == b"saved article"
    assert any(w >= 4 for w in waits)
    with pytest.raises(FetchError, match="disallows"):
        f.get("https://news.example.com/blocked")
    assert len(attempts) == 2


def test_external_url_validation():
    for url in ("file:///etc/passwd", "http://127.0.0.1/", "https://user:pass@example.com/", "http://localhost:1234/"):
        with pytest.raises(FetchError):
            public_url(url)


def test_all_source_failures_are_visible(store, job):
    run_job(store, job, fetcher_factory=fixture_fetcher(lambda r: httpx.Response(403, text="no")))
    assert store.job(job)["status"] == "partial"
    assert store.rows("SELECT status FROM source_runs")[0]["status"] == "failed"
    assert store.rows("SELECT level FROM events WHERE level='error'")


def test_cancel_resume_and_lease(store, job):
    store.execute("UPDATE jobs SET cancel=1 WHERE id=?", (job,))
    run_job(store, job, fetcher_factory=fixture_fetcher(lambda r: httpx.Response(200, text="")))
    assert store.job(job)["status"] == "cancelled"
    token = lease(store)
    assert token
    assert lease(store) is None
    store.execute("DELETE FROM worker_lease")


def test_archive_checksum_and_backup(store, tmp_path):
    key = store.archive(b"original bytes")
    assert store.archive(b"original bytes") == key
    destination = tmp_path / "backup"
    store.backup(destination)
    from newscollector.core import Settings
    from newscollector.store import Store
    restored = Store(Settings(destination))
    assert restored.read_artifact(key) == b"original bytes"
    artifact = store.artifacts / key[:2] / (key + ".gz")
    import gzip
    artifact.write_bytes(gzip.compress(b"modified bytes"))
    with pytest.raises(ValueError, match="checksum"):
        store.read_artifact(key)


def test_plain_ticker_not_relevance(company):
    assert not relevance(company, "TST appears in an unrelated acronym")[0]
    assert relevance(company, "NYSE: TST launches a product")[0]


def test_sec_missing_contact_is_actionable(store, job, company):
    f = fixture_fetcher(lambda r: httpx.Response(200, text=""))(store, job)
    store.settings.contact = ""
    with pytest.raises(ValueError, match="contact"):
        sec_candidates(f, company, Request(tickers=["TST"], sources=["sec"], since="2026-10-01", until="2026-10-31"))


def test_discoveries_survive_a_later_metadata_request_failure(store, job, company):
    store.settings.contact = "test@example.org"
    spec = json.loads(store.job(job)["request"])
    spec["sources"] = ["sec"]
    spec["companies"]["TST"]["cik"] = "123"
    store.execute("UPDATE jobs SET request=? WHERE id=?", (json.dumps(spec), job))
    recent = {"filings": {"recent": {"form": ["8-K"], "accessionNumber": ["0000000123-26-000001"],
        "primaryDocument": ["report.htm"], "filingDate": ["2026-10-22"]},
        "files": [{"name": "old.json", "filingFrom": "2026-10-01", "filingTo": "2026-10-23"}]}}
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path.endswith("old.json"):
            return httpx.Response(403)
        if request.url.path.endswith(".json"):
            return httpx.Response(200, json=recent)
        return httpx.Response(200, text="<html><article><h1>Test Robotics filing</h1><p>Test Robotics reported its financial results and described the operating environment. The company explained that future conditions remain uncertain and that the figures should be read with the full filing disclosures.</p></article></html>", headers={"content-type": "text/html"})
    run_job(store, job, fetcher_factory=fixture_fetcher(handler))
    assert store.rows("SELECT state FROM discoveries WHERE job_id=?", (job,)) == [{"state": "accepted"}]
    assert store.rows("SELECT status,discovered FROM source_runs")[0] == {"status": "failed", "discovered": 1}
    assert store.job(job)["status"] == "partial"


def test_cancel_after_discovery_resumes_without_losing_first_seen(store, job):
    requested = []
    def handler(request):
        requested.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path == "/feed":
            store.execute("UPDATE jobs SET cancel=1 WHERE id=?", (job,))
            return httpx.Response(200, text='<rss><channel><title>News</title><item><title>Test Robotics report</title><link>https://news.example.com/report</link><pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate></item></channel></rss>')
        return httpx.Response(200, text="<html><article><h1>Test Robotics report</h1><p>Test Robotics provided an update about its factory and the expected delivery schedule. This fixture checks that a cancelled job can resume its pending article without losing the original discovery time.</p></article></html>", headers={"content-type": "text/html"})
    factory = fixture_fetcher(handler)
    run_job(store, job, fetcher_factory=factory)
    assert store.job(job)["status"] == "cancelled"
    observation = store.rows("SELECT first_seen,state FROM discoveries")[0]
    assert observation["state"] == "pending"
    store.execute("UPDATE jobs SET status='queued',cancel=0 WHERE id=?", (job,))
    run_job(store, job, fetcher_factory=factory)
    assert requested.count("/feed") == 1
    assert store.rows("SELECT state FROM discoveries")[0]["state"] == "accepted"
    assert store.rows("SELECT first_seen FROM versions")[0]["first_seen"] == observation["first_seen"]

import json

import httpx
from typer.testing import CliRunner

from newscollector.cli import app
from newscollector.core import Candidate, Company, Request, Settings, timestamp, utc
from newscollector.fetch import Fetched
from newscollector.query import get_news, serialize
from newscollector.sources import feed_candidates, sec_candidates
from newscollector.worker import discover, enqueue, ingest, run_job
from test_collection import fixture_fetcher


def test_cli_demo_export_verify_and_csv(tmp_path):
    runner = CliRunner()
    home = tmp_path / "offline"
    assert runner.invoke(app, ["demo", "--home", str(home)]).exit_code == 0
    for variant in ("headline", "lead", "full"):
        output = tmp_path / (variant + ".jsonl")
        result = runner.invoke(app, ["export", "--home", str(home), "--ticker", "DEMO", "--variant", variant, "--output", str(output)])
        assert result.exit_code == 0, result.output
        rows = [json.loads(x) for x in output.read_text(encoding="utf-8").splitlines()]
        assert rows and all(x["ticker"] == "DEMO" for x in rows)
        assert output.with_suffix(".jsonl.manifest.json").exists()
    assert runner.invoke(app, ["verify", "--home", str(home)]).exit_code == 0


def test_publisher_embedded_feed_body_needs_no_page_request(store, job, company):
    requested = []
    content = "<p>Test Robotics announced a new facility. The company explained how production and staffing would change, and described uncertainty about future demand.</p>" * 8
    def handler(request):
        requested.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        if request.url.path == "/feed":
            return httpx.Response(200, text=f'<rss xmlns:content="http://purl.org/rss/1.0/modules/content/"><channel><title>News</title><item><title>Test Robotics opens factory</title><link>https://news.example.com/blocked</link><pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate><content:encoded><![CDATA[{content}]]></content:encoded></item></channel></rss>')
        return httpx.Response(403)
    run_job(store, job, fetcher_factory=fixture_fetcher(handler))
    assert "/blocked" not in requested
    rows = store.rows("SELECT * FROM versions")
    assert len(rows) == 1 and rows[0]["accepted"]
    assert json.loads(rows[0]["metadata"])["content_scope"] == "publisher_feed_body"
    assert json.loads(rows[0]["metadata"])["discovery_artifact"]


def test_general_feed_does_not_automatically_accept_all_items(store, job, company):
    assert company.feed_scope == "general"
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, text='<rss><channel><title>Broad news</title><item><title>Unrelated company</title><link>https://news.example.com/other</link></item></channel></rss>')
    candidates, _, _ = feed_candidates(fixture_fetcher(handler)(store, job), company,
        Request(tickers=["TST"], since="2026-10-01", until="2026-10-31"))
    assert not candidates[0].company_scoped


def test_sec_follows_historical_file_and_keeps_acceptance_separate(store, job, company):
    store.settings.contact = "test@example.org"
    company.cik = "123"
    recent = {"filings": {"recent": {"form": []}, "files": [{"name": "CIK0000000123-submissions-001.json", "filingFrom": "2026-10-01", "filingTo": "2026-10-31"}]}}
    historical = {"form": ["8-K"], "accessionNumber": ["0000000123-26-000001"],
                  "primaryDocument": ["report.htm"], "filingDate": ["2026-10-22"],
                  "acceptanceDateTime": ["2026-10-22T20:10:00Z"], "items": ["2.02,9.01"]}
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        return httpx.Response(200, json=historical if "submissions-001" in request.url.path else recent)
    candidates, note, artifacts = sec_candidates(fixture_fetcher(handler)(store, job), company,
        Request(tickers=["TST"], sources=["sec"], since="2026-10-01", until="2026-10-31"))
    assert len(artifacts) == 2
    assert len(candidates) == 1
    assert candidates[0].published["utc"] is None
    assert candidates[0].provider_time["semantic"] == "filing_acceptance"
    assert candidates[0].metadata["items"] == "2.02,9.01"


def test_date_only_z_remains_date():
    assert timestamp("2026-10-23Z")["precision"] == "date"
    assert timestamp("2026-10-23Z")["utc"] is None


def test_parquet_roundtrip():
    import pyarrow.parquet as pq
    import io
    payload = serialize([{"ticker": "TEST", "text": "company news"}], "parquet")
    assert pq.read_table(io.BytesIO(payload)).to_pylist() == [{"ticker": "TEST", "text": "company news"}]


def test_known_publisher_controls_not_article_content():
    from newscollector.extract import extract
    body = "The company announced revenue of 120 million dollars and said it did not expect demand to recover immediately. It described its production plans and uncertainty about delivery schedules."
    page = f'<html><article><h1>Company results</h1><p>{body}</p><p>Share article</p><p>Media</p><p>- Text of this article</p><p>- Media in this article</p></article></html>'.encode()
    result = extract(page, "https://www.apple.com/newsroom/2026/10/example/")
    assert "120 million" in result["body"]
    assert "did not expect" in result["body"]
    assert "Share article\nMedia" not in result["body"]

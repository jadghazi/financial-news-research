import json

import httpx
import pytest
from typer.testing import CliRunner

from newscollector.cli import app
from newscollector.core import Candidate, Company, Request, SourceProfile, assess, now, timestamp, utc
from newscollector.query import get_news
from newscollector.quality import audit_report, corpus_report, create_audit, near_pairs, save_audit_review
from newscollector.sources import _sec_items, issuer_candidates
from newscollector.worker import discover, enqueue, run_job
from test_collection import fixture_fetcher
from test_temporal import article


def test_baba_listing_and_foreign_issuer_forms(store):
    baba = store.settings.companies()["BABA"]
    assert baba.calendar == "XNYS" and baba.exchange == "NYSE"
    assert "ADS" in baba.security_type
    assert all(symbol != "BABA" for symbol in baba.related_symbols)
    data = {"form": ["6-K", "20-F", "8-K"], "accessionNumber": ["a", "b", "c"],
            "primaryDocument": ["a.htm", "b.htm", "c.htm"], "filingDate": ["2025-01-01"] * 3}
    items = _sec_items([(data, "fixture", "2026-10-24T12:00:00Z")], baba, baba.cik)
    assert len(items) == 2
    assert "6-K" in items[0].title and "20-F" in items[1].title
    assert all(item.published["utc"] is None for item in items)


def test_issuer_index_preserves_dates_and_deduplicates_links(store, job):
    baba = store.settings.companies()["BABA"]
    node = '<a href="/en-US/document-123"><div class="component-title">Alibaba issues results</div><div class="date-date">October 2, 2025</div><div>Mobile repeated title</div></a>'
    def handler(request):
        return httpx.Response(200, text="User-agent: *\nAllow: /\n" if request.url.path == "/robots.txt" else "<html>" + node * 2 + "</html>")
    f = fixture_fetcher(handler)(store, job)
    emitted = []
    items, note, raw = issuer_candidates(f, baba, Request(tickers=["BABA"], sources=["issuer"], since="2025-01-01", until="2025-12-31"), emit=emitted.append)
    assert len(items) == len(emitted) == 1 and len(raw) == 1
    assert items[0].title == "Alibaba issues results"
    assert items[0].published["precision"] == "date"
    assert items[0].published["utc"] is None
    assert items[0].published["source_raw"] == "October 2, 2025"
    assert "INCOMPLETE" in note


def test_relevance_context_and_pure_price_recaps():
    baba = Company(ticker="BABA", name="Alibaba Group", aliases=["Alibaba"], sector_terms=["China e-commerce"])
    sector = Candidate(url="https://example.org/one", title="China e-commerce rules revised", publisher="Fixture", source="publishers", metadata={"scope": "sector", "role": "independent_reporting"})
    assert assess(baba, sector)["scope"] == "sector"
    unrelated = sector.model_copy(update={"title": "A new school opens"})
    assert not assess(baba, unrelated)["accepted"]
    macro = unrelated.model_copy(update={"metadata": {"scope": "market", "role": "primary_macro"}})
    assert assess(baba, macro)["scope"] == "market"
    recap = sector.model_copy(update={"title": "Alibaba shares rise 5% today"})
    assert "price_recap_candidate" in assess(baba, recap)["flags"]
    news = recap.model_copy(update={"title": "Alibaba shares rise after new earnings guidance"})
    assert "price_recap_candidate" not in assess(baba, news)["flags"]
    assert not assess(baba, unrelated, body=("Unrelated material. " * 200) + " Alibaba") ["accepted"]


def test_context_and_price_recap_exports_are_explicit(store, job, company):
    source_run = store.execute("INSERT INTO source_runs(job_id,ticker,source,started_at,status,coverage) VALUES(?,?,?,?,?,?)", (job, "TST", "context", now(), "completed", "fixture"))
    for title, scope, suffix in [("Central bank issues policy statement", "market", "macro"), ("Test Robotics stock falls 3%", "company", "recap")]:
        c = Candidate(url="https://news.example.com/" + suffix, title=title, publisher="Fixture", source="context",
                      metadata={"scope": scope, "role": "primary_macro" if scope == "market" else "independent_reporting"})
        discover(store, job, source_run, "TST", company, c, "2026-10-23T13:00:00Z")
    assert not get_news(store, "TST", "2026-10-23T14:00:00Z")["records"]
    context = get_news(store, "TST", "2026-10-23T14:00:00Z", scopes=["market"])["records"]
    assert len(context) == 1 and context[0]["relevance_scope"] == "market"
    recap = get_news(store, "TST", "2026-10-23T14:00:00Z", include_price_recaps=True)["records"]
    assert len(recap) == 1 and "price_recap_candidate" in recap[0]["quality_flags"]


def test_exploratory_history_excludes_future_or_ambiguous_publication(store, job, company):
    article(store, job, company, published="2025-03-01T12:00:00Z", slug="before")
    article(store, job, company, published="2025-03-03T12:00:00Z", slug="after")
    rows = get_news(store, "TST", "2025-03-02T12:00:00Z", policy="historical")["records"]
    assert len(rows) == 1 and rows[0]["url"].endswith("before")
    c = Candidate(url="https://news.example.com/dateonly", title="Test Robotics update", source="fixture", publisher="Fixture", published=timestamp("2025-03-02"))
    discover(store, job, 99, "TST", company, c, "2026-10-23T14:00:00Z")
    rows = get_news(store, "TST", "2025-03-02T23:00:00Z", policy="historical")["records"]
    assert all(not r["url"].endswith("dateonly") for r in rows)


def test_profiles_are_frozen_and_feed_only_never_fetches_page(store, company):
    profile = SourceProfile(name="Independent fixture", url="https://news.example.com/independent", scope="company",
        role="independent_reporting", body_mode="feed_only", publisher_group="Fixture group",
        documentation="https://example.org/terms", rights_note="Synthetic fixture", license_url="https://example.org/license")
    store.settings.save_source("fixture_news", profile)
    company.source_ids = ["fixture_news"]
    store.settings.save_company(company)
    job = enqueue(store, Request(tickers=["TST"], sources=["publishers"], since="2026-10-01", until="2026-10-31"))
    store.settings.save_source("fixture_news", profile.model_copy(update={"url": "https://changed.example.org/feed"}))
    paths = []
    def handler(request):
        paths.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /\n")
        assert request.url.path == "/independent"
        content = "<p>Test Robotics reported plans to build a new factory. The company said the expansion is subject to demand and financing conditions.</p>" * 8
        return httpx.Response(200, text=f'<rss xmlns:content="http://purl.org/rss/1.0/modules/content/" xmlns:dc="http://purl.org/dc/elements/1.1/"><channel><title>Independent fixture</title><item><title>Test Robotics factory plans</title><link>https://news.example.com/never-fetch</link><pubDate>Fri, 23 Oct 2026 12:00:00 GMT</pubDate><dc:creator>Fixture Author</dc:creator><content:encoded><![CDATA[{content}]]></content:encoded></item></channel></rss>')
    run_job(store, job, fetcher_factory=fixture_fetcher(handler))
    assert "/never-fetch" not in paths
    result = get_news(store, "TST", now(), variant="full")["records"]
    assert len(result) == 1
    assert result[0]["source_role"] == "independent_reporting"
    assert result[0]["license_url"] == "https://example.org/license" and result[0]["authors"] == ["Fixture Author"]


def test_near_duplicate_screen_protects_negation_numbers_and_dates():
    titles = ["Alibaba Group announces a new cloud services agreement today",
              "Alibaba Group announces new cloud services agreement today",
              "Alibaba Group does not announce new cloud services agreement today",
              "Alibaba Group announces new cloud services agreement for 2025 today"]
    records = [{"record_id": str(i), "headline": t, "published_at": "2025-10-02T12:00:00Z"} for i, t in enumerate(titles)]
    report = near_pairs(records)
    assert len(report["pairs"]) == 1
    assert report["pairs"][0]["left"] == "0" and report["pairs"][0]["right"] == "1"
    assert report["pairs"][0]["headline_similarity"] >= .86
    assert near_pairs(records, limit=2)["truncated_screen"]


def test_frozen_sampling_reproducible_and_uncertainty_not_hidden(store, job, company):
    for i in range(8):
        c = Candidate(url=f"https://news.example.com/{i}", title="Test Robotics update" if i < 5 else "Other company update", publisher="Fixture", source="fixture")
        discover(store, job, i, "TST", company, c, "2026-10-23T13:00:00Z")
    a = create_audit(store, "TST", seed=42, per_group=2, as_of="2026-10-23T14:00:00Z")
    b = create_audit(store, "TST", seed=42, per_group=2, as_of="2026-10-23T14:00:00Z")
    assert [r["record_id"] for r in a["records"]] == [r["record_id"] for r in b["records"]]
    assert a["population"]["accepted"] == {"population": 5, "sample": 2, "inclusion_probability": .4}
    assert a["population"]["rejected"]["population"] == 3
    one = next(r for r in a["records"] if r["stratum"] == "accepted")
    before = get_news(store, "TST", "2026-10-23T14:00:00Z")
    save_audit_review(store, a["audit_id"], one["record_id"], "Fixture reviewer", "Relevant", "Cannot assess", "Uncertain")
    result = audit_report(store, a["audit_id"], "Fixture reviewer")
    assert result["strata"]["accepted"]["fraction_relevant"] is None
    two = next(r for r in a["records"] if r["stratum"] == "accepted" and r != one)
    save_audit_review(store, a["audit_id"], two["record_id"], "Fixture reviewer", "Uncertain", "Cannot assess", "Uncertain")
    assert audit_report(store, a["audit_id"], "Fixture reviewer")["strata"]["accepted"]["wilson_95_interval"] is None
    save_audit_review(store, a["audit_id"], two["record_id"], "Fixture reviewer", "Relevant", "Cannot assess", "Uncertain")
    result = audit_report(store, a["audit_id"], "Fixture reviewer")
    assert result["strata"]["accepted"]["fraction_relevant"] == 1
    assert get_news(store, "TST", "2026-10-23T14:00:00Z") == before
    with pytest.raises(ValueError, match="frozen"):
        save_audit_review(store, a["audit_id"], "not-in-sample", "Fixture", "Relevant", "Faithful", "Supported")


def test_quality_empty_corpus_is_not_reported_as_adequate(store):
    report = corpus_report(store, "BABA")
    assert report["unique_headline_urls"] == 0 and report["publisher_hhi"] is None
    assert any("No eligible" in warning for warning in report["warnings"])
    assert report["limits"]


def test_documented_backfill_options(monkeypatch):
    captured = []
    monkeypatch.setattr("newscollector.cli._collect", lambda *args, **kwargs: captured.append(args))
    result = CliRunner().invoke(app, ["backfill", "--tickers", "BABA", "--source", "issuer,publishers", "--since", "2025-01-01", "--until", "2025-12-31"])
    assert result.exit_code == 0, result.output
    assert captured[0][:4] == ("BABA", "issuer,publishers", "2025-01-01", "2025-12-31")


def test_changed_runtime_cannot_claim_current_code_in_exports(store, job, company, monkeypatch):
    import newscollector
    article(store, job, company)
    monkeypatch.setitem(newscollector.RUNTIME_CODE_SHA256, "query.py", "different-loaded-source")
    with pytest.raises(ValueError, match="Restart"):
        get_news(store, "TST", now())
    from newscollector.query import manifest
    with pytest.raises(ValueError, match="Restart"):
        manifest({"records": []}, b"", "jsonl")

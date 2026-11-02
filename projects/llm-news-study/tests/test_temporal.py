import json

import pytest

from newscollector.core import Candidate, session_for, timestamp, utc
from newscollector.fetch import Fetched
from newscollector.query import export_news, get_news, serialize
from newscollector.worker import discover, ingest


def article(store, job, company, *, title="Test Robotics expands factory", body=None, seen="2026-10-23T13:00:00Z",
            fetched="2026-10-23T13:01:00Z", extracted="2026-10-23T13:02:00Z", published="2026-10-23T12:00:00Z", slug="one"):
    text = body or "Test Robotics expanded its factory to support future production. The company discussed staffing and capital expenditure, while warning that customer demand and delivery schedules remained uncertain. No stock outcomes are used in this synthetic example."
    raw = f'<html><head><meta property="article:published_time" content="{published}"></head><body><article><h1>{title}</h1><p>{text}</p></article></body></html>'.encode()
    artifact = store.archive(raw)
    sid = store.execute("INSERT INTO source_runs(job_id,ticker,source,started_at,status,coverage) VALUES(?,?,?,?,?,?)", (job, company.ticker, "fixture", utc(seen), "completed", "fixture"))
    c = Candidate(url="https://news.example.com/"+slug, title=title, publisher="Fixture News", source="fixture",
                  published=timestamp(published), discovery_artifact=artifact, metadata={"sentiment_reasoning": "NEVER EXPORT"})
    did = discover(store, job, sid, company.ticker, company, c, utc(seen))
    vid = ingest(store, did, company, Fetched(c.url, raw, artifact, utc(fetched), "text/html", 200), extracted_at=utc(extracted))
    return did, vid


def test_headline_can_precede_body(store, job, company):
    article(store, job, company)
    cutoff = "2026-10-23T13:00:30Z"
    assert len(get_news(store, "TST", cutoff, variant="headline")["records"]) == 1
    assert get_news(store, "TST", cutoff, variant="full")["records"] == []
    assert len(get_news(store, "TST", "2026-10-23T13:03:00Z", variant="full")["records"]) == 1


def test_matched_text_variants_share_ids_and_wait_for_body(store, job, company):
    article(store, job, company)
    assert get_news(store, "TST", "2026-10-23T13:00:30Z", variant="headline", cohort="body_available")["records"] == []
    variants = [get_news(store, "TST", "2026-10-23T13:03:00Z", variant=v, cohort="body_available")["records"] for v in ("headline", "lead", "full")]
    assert all(len(records) == 1 for records in variants)
    assert len({records[0]["record_id"] for records in variants}) == 1
    assert variants[0][0]["text"] == variants[0][0]["headline"]
    assert all(records[0]["cohort"] == "body_available" for records in variants)
    assert all("_cohort_text" not in records[0] for records in variants)
    old = store.rows("SELECT * FROM versions")[0]
    old.pop("id")
    old.pop("document_id")
    for key in ("paragraphs", "published", "provider_time", "metadata"):
        old[key] = json.loads(old[key])
    old["extracted_at"] = utc("2026-10-23T13:03:00Z")
    old["extractor"] += "/revalidated"
    newest = store.save_version(old)
    for variant in ("headline", "lead", "full"):
        records = get_news(store, "TST", "2026-10-23T13:04:00Z", variant=variant, cohort="body_available")["records"]
        assert records[0]["record_id"] == newest


def test_future_revision_does_not_replace_past(store, job, company):
    _, old = article(store, job, company)
    before = get_news(store, "TST", "2026-10-23T14:00:00Z", variant="full")["records"]
    article(store, job, company, body="Test Robotics revised the original release. The updated publication includes new information that was unavailable at the earlier cutoff and must never replace previously observed text. This fixture is intentionally substantially different from its original.",
            seen="2026-10-23T15:00:00Z", fetched="2026-10-23T15:01:00Z", extracted="2026-10-23T15:02:00Z")
    after = get_news(store, "TST", "2026-10-23T14:00:00Z", variant="full")["records"]
    assert before == after
    assert after[0]["record_id"] == old
    assert get_news(store, "TST", "2026-10-23T16:00:00Z", variant="full")["records"][0]["record_id"] != old


def test_old_date_downloaded_late_not_historical_proof(store, job, company):
    article(store, job, company, published="2025-01-01T12:00:00Z")
    assert not get_news(store, "TST", "2025-01-02T12:00:00Z", variant="full")["records"]
    assert len(get_news(store, "TST", "2025-01-02T12:00:00Z", variant="full", policy="historical")["records"]) == 1
    with pytest.raises(ValueError, match="not verified"):
        get_news(store, "TST", "2025-01-02T12:00:00Z", policy="archived_replay")


def test_future_publication_quarantined(store, job, company):
    article(store, job, company, published="2026-10-24T12:00:00Z")
    assert not get_news(store, "TST", "2026-10-23T16:00:00Z", variant="full")["records"]
    assert not get_news(store, "TST", "2026-10-23T16:00:00Z", variant="headline")["records"]


def test_duplicate_only_considers_eligible_copies(store, job, company):
    article(store, job, company)
    before = get_news(store, "TST", "2026-10-23T14:00:00Z", variant="full")["records"]
    article(store, job, company, slug="copy", seen="2026-10-23T15:00:00Z", fetched="2026-10-23T15:01:00Z", extracted="2026-10-23T15:02:00Z", published="2026-10-22T12:00:00Z")
    assert get_news(store, "TST", "2026-10-23T14:00:00Z", variant="full")["records"] == before
    later = get_news(store, "TST", "2026-10-23T16:00:00Z", variant="full")
    assert len(later["records"]) == 1
    assert len(later["duplicates_suppressed"]) == 1


def test_allowlist_manifest_and_no_overwrite(store, job, company, tmp_path):
    article(store, job, company)
    result = get_news(store, "TST", "2026-10-23T14:00:00Z", variant="lead", max_chars=70)
    payload = serialize(result["records"], "jsonl")
    assert b"sentiment" not in payload
    assert b"NEVER EXPORT" not in payload
    assert result["records"][0]["truncated"]
    path, manifest = export_news(result, tmp_path / "export.jsonl")
    assert len(path.read_bytes()) > 0
    assert json.loads(manifest.read_text())["record_count"] == 1
    with pytest.raises(ValueError, match="already exists"):
        export_news(result, path)


@pytest.mark.parametrize("raw,precision", [("2026-10-23", "date"), ("2026-10-23T12:00:00", "timezone_missing"), ("not a date", "unknown")])
def test_never_invent_utc(raw, precision):
    assert timestamp(raw)["utc"] is None
    assert timestamp(raw)["precision"] == precision


@pytest.mark.parametrize("value,expected", [
    ("2026-10-23T12:00:00Z", "2026-10-23"),  # pre-open
    ("2026-10-23T19:59:59Z", "2026-10-23"),
    ("2026-10-23T20:00:00Z", "2026-10-26"),  # exactly Friday close
    ("2026-10-24T12:00:00Z", "2026-10-26"),
    ("2026-11-26T12:00:00Z", "2026-11-27"),  # Thanksgiving
    ("2026-11-27T17:59:59Z", "2026-11-27"),
    ("2026-11-27T18:00:00Z", "2026-11-30"),  # 13:00 early close
    ("2026-03-06T21:00:00Z", "2026-03-09"),
    ("2026-03-09T20:00:00Z", "2026-03-10"),  # daylight-saving boundary
    ("2026-11-02T21:00:00Z", "2026-11-03"),
    ("2025-01-09T15:00:00Z", "2025-01-10"),  # exceptional Carter closure
])
def test_actual_exchange_sessions(value, expected):
    assert session_for(value) == expected


def test_aware_cutoff_required(store):
    with pytest.raises(ValueError, match="timezone-aware"):
        get_news(store, "AAPL", "2026-10-23T12:00:00")


def test_date_only_future_is_not_eligible(store, job, company):
    article(store, job, company, published="2026-10-25")
    assert not get_news(store, "TST", "2026-10-23T14:00:00Z", variant="headline")["records"]
    assert not get_news(store, "TST", "2026-10-23T14:00:00Z", variant="full")["records"]


def test_current_calendar_config_cannot_change_past_export(store, job, company):
    article(store, job, company)
    cutoff = "2026-10-23T14:00:00Z"
    before = get_news(store, "TST", cutoff, variant="full")["records"]
    company.calendar = "INVALID"
    store.settings.save_company(company)
    assert get_news(store, "TST", cutoff, variant="full")["records"] == before


def test_repeated_headline_preserves_first_observation(store, job, company):
    first, _ = article(store, job, company)
    article(store, job, company, seen="2026-10-23T15:00:00Z", fetched="2026-10-23T15:01:00Z", extracted="2026-10-23T15:02:00Z")
    record = get_news(store, "TST", "2026-10-23T16:00:00Z")["records"][0]
    assert record["record_id"] == f"headline-{first}"
    assert record["first_seen_at"] == utc("2026-10-23T13:00:00Z")


def test_csv_formula_escaping():
    assert "'=HYPERLINK" in serialize([{"headline": '=HYPERLINK("unsafe")'}], "csv").decode("utf-8-sig")

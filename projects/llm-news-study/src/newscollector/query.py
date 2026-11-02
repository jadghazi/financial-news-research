"""Fail-closed temporal queries and allowlisted model-ready text exports."""
from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import platform
import sqlite3
import uuid
from datetime import date
from pathlib import Path

from . import __version__, RUNTIME_CODE_SHA256, assert_runtime_current
from .core import Candidate, digest, now, publication_day, session_for, utc
from .store import Store


def get_news(store: Store, ticker: str, as_of: str, *, variant: str = "headline", policy: str = "observed",
             since: str | None = None, until: str | None = None, lead_paragraphs: int = 2,
             max_chars: int | None = None, deduplicate: bool = True,
             scopes: list[str] | None = None, include_price_recaps: bool = False,
             cohort: str = "available", include_social: bool = False, job_id: str | None = None) -> dict:
    assert_runtime_current()
    cutoff = utc(as_of)
    if since:
        date.fromisoformat(since)
    if until:
        date.fromisoformat(until)
    if since and until and since > until:
        raise ValueError("Publication end date must not precede the start date.")
    ticker = ticker.upper().strip()
    if policy not in ("observed", "historical"):
        raise ValueError("Archived reconstruction is not verified in this release. Choose observed or explicitly exploratory historical browsing.")
    if variant not in ("headline", "lead", "full"):
        raise ValueError("Text variant must be headline, lead or full.")
    if lead_paragraphs < 1 or (max_chars is not None and max_chars < 1):
        raise ValueError("Paragraph and character limits must be positive.")
    scopes = ["company"] if scopes is None else scopes
    if not scopes or any(s not in ("company", "sector", "market") for s in scopes):
        raise ValueError("Select one or more scopes: company, sector, market.")
    if cohort not in ("available", "body_available"):
        raise ValueError("Cohort must be available or body_available.")
    records, exclusions = [], []
    # Scope before deduplication: an older search's copy must not hide a new run's
    # observation or introduce headlines outside the selected search (D56).
    if job_id:
        jobs = store.rows("SELECT request FROM jobs WHERE id=?", (job_id,))
        if not jobs or ticker not in json.loads(jobs[0]["request"]).get("tickers", []):
            raise ValueError("Choose a saved collection for this ticker.")
    clause, args = (" AND job_id=?", (ticker, job_id)) if job_id else ("", (ticker,))
    if variant == "headline" and cohort == "available":
        review_cutoff = cutoff if policy == "observed" else now()
        blocked = {r["discovery_id"]: r["reason"] for r in store.rows("SELECT discovery_id,reason FROM headline_quarantines WHERE ticker=? AND at<=?", (ticker, review_cutoff))}
        rows = store.rows("SELECT * FROM discoveries WHERE ticker=?" + clause + " ORDER BY first_seen,id", args)
        for row in rows:
            c = Candidate.model_validate_json(row["candidate"])
            pub = c.published.get("utc")
            # Request date limits apply to publication evidence, not arbitrary crawl dates.
            date_value = publication_day(c.published)
            reason = None
            if row["id"] in blocked:
                reason = "Recorded headline quality check: " + blocked[row["id"]]
            elif c.kind == "social_post" and not include_social:
                reason = "Social post excluded; explicitly enable social inputs to include it."
            elif not row["headline_ok"]:
                reason = row["headline_reason"]
            elif not _within(date_value, since, until):
                reason = "Outside requested dates or publication date unknown."
            elif policy == "observed" and max(row["first_seen"], row["assessed_at"]) > cutoff:
                reason = "Headline/assessment had not been observed by cutoff."
            elif policy == "observed" and pub and utc(pub) > cutoff:
                reason = "Reported publication is later than cutoff."
            if reason:
                exclusions.append({"id": f"headline-{row['id']}", "reason": reason})
                continue
            quality = c.metadata.get("assessment", {})
            extra_reason = _scope_reason(quality, scopes, include_price_recaps) or _historical_reason(c.published, cutoff, policy)
            if extra_reason:
                exclusions.append({"id": f"headline-{row['id']}", "reason": extra_reason})
                continue
            records.append({"record_id": f"headline-{row['id']}", "document_id": digest(row["url"])[:24],
                "ticker": ticker, "headline": c.title, "text": c.title, "url": row["url"],
                "publisher": c.publisher, "source": c.source, "published_at": pub,
                "publication_date": date_value, "collection_id": row["job_id"],
                "publication_assertion": c.published,
                "publication_precision": c.published.get("precision", "unknown"),
                "provider_time": c.provider_time.get("utc"), "provider_time_semantic": c.provider_time.get("semantic"),
                "first_seen_at": row["first_seen"], "available_at": max(row["first_seen"], row["assessed_at"]),
                "raw_hash": c.discovery_artifact, "text_hash": digest(c.title), "kind": c.kind,
                "config_hash": digest({"company": c.metadata.get("company_snapshot", {}), "source": c.metadata["source_profile"]}) if c.metadata.get("source_profile") else digest(c.metadata.get("company_snapshot", {})),
                "policy_hash": c.metadata.get("policy_hash"),
                "relevance_scope": quality.get("scope", "company"), "relevance_evidence": quality.get("reason", row["headline_reason"]),
                "source_role": quality.get("role", "unclassified"), "quality_flags": quality.get("flags", []),
                "publisher_group": c.metadata.get("publisher_group", c.publisher),
                "authors": c.metadata.get("authors", []), "license_url": c.metadata.get("license_url", ""),
                "social_provenance": c.metadata.get("social_account"),
                "_calendar": c.metadata.get("company_snapshot", {}).get("calendar", "XNYS")})
    else:
        version_clause = " AND json_extract(metadata,'$.discovery_id') IN (SELECT id FROM discoveries WHERE job_id=?)" if job_id else ""
        rows = store.rows("SELECT * FROM versions WHERE ticker=?" + version_clause + " ORDER BY extracted_at,id", args)
        for row in rows:
            meta = json.loads(row["metadata"])
            pub = json.loads(row["published"])
            provider = json.loads(row["provider_time"])
            available = max(row["first_seen"], row["fetched_at"], row["extracted_at"])
            reason = None
            quarantines = store.rows("SELECT reason FROM content_quarantines WHERE ticker=? AND raw_hash=? AND at<=? ORDER BY at DESC LIMIT 1",
                                     (ticker, row["raw_hash"], cutoff if policy == "observed" else now()))
            if row["kind"] == "social_post" and not include_social:
                reason = "Social post excluded; explicitly enable social inputs to include it."
            elif quarantines:
                reason = "Recorded content quarantine: " + quarantines[0]["reason"]
            elif not row["accepted"]:
                reason = row["reason"]
            elif not _within(publication_day(pub), since, until):
                reason = "Outside requested dates or publication date unknown."
            elif policy == "observed" and available > cutoff:
                reason = "This content version/extraction did not exist locally by cutoff."
            elif policy == "observed" and (meta.get("date_conflict") or meta.get("future_publication")):
                reason = "Conflicting or future publication timestamp; quarantined from strict export."
            elif policy == "observed" and pub.get("utc") and utc(pub["utc"]) > cutoff:
                reason = "Publication is later than cutoff."
            if reason:
                exclusions.append({"id": row["id"], "reason": reason})
                continue
            quality = meta.get("assessment", {})
            extra_reason = _scope_reason(quality, scopes, include_price_recaps) or _historical_reason(pub, cutoff, policy)
            if extra_reason:
                exclusions.append({"id": row["id"], "reason": extra_reason})
                continue
            paragraphs = json.loads(row["paragraphs"])
            content = row["body"] if variant == "full" else "\n\n".join(paragraphs[:lead_paragraphs]) if variant == "lead" else ""
            records.append({"record_id": row["id"], "document_id": row["document_id"],
                "ticker": ticker, "headline": row["title"], "text": row["title"] + ("\n\n" + content if content else ""),
                "url": row["url"], "publisher": row["publisher"], "source": row["source"],
                "published_at": pub.get("utc"), "publication_precision": pub.get("precision", "unknown"),
                "publication_assertion": pub, "publication_date": publication_day(pub),
                "provider_time": provider.get("utc"), "provider_time_semantic": provider.get("semantic"),
                "first_seen_at": row["first_seen"], "available_at": available,
                "raw_hash": row["raw_hash"], "text_hash": row["text_hash"], "kind": row["kind"],
                "extractor": row["extractor"], "config_hash": meta.get("config_hash"),
                "relevance_scope": quality.get("scope", "company"), "relevance_evidence": quality.get("reason", row["reason"]),
                "source_role": quality.get("role", "unclassified"), "quality_flags": quality.get("flags", []),
                "publisher_group": meta.get("publisher_group", row["publisher"]),
                "authors": meta.get("authors", []), "license_url": meta.get("license_url", "")})
            records[-1]["_cohort_text"] = row["title"] + "\n\n" + row["body"]
            records[-1]["content_scope"] = meta.get("content_scope", "publisher_page")
            records[-1]["content_url"] = meta.get("source_url", row["url"])
            if meta.get("pdf_pages"):
                records[-1]["pdf_pages"] = meta["pdf_pages"]
                records[-1]["layout_note"] = meta.get("layout_note")
            records[-1]["_calendar"] = meta.get("company_config", {}).get("calendar", "XNYS")
    # Choose latest eligible version of each URL without looking at later versions.
    by_document = {}
    for record in records:
        previous = by_document.get(record["document_id"])
        fields = ("headline", "text_hash", "publisher", "source", "published_at", "provider_time", "config_hash", "policy_hash")
        # Repeated polling of an unchanged headline must not reset its first-seen time.
        if variant == "headline" and cohort == "available" and previous and all(previous.get(k) == record.get(k) for k in fields):
            continue
        by_document[record["document_id"]] = record
    records = sorted(by_document.values(), key=lambda r: (r["available_at"], r["record_id"]))
    seen, result, duplicates = set(), [], []
    for record in records:
        cohort_text = record.pop("_cohort_text", record["text"])
        normalized = " ".join((cohort_text if cohort == "body_available" else record["text"]).casefold().split())
        if variant == "headline" and cohort == "available":
            # A repeated title on a different date can describe a different event.
            # Generic regulatory titles can also describe multiple distinct notices
            # on one day. Unknown dates provide no basis for cross-URL suppression.
            publication = record.get("publication_assertion", {})
            day = record.get("publication_date") or publication_day(publication)
            if record["kind"] == "regulatory_notice" or len(day) != 10:
                normalized += "\nurl:" + record["url"]
            else:
                normalized += "\ndate:" + day
        match = digest(normalized)
        if deduplicate and match in seen:
            duplicates.append(record["record_id"])
            continue
        seen.add(match)
        record["text_variant"] = variant
        record["cohort"] = cohort
        record["temporal_policy"] = policy
        record["as_of"] = cutoff if policy == "observed" else None
        record["truncated"] = bool(max_chars and len(record["text"]) > max_chars)
        if max_chars:
            record["text"] = record["text"][:max_chars]
        record["exported_text_hash"] = digest(record["text"])
        try:
            record["available_session"] = session_for(record["available_at"], record.pop("_calendar", "XNYS"))
        except Exception:
            record["available_session"] = None
        result.append(record)
    coverage = store.rows("SELECT source,ticker,status,coverage,started_at,finished_at,detail FROM source_runs WHERE ticker=?" + clause + " ORDER BY id", args)
    if policy == "observed":
        coverage = [r for r in coverage if r["finished_at"] and r["finished_at"] <= cutoff]
    return {"records": result, "exclusions": exclusions, "duplicates_suppressed": duplicates,
            "coverage": coverage, "query": {"ticker": ticker, "as_of": cutoff, "variant": variant,
                "policy": policy, "since": since, "until": until, "lead_paragraphs": lead_paragraphs,
                "max_chars": max_chars, "deduplicate": deduplicate, "scopes": scopes,
                "include_price_recaps": include_price_recaps, "cohort": cohort, "include_social": include_social,
                "job_id": job_id, "publication_date_timezone": "America/New_York (date-only assertions unchanged)"},
            "warning": "Historical browsing uses presently downloaded text and is NOT a verified historical prediction dataset." if policy == "historical" else
                       "Observed times describe this collector's actual receipt and processing; source coverage remains bounded."}


def _scope_reason(assessment, scopes, include_price_recaps):
    if assessment.get("scope", "company") not in scopes:
        return "Outside the selected company/sector/market relevance scopes."
    if not include_price_recaps and "price_recap_candidate" in assessment.get("flags", []):
        return "Price-recap candidate excluded by export policy; original retained for review."
    return None


def _historical_reason(publication, cutoff, policy):
    if policy != "historical":
        return None
    if publication.get("utc"):
        return "Publication is later than the requested historical cutoff." if utc(publication["utc"]) > cutoff else None
    raw = publication.get("raw") or ""
    if publication.get("precision") == "date" and len(raw) >= 10 and raw[:10] < cutoff[:10]:
        return None
    return "Cannot establish publication before the historical cutoff (missing/ambiguous time)."


def _within(value: str, since: str | None, until: str | None) -> bool:
    if (since or until) and (not value or len(value) != 10):
        return False
    return (not since or value >= since) and (not until or value <= until)


def serialize(records: list[dict], format: str) -> bytes:
    if format == "jsonl":
        return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records).encode("utf-8")
    if format == "csv":
        fields = list(dict.fromkeys(k for r in records for k in r)) or ["ticker", "headline", "text", "url"]
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=fields)
        writer.writeheader()
        for record in records:
            # Spreadsheet viewers must not execute untrusted article strings as formulas.
            writer.writerow({k: ("'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@", "\t", "\r")) else v) for k, v in record.items()})
        return buf.getvalue().encode("utf-8-sig")
    if format == "parquet":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ValueError("Install the optional parquet extra to export Parquet.") from exc
        output = io.BytesIO()
        pq.write_table(pa.Table.from_pylist(records), output)
        return output.getvalue()
    raise ValueError("Choose jsonl, csv or parquet.")


def manifest(result: dict, payload: bytes, format: str) -> dict:
    assert_runtime_current()
    dependencies = {}
    for name in ("httpx", "feedparser", "trafilatura", "pypdf", "exchange-calendars", "streamlit"):
        dependencies[name] = importlib.metadata.version(name)
    return {"schema_version": "1.0", "collector_version": __version__, "created_at": now(),
            "code_sha256": dict(RUNTIME_CODE_SHA256),
            "python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "dependencies": dependencies,
            "query": result["query"], "warning": result["warning"], "record_count": len(result["records"]),
            "format": format, "data_sha256": digest(payload), "csv_formula_escape": format == "csv",
            "records": [{k: r.get(k) for k in ("record_id", "document_id", "raw_hash", "exported_text_hash", "extractor", "config_hash")} for r in result["records"]],
            "exclusions": result["exclusions"], "duplicates_suppressed": result["duplicates_suppressed"],
            "coverage": result["coverage"], "coverage_complete": False}


def export_news(result: dict, destination: Path, format: str = "jsonl") -> tuple[Path, Path]:
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    companion = destination.with_suffix(destination.suffix + ".manifest.json")
    if destination.exists() or companion.exists():
        raise ValueError("Export already exists. Choose a new filename to preserve its manifest.")
    payload = serialize(result["records"], format)
    meta = manifest(result, payload, format)
    # Exclusive writes prevent silently overwriting scientific evidence.
    with destination.open("xb") as out:
        out.write(payload)
    with companion.open("x", encoding="utf-8") as out:
        json.dump(meta, out, indent=2, ensure_ascii=False)
    return destination, companion

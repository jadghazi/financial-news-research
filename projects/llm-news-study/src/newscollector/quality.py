"""Corpus diagnostics and frozen probability-sample review; never model predictions."""
from __future__ import annotations

import json
import math
import random
import re
import uuid
from collections import Counter
from difflib import SequenceMatcher

from .core import Candidate, digest, now, utc
from .query import _within, get_news


def near_pairs(records: list[dict], limit: int = 500) -> dict:
    """Flag candidates for review only: never silently remove potentially distinct events."""
    subset = records[:limit]
    pairs = []
    for i, left in enumerate(subset):
        a = re.findall(r"\w+", left["headline"].casefold())
        for right in subset[i + 1:]:
            b = re.findall(r"\w+", right["headline"].casefold())
            if min(len(a), len(b)) < 5:
                continue
            protected = lambda t: {x for x in t if any(c.isdigit() for c in x) or x in ("not", "no", "never", "without")}
            if protected(a) != protected(b):
                continue
            # Identical text published on different days can be a recurring announcement.
            if left.get("published_at") and right.get("published_at") and left["published_at"][:10] != right["published_at"][:10]:
                continue
            score = SequenceMatcher(None, a, b, autojunk=False).ratio()
            if score >= .86:
                pairs.append({"left": left["record_id"], "right": right["record_id"], "headline_similarity": round(score, 3)})
    return {"pairs": pairs, "screened_records": len(subset), "truncated_screen": len(records) > limit,
            "meaning": "Heuristic headline similarity for review, not verified event/syndication clusters."}


def corpus_report(store, ticker: str, *, as_of: str | None = None, since=None, until=None, scopes=None):
    cutoff = utc(as_of or now())
    scopes = scopes or ["company", "sector", "market"]
    arguments = dict(policy="observed", since=since, until=until, scopes=scopes, include_price_recaps=True)
    headlines = get_news(store, ticker, cutoff, variant="headline", deduplicate=False, **arguments)
    bodies = get_news(store, ticker, cutoff, variant="full", deduplicate=False, **arguments)
    records = headlines["records"]
    publishers = Counter(r.get("publisher_group") or r["publisher"] for r in records)
    roles = Counter(r["source_role"] for r in records)
    body_publishers = Counter(r.get("publisher_group") or r["publisher"] for r in bodies["records"])
    hhi = sum((n / len(records)) ** 2 for n in publishers.values()) if records else None
    warnings = []
    independent = sum(n for role, n in roles.items() if role in ("independent_reporting", "independent_analysis"))
    if not independent:
        warnings.append("No independently classified publisher headlines in this selection. Unclassified aggregator results do not establish independent origin.")
    if len(publishers) < 3:
        warnings.append("Fewer than three publisher groups observed. This is a diagnostic, not a universal adequacy threshold.")
    if hhi is not None and max(publishers.values()) / len(records) > .7:
        warnings.append("One publisher group supplies more than 70% of eligible headlines; inspect source concentration.")
    precision = Counter(r["publication_precision"] for r in records)
    if precision.get("date") or precision.get("unknown") or precision.get("timezone_missing"):
        warnings.append("Some publication times cannot establish an exact historical decision boundary.")
    if not records:
        warnings.append("No eligible observed headlines; inspect dates, sources and exclusions.")
    return {"schema": "corpus-quality/1", "created_at": now(), "ticker": ticker.upper(), "as_of": cutoff,
            "publication_filter": {"since": since, "until": until}, "scopes": scopes,
            "unique_headline_urls": len(records), "eligible_body_urls": len(bodies["records"]),
            "headline_urls_with_body": len({r["url"] for r in records} & {r["url"] for r in bodies["records"]}),
            "publisher_groups": dict(publishers), "body_publisher_groups": dict(body_publishers),
            "publisher_hhi": hhi, "source_roles": dict(roles),
            "relevance_scopes": dict(Counter(r["relevance_scope"] for r in records)),
            "publication_precision": dict(precision),
            "headline_publication_dates": dict(sorted(Counter((r["published_at"] or r.get("publication_assertion", {}).get("raw") or "unknown")[:10] for r in records).items())),
            "price_recap_candidates": sum("price_recap_candidate" in r["quality_flags"] for r in records),
            "near_duplicate_review": near_pairs(records), "source_runs": headlines["coverage"],
            "warnings": warnings,
            "limits": ["Not a relevance/extraction precision estimate until a frozen sample is independently reviewed.",
                       "Publisher groups are configured or unverified source identities, not audited ownership or wire-origin clusters.",
                       "No claim of all-news recall, complete historical coverage, historical article versions, or forecasting validity.",
                       "Body and headline eligibility are assessed separately; their URL counts have different denominators."]}


def create_audit(store, ticker: str, *, per_group: int = 25, seed: int = 20261024,
                 since=None, until=None, as_of: str | None = None) -> dict:
    if not 1 <= per_group <= 250:
        raise ValueError("Choose 1 to 250 items per accepted/rejected group.")
    cutoff = utc(as_of or now())
    latest = {}
    for row in store.rows("SELECT * FROM discoveries WHERE ticker=? AND first_seen<=? AND assessed_at<=? ORDER BY first_seen,id",
                          (ticker.upper(), cutoff, cutoff)):
        candidate = Candidate.model_validate_json(row["candidate"])
        if not _within((candidate.published.get("utc") or candidate.published.get("raw") or "")[:10], since, until):
            continue
        latest[row["url"]] = (row, candidate)
    frame = {"accepted": [], "rejected": []}
    for row, candidate in latest.values():
        versions = store.rows("SELECT * FROM versions WHERE ticker=? AND document_id=? AND extracted_at<=? AND fetched_at<=? ORDER BY extracted_at DESC,id DESC LIMIT 1",
                              (ticker.upper(), digest(row["url"])[:24], cutoff, cutoff))
        version = versions[0] if versions else None
        frame["accepted" if row["headline_ok"] else "rejected"].append({
            "record_id": f"discovery-{row['id']}", "discovery_id": row["id"], "version_id": version["id"] if version else None,
            "ticker": ticker.upper(), "headline": candidate.title, "body": version["body"] if version else "",
            "url": row["url"], "publisher": candidate.publisher, "source": candidate.source,
            "published": candidate.published, "first_seen": row["first_seen"],
            "raw_hash": version["raw_hash"] if version else candidate.discovery_artifact,
            "headline_accepted": bool(row["headline_ok"]), "body_available": bool(version),
            "body_quarantines": store.rows("SELECT at,reason FROM content_quarantines WHERE ticker=? AND raw_hash=? AND at<=?", (ticker.upper(), version["raw_hash"], cutoff)) if version else [],
            "body_accepted": bool(version["accepted"]) if version else None,
            "collection_reason": row["reason"], "headline_reason": row["headline_reason"]})
    rng = random.Random(seed)
    population, selected = {}, []
    for group, records in frame.items():
        records.sort(key=lambda r: r["record_id"])
        count = min(per_group, len(records))
        population[group] = {"population": len(records), "sample": count,
                             "inclusion_probability": count / len(records) if records else None}
        selected.extend({**r, "stratum": group} for r in rng.sample(records, count))
    rng.shuffle(selected)
    spec = {"schema": "relevance-audit/1", "created_at": now(), "ticker": ticker.upper(), "as_of": cutoff,
            "seed": seed, "since": since, "until": until, "per_group": per_group, "population": population,
            "unit": "latest observed discovery per exact URL; source observations collapsed, syndicated URLs remain distinct",
            "selection": "simple random sample without replacement within headline-accepted and headline-rejected strata",
            "purpose": "headline relevance; separately assess body faithfulness and time evidence where available",
            "limits": "Not recall against all published news, event-level independence, or historical text-version certification.",
            "records": selected}
    spec["sample_sha256"] = digest(json.dumps(selected, sort_keys=True, ensure_ascii=False))
    spec["audit_id"] = uuid.uuid4().hex[:20]
    store.execute("INSERT INTO audits(id,created_at,spec) VALUES(?,?,?)", (spec["audit_id"], spec["created_at"], json.dumps(spec, ensure_ascii=False)))
    return spec


def save_audit_review(store, audit_id, record_id, reviewer, relevance, text_quality, timing, note=""):
    rows = store.rows("SELECT spec FROM audits WHERE id=?", (audit_id,))
    if not rows or record_id not in {r["record_id"] for r in json.loads(rows[0]["spec"])["records"]}:
        raise ValueError("Review item is not in the frozen audit sample.")
    if relevance not in ("Relevant", "Irrelevant", "Uncertain"):
        raise ValueError("Choose Relevant, Irrelevant or Uncertain.")
    if text_quality not in ("Faithful", "Incomplete/boilerplate", "Cannot assess") or timing not in ("Supported", "Conflicting", "Uncertain"):
        raise ValueError("Use the documented text-quality and time-evidence labels.")
    if not reviewer.strip():
        raise ValueError("A reviewer name is required.")
    store.execute("INSERT OR REPLACE INTO audit_reviews VALUES(?,?,?,?,?,?,?,?)",
                  (audit_id, record_id, reviewer.strip(), relevance, text_quality, timing, note, now()))


def _wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p, denominator = successes / total, 1 + z*z / total
    center = (p + z*z / (2*total)) / denominator
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-half), min(1, center+half)]


def audit_report(store, audit_id, reviewer):
    rows = store.rows("SELECT spec FROM audits WHERE id=?", (audit_id,))
    if not rows:
        raise ValueError("Unknown audit.")
    spec = json.loads(rows[0]["spec"])
    reviews = {r["record_id"]: r for r in store.rows("SELECT * FROM audit_reviews WHERE audit_id=? AND reviewer=?", (audit_id, reviewer))}
    strata = {}
    for group, population in spec["population"].items():
        chosen = [r for r in spec["records"] if r["stratum"] == group]
        labels = [reviews[r["record_id"]] for r in chosen if r["record_id"] in reviews]
        counts = Counter(r["relevance"] for r in labels)
        complete = len(labels) == len(chosen) and len(chosen) > 0
        unambiguous = complete and not counts["Uncertain"]
        strata[group] = {**population, "reviewed": len(labels), "labels": dict(counts), "complete": complete,
                         "fraction_relevant": counts["Relevant"] / len(labels) if unambiguous else None,
                         "wilson_95_interval": _wilson(counts["Relevant"], len(labels)) if unambiguous and len(labels) < population["population"] else None,
                         "text_quality": dict(Counter(r["text_quality"] for r in labels)),
                         "timing_evidence": dict(Counter(r["timing"] for r in labels))}
    return {"audit_id": audit_id, "reviewer": reviewer, "strata": strata,
            "interpretation": "Accepted-group relevance estimates headline precision for this frozen URL frame. Rejected-group relevant fraction measures missed items among discovered rejected URLs; it is not all-news recall.",
            "limits": "Intervals require complete, unambiguous labels and assume the URL sample is appropriate; related stories and reviewer error can invalidate independent-binomial uncertainty. Census labels do not need sampling inference. Human review and event-level validation remain necessary."}

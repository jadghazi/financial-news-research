"""Pinned SPNews acquisition and DEVELOPMENT-only index cohort preparation.

CSV/text only. No author-code execution, embeddings, training, or 2024 price
conversion. Snapshot dates are UNVERIFIED observation-day proxies, not actual
collection timestamps. Source files remain compressed, content-verified, and
resumable. The publication-only alternative is explicitly retrospective.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time
import urllib.parse
import urllib.request

REVISION = "d19aa5e81cd1768d8b0958cc2a1ed505ba1e5b0a"
REPOSITORY = "FreddieNIU/Financial-Graph-Evaluation"
DAILY = "data/SPNews_v2/raw/news/preprocessed_daily_2/"
PRICE = "data/SPNews_v2/raw/sp500index.csv"
START, END, BOUNDARY = "2022-09-20", "2023-12-31", "2024-01-01"
BUDGET = 650_000_000
DESCRIPTION = "The S&P 500 index, a broad US large-cap equity market index."
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "AdjClose", "Volume")
REQUIRED_NEWS = {"Id", "Company", "Date", "Title", "Publisher", "Link", "ProviderPublishTime", "Related Company"}


def _sha(raw): return hashlib.sha256(raw).hexdigest()
def _blob(raw): return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
def _iso(value): return value.isoformat().replace("+00:00", "Z")
def _utc(day, hour=0): return datetime.fromisoformat(day).replace(hour=hour, tzinfo=timezone.utc)
def _norm(value): return " ".join(str(value).split())


def _write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "FININ-SPNews-cohort/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read(), {"url": response.geturl(), "status": response.status}
        except Exception:
            if attempt == 2: raise
            time.sleep(attempt + 1)


def _epoch(value):
    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer(): return None
        result = datetime.fromtimestamp(number, timezone.utc)
        return result if 2000 <= result.year < 2030 else None
    except (ValueError, TypeError, OverflowError, OSError):
        return None


def parse_stories(snapshots):
    """Iterate ordered (ISO date, CSV bytes), retaining first observable version.

    Later publication/title revisions and company links never revise an earlier
    story. Deduplicate by BOTH source ID and exact normalized title (casefolded).
    Initial-day IDs/titles are blacklisted even if their rows are malformed.
    """
    csv.field_size_limit(100_000_000)
    seen_ids, seen_titles, blocked_ids, blocked_titles = set(), set(), set(), set()
    stories, daily, total = [], [], Counter()
    previous = None
    for snapshot, raw in snapshots:
        date.fromisoformat(snapshot)
        if previous is not None and snapshot <= previous:
            raise ValueError("Snapshot dates must be unique and chronological")
        initial = previous is None
        previous = snapshot
        reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig", errors="strict")))
        if not REQUIRED_NEWS.issubset(reader.fieldnames or []):
            raise ValueError(f"Fatal news schema mismatch at {snapshot}")
        counts = Counter(); first_this_day = {}; title_this_day = {}; aliases = {}
        observed_ids, observed_titles = set(), set()
        proxy = _utc(snapshot) + timedelta(days=2)
        for row_number, row in enumerate(reader, 2):
            counts["raw_rows"] += 1
            source_id = _norm(row.get("Id", ""))
            title = str(row.get("Title", "")).strip()  # Preserve original internal whitespace and Unicode.
            normalized = _norm(title).casefold()
            if initial:
                if source_id: blocked_ids.add(source_id)
                if normalized: blocked_titles.add(normalized)
            if None in row or any(value is None for value in row.values()):
                counts["rejected_malformed_csv"] += 1; continue
            # Filename consistency is required; an unknown collector timezone is
            # retained as a limitation rather than silently inferred.
            pieces = _norm(row["Date"]).split("-")
            try: observed = date(*map(int, pieces)).isoformat()
            except (ValueError, TypeError): observed = None
            if observed != snapshot:
                raise ValueError(f"Fatal snapshot Date/filename inconsistency: {snapshot}:{row_number}")
            published = _epoch(row["ProviderPublishTime"])
            if published is None:
                counts["rejected_invalid_publication_epoch"] += 1; continue
            if published > proxy:
                raise ValueError(f"Fatal publication beyond observation-day guard: {snapshot}:{row_number}")
            if not source_id or not title:
                counts["rejected_empty_id_or_title"] += 1; continue
            if normalized in {"title", "headline", "headlines", "nan", "none", "null", "id", "providerpublishtime"} or "\ufffd" in title:
                counts["rejected_header_or_encoding"] += 1; continue
            observed_ids.add(source_id); observed_titles.add(normalized)
            if initial:
                counts["excluded_initial_backlog_rows"] += 1; continue
            if source_id in blocked_ids or normalized in blocked_titles:
                counts["excluded_backlog_reappearances"] += 1; continue
            if source_id in seen_ids or normalized in seen_titles:
                counts["excluded_previously_observed_id_or_title"] += 1; continue
            company, related = _norm(row["Company"]), _norm(row["Related Company"])
            if source_id in aliases:
                prior = first_this_day[aliases[source_id]]
                if prior["normalized_title"] != normalized:
                    counts["same_day_title_revisions_ignored"] += 1
                else:
                    if company: prior["company_links"].add(company)
                    if related: prior["related_company_links"].add(related)
                counts["same_snapshot_duplicate_id"] += 1
                continue
            if normalized in title_this_day:
                prior = first_this_day[title_this_day[normalized]]
                prior["source_ids"].add(source_id)
                aliases[source_id] = title_this_day[normalized]
                if company: prior["company_links"].add(company)
                if related: prior["related_company_links"].add(related)
                # Also remember this source ID to prevent later revisions.
                counts["same_snapshot_duplicate_title"] += 1
                continue
            first_this_day[source_id] = {
                "id": _sha(title.encode("utf-8")), "text": title,
                "source_id": source_id, "source_ids": {source_id},
                "normalized_title": normalized, "created_at": _iso(published),
                "available_at": _iso(max(published, proxy)),
                "first_snapshot_date": snapshot, "first_snapshot_row": row_number,
                "publisher": row["Publisher"], "link": row["Link"],
                "company_links": {company} if company else set(),
                "related_company_links": {related} if related else set(),
            }
            title_this_day[normalized] = source_id
            aliases[source_id] = source_id
        for story in first_this_day.values():
            seen_ids.update(story["source_ids"]); seen_titles.add(story["normalized_title"])
            for key in ("source_ids", "company_links", "related_company_links"):
                story[key] = sorted(story[key])
            stories.append(story)
        seen_ids.update(observed_ids); seen_titles.update(observed_titles)
        counts["new_deduplicated_stories"] = len(first_this_day)
        daily.append({"snapshot_date": snapshot, **dict(counts)})
        total.update(counts)
    return stories, {"daily": daily, "totals": dict(total), "initial_backlog_ids": len(blocked_ids),
                     "initial_backlog_titles": len(blocked_titles), "unique_stories": len(stories)}


def read_prices(raw, boundary=BOUNDARY):
    """Dates beyond the boundary may be read; their numeric cells never parsed."""
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = set(reader.fieldnames or [])
    adjusted = "AdjClose" if "AdjClose" in fields else "Adj Close"
    columns = (*PRICE_COLUMNS[:4], adjusted, "Volume")
    if not {"Date", *columns}.issubset(fields): raise ValueError("Fatal price schema mismatch")
    calendar, prices = [], {}
    for row in reader:
        day = date.fromisoformat(row["Date"]).isoformat()
        calendar.append(day)
        if day >= boundary: continue
        values = [float(row[key]) for key in columns]
        if not all(math.isfinite(x) for x in values) or min(values[:5]) <= 0 or values[5] < 0:
            raise ValueError("Invalid development price numeric data")
        prices[day] = values
    if len(set(calendar)) != len(calendar): raise ValueError("Duplicate index price dates")
    return sorted(calendar), prices


def _date_runs(days):
    runs = []
    for day in sorted(set(days)):
        if runs and date.fromisoformat(day) == date.fromisoformat(runs[-1]["end"]) + timedelta(days=1):
            runs[-1]["end"] = day; runs[-1]["days"] += 1
        else: runs.append({"start": day, "end": day, "days": 1})
    return runs


def _window_coverage(previous, base, snapshot_dates):
    """Observation dates whose Date+2day midnight lies inside this window.

    This describes FILE presence under the proxy rule, never validates collection
    timing or identifies true market-wide no-news days. Weekend windows may need
    several observation dates. Unknown snapshot inventory cannot imply coverage.
    """
    left = date.fromisoformat(previous) - timedelta(days=1)
    right = date.fromisoformat(base) - timedelta(days=2)
    expected = [(left + timedelta(days=i)).isoformat() for i in range((right-left).days+1)]
    observed = set(snapshot_dates) if snapshot_dates is not None else set()
    missing = [day for day in expected if day not in observed]
    return {"proxy_window_snapshot_dates_expected": expected,
            "proxy_window_snapshot_dates_observed": [day for day in expected if day in observed],
            "proxy_window_snapshot_dates_missing": missing,
            "collection_window_complete": not missing if snapshot_dates is not None else None,
            "collection_coverage_reference": "File inventory for UNVERIFIED Date+2day proxy window; not proof of actual publication/collection coverage"}


def build_cohort(stories, calendar, prices, *, policy="snapshot_proxy", start=START, boundary=BOUNDARY, cap=64, snapshot_dates=None):
    if policy not in {"snapshot_proxy", "publication_only_retrospective"}: raise ValueError("Unknown clock policy")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1: raise ValueError("Invalid headline cap")
    clock = "available_at" if policy == "snapshot_proxy" else "created_at"
    ordered = sorted(stories, key=lambda story: (story[clock], story["id"]))
    times = [story[clock] for story in ordered]
    rows = []
    for position in range(1, len(calendar) - 1):
        previous, base, target = calendar[position - 1:position + 2]
        if base < start or target >= boundary: continue
        if any(day not in prices for day in (previous, base, target)):
            raise ValueError("Missing development price session; never skip to a later target")
        window, cutoff = _iso(_utc(previous, 22)), _iso(_utc(base, 22))
        a, b = bisect_right(times, window), bisect_right(times, cutoff)
        eligible = ordered[a:b]
        chosen = sorted(eligible, key=lambda message: message["id"])[:cap]
        # Sorting by SHA256 is stable hash selection, independent of news age,
        # direction, source row order and any observed labels.
        messages = []
        for story in chosen:
            message = {key: story[key] for key in story if key != "normalized_title"}
            if policy != "snapshot_proxy": message["available_at"] = message["created_at"]
            message["publication_age_hours"] = (_utc(base, 22) - datetime.fromisoformat(message["created_at"].replace("Z", "+00:00"))).total_seconds()/3600
            message["availability_age_hours"] = (_utc(base, 22) - datetime.fromisoformat(message["available_at"].replace("Z", "+00:00"))).total_seconds()/3600
            messages.append(message)
        raw, prior = prices[base], prices[previous]
        relative = [raw[i]/prior[3] - 1 for i in range(4)] + [raw[4]/prior[4] - 1, math.log1p(raw[5])]
        coverage = _window_coverage(previous, base, snapshot_dates)
        if coverage["collection_window_complete"] is None:
            coverage["news_absence_status"] = "collection_inventory_unknown"
        elif coverage["collection_window_complete"]:
            coverage["news_absence_status"] = "eligible_stories_present_observed_snapshots" if eligible else "no_new_eligible_stories_in_observed_snapshots"
        else:
            coverage["news_absence_status"] = "eligible_stories_present_incomplete_snapshots" if eligible else "no_eligible_stories_collection_incomplete"
        coverage["collection_missing"] = coverage["collection_window_complete"] is not True
        coverage["missing_collection_dates"] = coverage["proxy_window_snapshot_dates_missing"]
        coverage["observed_empty_news"] = not eligible and coverage["collection_window_complete"] is True and policy == "snapshot_proxy"
        coverage["coverage_status"] = ("unverified_publication_vintage" if policy != "snapshot_proxy"
                                       else "inventory_unknown" if snapshot_dates is None
                                       else "missing_collection_snapshots" if coverage["collection_missing"]
                                       else "observed_proxy_snapshots_complete")
        rows.append({"id": f"post_close:SP500:{target}", "ticker": "SP500", "forecast_schedule": "post_close",
                     "forecast_date": base, "price_date": base, "cutoff_utc": cutoff,
                     "availability_window_start_utc": window, "price_available_utc": cutoff,
                     "target_date": target, "target_available_utc": _iso(_utc(target, 22)),
                     "target_up": int(prices[target][4] > raw[4]), "target_return": prices[target][4]/raw[4]-1,
                     "split": "train" if target <= "2023-01-31" else "valid",
                     "description": DESCRIPTION, "market_raw": list(raw), "market_relative": relative,
                     "headlines": messages, "news_count_precap": len(eligible), "news_count_retained": len(messages),
                     "availability_policy": policy, **coverage})
    return rows


def _coverage(rows):
    longest = current = incomplete_run = longest_incomplete = 0
    for row in rows:
        current = current + 1 if row["news_count_precap"] == 0 else 0
        longest = max(longest, current)
        incomplete_run = incomplete_run + 1 if row["collection_window_complete"] is False else 0
        longest_incomplete = max(longest_incomplete, incomplete_run)
    counts = sorted(row["news_count_precap"] for row in rows)
    return {"sessions": len(rows), "empty_news_sessions": sum(x == 0 for x in counts),
            "longest_empty_session_run": longest, "capped_sessions": sum(x > 64 for x in counts),
            "precap_min": min(counts, default=0), "precap_median": counts[len(counts)//2] if counts else 0,
            "precap_max": max(counts, default=0),
            "collection_incomplete_sessions": sum(row["collection_window_complete"] is False for row in rows),
            "longest_collection_incomplete_session_run": longest_incomplete,
            "empty_with_observed_collection_sessions": sum(row["observed_empty_news"] for row in rows),
            "empty_with_incomplete_collection_sessions": sum(row["news_count_precap"] == 0 and row["collection_window_complete"] is False for row in rows)}


def run(out: Path, audit: Path, *, workers=4):
    """Resume content-verified acquisition; refuse protocol changes/overwrite."""
    out, audit = Path(out), Path(audit)
    source_hash = _sha(Path(__file__).read_bytes())
    tree_bytes = (audit / "source_tree.json").read_bytes()
    tree = json.loads(tree_bytes)
    audit_manifest = json.loads((audit / "manifest.json").read_text(encoding="utf-8"))
    if audit_manifest["revision"] != REVISION or tree.get("truncated") or audit_manifest["source_tree_sha256"] != _sha(tree_bytes):
        raise ValueError("Pinned source-tree audit provenance mismatch")
    entries = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    selected = sorted(path for path in entries if path.startswith(DAILY) and path.endswith(".csv") and START <= Path(path).stem <= END)
    expected_bytes = sum(entries[path]["size"] for path in selected) + entries[PRICE]["size"]
    if not selected or Path(selected[0]).stem != START or expected_bytes > BUDGET:
        raise ValueError("Pinned acquisition does not fit prescribed dates/budget")
    protocol = {"revision": REVISION, "repository": REPOSITORY, "start": START, "end": END,
                "development_target_before": BOUNDARY, "include_test": False,
                "selection_not_based_on_accuracy": True, "source_code_sha256": source_hash,
                "tree_sha256": _sha(tree_bytes), "raw_transfer_budget": BUDGET,
                "selected_raw_bytes": expected_bytes, "daily_files": len(selected),
                "clock": "UNVERIFIED observation-day proxy: max(publication, snapshot Date+2days00UTC)",
                "alternative": "publication-only retrospective, unavailable-vintage assumption",
                "first_snapshot": "blacklist all observed initial IDs and normalized titles",
                "headline_text": "strip surrounding whitespace only; original internal whitespace/Unicode retained; dedup uses casefolded whitespace-normalized titles",
                "split_annotation": "target<=2023-01-31 train, remainder2023 valid; fresh temporal fold indices MUST be declared by runner",
                "no_model_fit_or_inference": True}
    if out.exists():
        if not (out / "protocol_before_acquisition.json").exists():
            raise ValueError("Existing namespace protocol mismatch; refusing overwrite")
        frozen = json.loads((out / "protocol_before_acquisition.json").read_text())
        if frozen != protocol:
            amendment_paths = list(out.glob("source_*amendment.json"))
            amendments = [json.loads(path.read_text()) for path in amendment_paths]
            prior_choices = {key: value for key, value in frozen.items() if key != "source_code_sha256"}
            current_choices = {key: value for key, value in protocol.items() if key != "source_code_sha256"}
            authorized_hash = any(amendment.get("original_source_sha256") == frozen["source_code_sha256"] and amendment.get("repaired_source_sha256") == source_hash for amendment in amendments)
            if prior_choices != current_choices or not authorized_hash:
                raise ValueError("Existing namespace protocol mismatch; refusing overwrite")
            protocol = frozen  # Preserve original pre-acquisition protocol.
    else:
        out.mkdir(parents=True)
        _write(out / "protocol_before_acquisition.json", protocol)
        (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
        (out / "source_tree.json").write_bytes(tree_bytes)
    manifest_path = out / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"revision": REVISION, "files": {}, "network_download_bytes": 0, "reused_audit_bytes": 0}
    pending = []
    def validate(raw, remote_path):
        entry = entries[remote_path]
        if len(raw) != entry["size"] or _blob(raw) != entry["sha"]:
            raise ValueError(f"Pinned source hash/size disagreement: {remote_path}")
    for remote_path in [PRICE, *selected]:
        local = out / "sources_gzip" / (remote_path + ".gz")
        if remote_path in manifest["files"]:
            compressed = local.read_bytes()
            if _sha(compressed) != manifest["files"][remote_path]["gzip_sha256"]:
                raise ValueError("Committed compressed source SHA256 mismatch")
            raw = gzip.decompress(compressed); validate(raw, remote_path)
            if _sha(raw) != manifest["files"][remote_path]["raw_sha256"]:
                raise ValueError("Committed raw source SHA256 mismatch")
        elif local.exists():
            # Recover an atomically completed write if a prior process stopped
            # before its manifest commit. Trust only the pinned blob/size.
            compressed = local.read_bytes(); raw = gzip.decompress(compressed); validate(raw, remote_path)
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/" + urllib.parse.quote(remote_path, safe="/")
            manifest["files"][remote_path] = {"revision": REVISION, "url": url, "raw_bytes": len(raw),
                "raw_sha256": _sha(raw), "git_blob_sha1": _blob(raw), "gzip_sha256": _sha(compressed),
                "gzip_bytes": len(compressed), "recovered_verified_uncommitted_file": True}
            manifest["network_download_bytes"] += len(raw)
            _write(manifest_path, manifest)
        else: pending.append((remote_path, local))
    def fetch(pair):
        remote_path, local = pair
        url = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/" + urllib.parse.quote(remote_path, safe="/")
        reuse = audit / "sources" / remote_path
        if reuse.exists(): raw, metadata = reuse.read_bytes(), {"url": url, "reused_from_audit": str(reuse.resolve())}
        else: raw, metadata = _get(url)
        validate(raw, remote_path)
        compressed = gzip.compress(raw, compresslevel=6, mtime=0)
        local.parent.mkdir(parents=True, exist_ok=True)
        temporary = local.with_suffix(".gz.tmp"); temporary.write_bytes(compressed); temporary.replace(local)
        return remote_path, {"revision": REVISION, "raw_bytes": len(raw), "raw_sha256": _sha(raw), "git_blob_sha1": _blob(raw),
                             "gzip_sha256": _sha(compressed), "gzip_bytes": len(compressed), **metadata}
    acquisition_errors = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, pair): pair[0] for pair in pending}
        for future in as_completed(futures):
            try: remote_path, entry = future.result()
            except Exception as exc:
                acquisition_errors.append({"source_path": futures[future], "error_type": type(exc).__name__, "error": str(exc)})
                continue  # Commit other successfully completed files before stopping.
            manifest["files"][remote_path] = entry
            key = "reused_audit_bytes" if "reused_from_audit" in entry else "network_download_bytes"
            manifest[key] += entry["raw_bytes"]
            _write(manifest_path, manifest)
            count = len(manifest["files"])
            if count % 25 == 0 or count == len(selected) + 1:
                print(f"committed {count}/{len(selected)+1} sources; network {manifest['network_download_bytes']/1e6:.1f}MB", flush=True)
    if acquisition_errors:
        failed_record = out / f"acquisition_failure_{time.time_ns()}.json"
        _write(failed_record, {"errors": acquisition_errors, "committed_files": len(manifest["files"]),
                              "no_cohort_preparation_started": True})
        raise RuntimeError(f"{len(acquisition_errors)} source requests failed; verified completed files retained for resume")
    if (out / "preparation_manifest.json").exists():
        prepared = json.loads((out / "preparation_manifest.json").read_text())
        for relative, digest in prepared["output_sha256"].items():
            if _sha((out / relative).read_bytes()) != digest: raise ValueError("Prepared output checksum mismatch")
        return prepared
    try:
        stories, statistics = parse_stories((Path(path).stem, gzip.decompress((out / "sources_gzip" / (path + ".gz")).read_bytes())) for path in selected)
    except Exception as exc:
        _write(out / f"preparation_failure_{time.time_ns()}.json", {"error_type": type(exc).__name__, "error": str(exc), "no_cohort_labels_prepared": True})
        raise
    calendar, prices = read_prices(gzip.decompress((out / "sources_gzip" / (PRICE + ".gz")).read_bytes()))
    observed = {Path(path).stem for path in selected}
    expected_days = [_iso(_utc(START) + timedelta(days=i))[:10] for i in range((date.fromisoformat(END)-date.fromisoformat(START)).days+1)]
    missing = [day for day in expected_days if day not in observed]
    provenance = {**protocol, "preparation_code_sha256": source_hash, "common_calendar": calendar,
                  "source_manifest_sha256": _sha(manifest_path.read_bytes()),
                  "price_vintage": "Pinned revised source; actual historical price vintage unverified",
                  "preprocessing": "raw and deterministic relative channels only; no learned transform fitted",
                  "limitations": ["Date is not a verified collection timestamp; two-day clock is a sensitivity assumption", "No point-in-time claims", "Potential pretrained-corpus overlap for later encoders", "No Reuters instrument sentiment", "No historical constituent filtering", "Missing snapshots can hide unavailable reports"],
                  "snapshot_missing_dates": missing, "snapshot_missing_date_runs": _date_runs(missing),
                  "collection_coverage_semantics": "Missing source snapshot files differ from observed snapshots with no new eligible stories; all index sessions retained, no outcome-based exclusions"}
    output_paths = []
    _write(out / "story_audit.json", statistics); output_paths.append("story_audit.json")
    with (out / "first_version_headlines.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
        for story in stories: stream.write(json.dumps(story, ensure_ascii=False, allow_nan=False) + "\n")
    output_paths.append("first_version_headlines.jsonl")
    summaries = {}
    for policy, folder in (("snapshot_proxy", "primary"), ("publication_only_retrospective", "publication_only_retrospective")):
        rows = build_cohort(stories, calendar, prices, policy=policy, snapshot_dates=observed)
        directory = out / folder; directory.mkdir(exist_ok=True)
        with (directory / "devsamples_post_close.jsonl").open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows: stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        _write(directory / "provenance.json", {**provenance, "availability_policy": policy,
                                               "publication_only_retrospective": policy != "snapshot_proxy"})
        output_paths.extend([f"{folder}/devsamples_post_close.jsonl", f"{folder}/provenance.json"])
        summaries[policy] = _coverage(rows)
    prepared = {"include_test": False, "development_target_before": BOUNDARY, "sources": len(manifest["files"]),
                "network_download_bytes": manifest["network_download_bytes"], "reused_audit_bytes": manifest["reused_audit_bytes"],
                "network_byte_accounting": "Completed raw payload bytes only; failed request partial transfers and transport overhead not measured",
                "missing_snapshot_dates": missing, "missing_snapshot_date_runs": _date_runs(missing), "unique_stories": len(stories), "coverage": summaries,
                "no_prediction_quality_measured": True, "output_sha256": {relative: _sha((out / relative).read_bytes()) for relative in output_paths}}
    _write(out / "preparation_manifest.json", prepared)
    print(json.dumps({**{key: value for key, value in prepared.items() if key not in {"output_sha256", "missing_snapshot_dates"}},
                      "missing_snapshot_date_count": len(missing)}, indent=2), flush=True)
    return prepared


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--audit", required=True, type=Path)
    args = parser.parse_args(); run(args.out, args.audit)

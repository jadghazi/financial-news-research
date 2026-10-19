"""Bounded safe CSV/source audit of the pinned author SPNews release.

Downloads eight initial daily files and three text sources plus index prices.
Never executes author code, unpickles objects, computes returns or trains.
Outputs stay in a fresh audit namespace; publication-time and availability
assumptions are distinguished explicitly.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request


REVISION = "d19aa5e81cd1768d8b0958cc2a1ed505ba1e5b0a"
REPOSITORY = "FreddieNIU/Financial-Graph-Evaluation"
DAILY = "data/SPNews_v2/raw/news/preprocessed_daily_2/"
TICKERWISE = "data/SPNews_v2/raw/news/preprocessed_tickerwise/"


def _hash(contents): return hashlib.sha256(contents).hexdigest()


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "FININ-academic-data-audit/1.0"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), {"url": response.geturl(), "status": response.status,
                                         "content_type": response.headers.get("Content-Type")}
        except Exception:
            if attempt == 2: raise
            time.sleep(1 + attempt)


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _parse_epoch(value):
    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer(): return None
        timestamp = datetime.fromtimestamp(number, tz=timezone.utc)
        if not 2000 <= timestamp.year <= 2030: return None
        return timestamp
    except (TypeError, ValueError, OverflowError, OSError): return None


def _snapshot_date(value):
    pieces = str(value).strip().split("-")
    try:
        return date(*[int(piece) for piece in pieces]) if len(pieces) == 3 else None
    except (TypeError, ValueError): return None


def run(out: Path) -> dict:
    out = Path(out)
    if out.exists(): raise FileExistsError("Refusing to overwrite an earlier audit namespace")
    out.mkdir(parents=True)
    _write(out / "protocol_before_acquisition.json", {
        "repository": REPOSITORY, "revision": REVISION,
        "sample": "first eight daily CSVs, convenience sample, no performance evaluation",
        "restrictions": "CSV/UTF8 text only; no pickle, author-code execution, model inference or return computation",
        "source_code_sha256": _hash(Path(__file__).read_bytes()),
    })
    (out / "audit_source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    url = f"https://api.github.com/repos/{REPOSITORY}/git/trees/{REVISION}?recursive=1"
    tree_bytes, response = _get(url)
    (out / "source_tree.json").write_bytes(tree_bytes)
    tree = json.loads(tree_bytes)
    if tree.get("truncated"): raise ValueError("Source tree is incomplete")
    files = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    daily = sorted((entry for path, entry in files.items() if path.startswith(DAILY) and path.endswith(".csv")), key=lambda entry: entry["path"])
    tickerwise = [entry for path, entry in files.items() if path.startswith(TICKERWISE)]
    chosen = daily[:8]
    if [Path(entry["path"]).stem for entry in chosen] != [f"2022-09-{day:02d}" for day in range(20, 28)]:
        raise ValueError("Pinned convenience sample differs from anticipated dates")
    selected = ["README.md", "code/Generate_dataset.py", "code/utils.py", "data/SPNews_v2/raw/sp500index.csv"] + [entry["path"] for entry in chosen]
    manifest = {"repository": REPOSITORY, "revision": REVISION, "source_tree_sha256": _hash(tree_bytes),
                "tree_response": response, "acquired_files": {},
                "daily_total": {"files": len(daily), "bytes": sum(entry["size"] for entry in daily)},
                "tickerwise_total": {"files": len(tickerwise), "bytes": sum(entry["size"] for entry in tickerwise)}}
    for remote_path in selected:
        remote = f"https://raw.githubusercontent.com/{REPOSITORY}/{REVISION}/" + urllib.parse.quote(remote_path, safe="/")
        contents, metadata = _get(remote)
        entry = files[remote_path]
        git_blob = hashlib.sha1(b"blob " + str(len(contents)).encode() + b"\0" + contents).hexdigest()
        if git_blob != entry["sha"] or len(contents) != entry["size"]:
            raise ValueError("Remote contents disagree with pinned git tree")
        local = out / "sources" / remote_path
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(contents)
        manifest["acquired_files"][remote_path] = {"bytes": len(contents), "sha256": _hash(contents), "git_blob_sha1": git_blob, **metadata}
        _write(out / "manifest.json", manifest)
        print(f"acquired {remote_path}: {len(contents):,} bytes", flush=True)
    summaries, observations = [], []
    id_dates, id_companies, id_titles = defaultdict(set), defaultdict(set), defaultdict(set)
    headline_dates = defaultdict(set); first_seen = {}; prior_ids = set()
    csv.field_size_limit(100_000_000)
    for entry in chosen:
        contents = (out / "sources" / entry["path"]).read_bytes()
        text = contents.decode("utf-8-sig", errors="strict")
        reader = csv.DictReader(io.StringIO(text))
        snapshot = date.fromisoformat(Path(entry["path"]).stem)
        day_end = datetime.combine(snapshot + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
        seen = set(); counts = Counter(); publication_dates = Counter(); invalid_examples = []
        titles = set(); companies = set(); firstseen_ids = set()
        for number, row in enumerate(reader, 2):
            counts["rows"] += 1
            if None in row or any(value is None for value in row.values()): counts["malformed_csv_rows"] += 1
            identifier, title, company = (str(row.get(key, "")).strip() for key in ("Id", "Title", "Company"))
            normalized_title = " ".join(title.split())
            if not identifier or not title: counts["empty_id_or_headline"] += 1
            if "\ufffd" in title: counts["headline_replacement_character_rows"] += 1
            if "\ufffd" in str(row.get("Text", "")): counts["body_replacement_character_rows"] += 1
            observed_date = _snapshot_date(row.get("Date"))
            if observed_date is None: counts["invalid_snapshot_date"] += 1
            elif observed_date != snapshot: counts["Date_filename_mismatch"] += 1
            published = _parse_epoch(row.get("ProviderPublishTime"))
            if published is None:
                counts["invalid_publication_epoch"] += 1
                if len(invalid_examples) < 5: invalid_examples.append({"line": number, "value": row.get("ProviderPublishTime")})
            else:
                publication_dates[published.date().isoformat()] += 1
                if published >= day_end: counts["published_after_snapshot_utc_day_end"] += 1
                if published.date() < snapshot: counts["published_before_snapshot_date"] += 1
                if published.date() == snapshot: counts["published_on_snapshot_utc_date"] += 1
            if identifier in seen: counts["duplicate_id_rows_within_snapshot"] += 1
            if identifier in prior_ids: counts["id_row_repeated_from_prior_snapshot"] += 1
            if identifier and identifier not in first_seen:
                first_seen[identifier] = snapshot.isoformat(); firstseen_ids.add(identifier)
            if identifier:
                id_dates[identifier].add(snapshot.isoformat()); id_companies[identifier].add(company); id_titles[identifier].add(normalized_title)
                seen.add(identifier)
            headline_dates[normalized_title].add(snapshot.isoformat()); titles.add(normalized_title); companies.add(company)
            observations.append({"id": identifier, "snapshot_date": snapshot.isoformat(), "company": company,
                                 "title_sha256": _hash(normalized_title.encode()),
                                 "publication_utc": published.isoformat() if published else None})
        summaries.append({"snapshot_date": snapshot.isoformat(), "columns": reader.fieldnames,
                          "counts": dict(counts), "unique_ids": len(seen), "unique_headlines": len(titles),
                          "companies": len(companies), "firstseen_unique_ids": len(firstseen_ids),
                          "firstseen_recommendation": "exclude initial backlog" if snapshot == date(2022, 9, 20) else "assign conservative first-snapshot availability, never publication-date backfill",
                          "publication_dates": dict(publication_dates), "invalid_epoch_examples": invalid_examples})
        prior_ids.update(seen)
    generated = (out / "sources/code/Generate_dataset.py").read_text(encoding="utf-8")
    utils = (out / "sources/code/utils.py").read_text(encoding="utf-8")
    evidence = {}
    for filename, source in (("Generate_dataset.py", generated), ("utils.py", utils)):
        lines = source.splitlines()
        terms = ["ProviderPublishTime", "ReturnMatrix[company].loc[nextday]", "['Date']", "quantiData_ofCompany[col].mean()", "quantiData_ofCompany[col].std()", "data.y >= 0", "fromtimestamp", "timezone", "datetime.now", ".news"]
        evidence[filename] = [{"line": i+1, "text": line.strip()} for i, line in enumerate(lines) if any(term in line for term in terms)]
    price_contents = (out / "sources/data/SPNews_v2/raw/sp500index.csv").read_text(encoding="utf-8-sig")
    price_reader = csv.DictReader(io.StringIO(price_contents)); price_rows = list(price_reader)
    price_date_column = "Date" if "Date" in price_reader.fieldnames else price_reader.fieldnames[0]
    dates_only = [row[price_date_column] for row in price_rows]
    audit = {
        "source": {"repository": REPOSITORY, "revision": REVISION, "original_publication_doi": "10.1145/3677052.3698644"},
        "sample": {"dates": [row["snapshot_date"] for row in summaries], "files": 8, "downloaded_daily_bytes": sum(entry["size"] for entry in chosen), "convenience_sample_not_global_coverage": True},
        "daily": summaries,
        "cross_snapshot": {"unique_ids": len(id_dates), "ids_in_multiple_snapshot_dates": sum(len(days)>1 for days in id_dates.values()),
                           "ids_associated_with_multiple_company_rows": sum(len(companies)>1 for companies in id_companies.values()),
                           "ids_with_inconsistent_headlines": sum(len(titles)>1 for titles in id_titles.values()),
                           "exact_headlines_in_multiple_snapshot_dates": sum(len(days)>1 for days in headline_dates.values())},
        "source_code_evidence": evidence,
        "verified_method": {"author_daily_alignment": "Generate_dataset.py filters news Date using formatted graph day; ProviderPublishTime is not used in that alignment",
                            "author_target": "Node target Y is supplied ReturnMatrix entry for the next observed matrix date; utils converts nonnegative return to first class",
                            "return_construction": "ReturnMatrix generation and vintage not verified from downloaded sources; do not import supplied targets",
                            "collector_timezone": "Unresolved. No collector code or timezone definition recovered from pinned source tree and requested .py files",
                            "normalization_risk": "Generate_dataset.py mean normalization uses each entire loaded company price column's mean/std, not a declared training partition; do not execute/reuse this preprocessing"},
        "price_metadata_only": {"columns": price_reader.fieldnames, "rows": len(price_rows), "date_column": price_date_column, "first_date_text": dates_only[0] if dates_only else None, "last_date_text": dates_only[-1] if dates_only else None, "price_floats_or_returns_analyzed": False},
        "availability_recommendation": {"publication": "ProviderPublishTime interpreted as UTC Unix seconds; field consistency audited, provider semantics not independently certified",
                                        "dedup": "deduplicate news globally by Id and normalized title, aggregate related tickers without counting the same report several times",
                                        "initial_backlog": "exclude first snapshot's reports from historical news inputs; cannot backfill them to earlier publication dates",
                                        "candidate_rule": "availability=max(publication_utc, first_snapshot_end_utc); keep actual publication age separate from availability lag",
                                        "timezone_guard": "If filename Date is collector local day, UTC day end is not necessarily conservative. Until verified, latest end under UTC-12 is following day12:00UTC; this remains a conditional proxy, not proof of historical collection availability",
                                        "forecast": "use known-available reports before a post-close cutoff and independently rebuild next-session labels from raw index prices"},
        "acquisition_recommendation": {"smallest_next_step": "Fetch a contiguous 60–90 additional daily snapshots, retain only Id/Title/Publisher/Link/ProviderPublishTime/Date/Company/Related Company in a new derived audit cohort; verify collection timing/gaps first",
                                       "full_daily_release": {"files": len(daily), "bytes": sum(entry["size"] for entry in daily)},
                                       "avoid_tickerwise": {"files": len(tickerwise), "bytes": sum(entry["size"] for entry in tickerwise)},
                                       "raw_columns": "Raw full daily CSVs must be downloaded before stripping bodies locally; headlines-only payload not independently released at these paths"},
        "limitations": ["Eight early snapshot dates cannot establish global coverage or prediction quality", "First-snapshot time/timezone unresolved", "Publication dates and collection Date are distinct", "No Reuters instrument sentiment", "Index and stock sources are revised source vintages", "Licensing beyond public repository access not established"],
    }
    _write(out / "audit.json", audit)
    _write(out / "observed_id_first_snapshot.json", first_seen)
    manifest["audit_sha256"] = _hash((out / "audit.json").read_bytes())
    _write(out / "manifest.json", manifest)
    print(json.dumps({"daily_files": len(daily), "daily_MB_decimal": sum(entry["size"] for entry in daily)/1e6,
                      "sample_MB_decimal": audit["sample"]["downloaded_daily_bytes"]/1e6, "cross_snapshot": audit["cross_snapshot"]}, indent=2), flush=True)
    return audit


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)

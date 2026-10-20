"""Immutable public company-day FININ research data; no model training here.

Source times are naive. Dated stories are delayed to D+2 00:00 UTC, then put
in their first eligible exchange close's news bucket. This is an assumption,
not certified historical availability. Raw source returns/summary are never
features. The future target is generated from adjacent exchange closes.

See docs/code_guide.md#data-and-timeline for the code map, contracts, and execution boundaries.
"""
from __future__ import annotations

import bisect
import csv
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from src.runtime_utils import sha256_file as _sha256_file


def sha256_file(path):
    return _sha256_file(Path(path))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_once(path, value):
    """Write new JSON or accept identical content; never replace conflicting evidence."""
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"Immutable artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def content_id(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def availability(source_date, delay_days=2):
    """Return the declared D+2-midnight-UTC proxy, not a recovered publication time."""
    if delay_days < 2:
        raise ValueError("Unverified source dates require the declared >=2-day proxy")
    return datetime.fromisoformat(source_date).replace(tzinfo=timezone.utc) + timedelta(days=delay_days)


def select_tickers(inventory, count=12):
    tickers = [x["path"][:-4] for x in inventory["tree"] if x["path"].endswith(".csv")]
    return sorted(tickers, key=lambda s: content_id("CMIN-student-panel-v1:" + s))[:count]


def acquire(config, source_manifest, directory):
    """Pinned US files only; validate Git blob hashes and never replace originals."""
    directory = Path(directory)
    manifest = read(source_manifest)
    if config["source_commit"] != manifest["commit"]:
        raise ValueError("Source pin mismatch")
    if config["tickers"] != select_tickers(manifest["inventories"]["raw_news"], len(config["tickers"])):
        raise ValueError("Cohort no longer matches frozen outcome-independent selection")
    write_once(directory / "config.json", config)
    jobs = []
    for kind in ("news", "price"):
        lookup = {x["path"]: x for x in manifest["inventories"]["raw_" + kind]["tree"]}
        for ticker in config["tickers"]:
            item = lookup[ticker + ".csv"]
            jobs.append((kind, ticker, item))

    def download(job):
        kind, ticker, item = job
        path = directory / "raw" / kind / (ticker + ".csv")
        url = f"https://raw.githubusercontent.com/BigRoddy/CMIN-Dataset/{config['source_commit']}/CMIN-US/{kind}/raw/{ticker}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            partial = path.with_suffix(".download")
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "FININ-student-research"})
                    with urllib.request.urlopen(req, timeout=60) as src, partial.open("wb") as dst:
                        for block in iter(lambda: src.read(1024 * 1024), b""):
                            dst.write(block)
                    if partial.stat().st_size != item["size"]:
                        raise ValueError("Source size differs from pinned inventory")
                    digest = hashlib.sha1(f"blob {item['size']}\0".encode())
                    with partial.open("rb") as stream:
                        for block in iter(lambda: stream.read(1024 * 1024), b""):
                            digest.update(block)
                    if digest.hexdigest() != item["sha"]:
                        raise ValueError("Pinned Git blob mismatch")
                    partial.rename(path)
                    break
                except Exception as exc:
                    with (directory / f"download_{kind}_{ticker}_failures.jsonl").open("a", encoding="utf-8") as log:
                        log.write(json.dumps({"attempt": attempt + 1, "error": repr(exc), "url": url}) + "\n")
                    if attempt == 2:
                        raise
                    time.sleep(1)
        if path.stat().st_size != item["size"]:
            raise ValueError("Existing pinned source size changed")
        digest = hashlib.sha1(f"blob {item['size']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != item["sha"]:
            raise ValueError("Existing pinned source content changed")
        print(f"verified {kind} {ticker}: {item['size']:,} bytes", flush=True)
        return {"kind": kind, "ticker": ticker, "url": url, "git_blob": item["sha"],
                "bytes": item["size"], "sha256": sha256_file(path)}

    with ThreadPoolExecutor(max_workers=3) as pool:
        files = list(pool.map(download, jobs))
    result = {"source_commit": config["source_commit"], "files": files,
              "bytes": sum(x["bytes"] for x in files), "source_inventory_sha256": sha256_file(source_manifest)}
    write_once(directory / "acquisition.json", result)
    return result


def price_channels(current, previous):
    """Create six finite channels from current OHLCV and adjacent adjusted closes."""
    close = float(current["Close"])
    high, low, opening = (float(current[k]) for k in ("High", "Low", "Open"))
    adj, prev = float(current["Adj Close"]), float(previous["Adj Close"])
    volume = float(current["Volume"])
    if min(close, high, low, opening, adj, prev) <= 0 or volume < 0:
        raise ValueError("Invalid raw prices")
    # Within-session ratios avoid combining split-adjusted levels with raw OHLC.
    value = np.array([adj / prev - 1, opening / close - 1, high / close - 1,
                      low / close - 1, high / low - 1, np.log1p(volume)], dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Nonfinite price features")
    return value


def prepare(config, directory):
    """Verify pinned raw files, align exchange sessions, and save development tensors.

    Price/news histories end at the forecast close; the target is the next
    exchange close. Empty news is retained, and reserved outcomes are excluded.
    See docs/code_guide.md#data-and-timeline for the full temporal contract.
    """
    import exchange_calendars as xcals
    directory = Path(directory)
    if read(directory / "config.json") != config:
        raise ValueError("Frozen source configuration changed")
    if (directory / "prepared.json").exists():
        manifest = read(directory / "prepared.json")
        for name, checksum in manifest["files_sha256"].items():
            if sha256_file(directory / name) != checksum:
                raise ValueError("Prepared artifact changed")
        return manifest
    if (directory / "daily_arrays.npz").exists():
        raise ValueError("Incomplete preparation retained; use a new version instead of overwriting")
    acquisition = read(directory / "acquisition.json")
    for f in acquisition["files"]:
        if sha256_file(directory / "raw" / f["kind"] / (f["ticker"] + ".csv")) != f["sha256"]:
            raise ValueError("Source content changed")
    calendar = xcals.get_calendar("XNYS", start="2018-01-01", end="2021-12-31")
    schedule = calendar.schedule.loc["2018-01-01":"2021-04-30"]
    sessions = [s.date().isoformat() for s in schedule.index]
    closes = [c.to_pydatetime() for c in schedule["close"]]
    write_once(directory / "calendar.json", {"package": xcals.__version__, "exchange": "XNYS",
               "sessions": [{"date": d, "close_utc": c.isoformat()} for d, c in zip(sessions, closes)]})
    n, t, k = len(config["tickers"]), len(sessions), config["headlines_per_arrival_session"]
    prices = np.full((n, t, 6), np.nan, np.float32)
    buckets = [[[] for _ in sessions] for _ in config["tickers"]]
    items, audit, rows = {}, {}, []
    for si, ticker in enumerate(config["tickers"]):
        with (directory / "raw" / "price" / (ticker + ".csv")).open(encoding="utf-8-sig", newline="") as stream:
            raw_prices = list(csv.DictReader(stream))
        bydate = {r["Date"]: r for r in raw_prices}
        if len(bydate) != len(raw_prices):
            raise ValueError("Duplicate price dates")
        for j in range(1, t):
            if sessions[j] in bydate and sessions[j - 1] in bydate:
                prices[si, j] = price_channels(bydate[sessions[j]], bydate[sessions[j - 1]])
        stats, years, months, earliest = Counter(), Counter(), Counter(), {}
        with (directory / "raw" / "news" / (ticker + ".csv")).open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if set(reader.fieldnames or []) != {"date", "time", "ticker", "name", "title", "summary", "link"}:
                raise ValueError("Unexpected raw news schema")
            for record in reader:
                stats["source_rows"] += 1
                text = " ".join((record.get("title") or "").split())
                source_date = record.get("date", "")
                try:
                    available = availability(source_date, config["availability_delay_calendar_days"])
                except (ValueError, TypeError):
                    stats["invalid_date"] += 1
                    continue
                if not text or text.lower() in {"nan", "none"}:
                    stats["empty_title"] += 1
                    continue
                if record.get("ticker") != ticker:
                    raise ValueError("Company association differs from source file")
                if not str(record.get("time", "")).startswith(source_date):
                    stats["date_time_disagreement"] += 1
                stats["replacement_character_titles"] += "\ufffd" in text
                years[source_date[:4]] += 1
                months[source_date[:7]] += 1
                key = content_id(text.casefold())
                story = {"id": content_id(text), "text": text, "source_date": source_date,
                         "source_time_naive": record.get("time"), "source_link": record.get("link"),
                         "available_utc": available.isoformat()}
                if key in earliest:
                    stats["repeated_normalized_titles"] += 1
                if key not in earliest or (source_date, text) < (earliest[key]["source_date"], earliest[key]["text"]):
                    earliest[key] = story
        for story in earliest.values():
            eligible = bisect.bisect_left(closes, datetime.fromisoformat(story["available_utc"]))
            if eligible < t:
                buckets[si][eligible].append(story)
        daily_counts = [len(day) for day in buckets[si]]
        for j, day in enumerate(buckets[si]):
            buckets[si][j] = sorted(day, key=lambda r: r["id"])[:k]
            for story in buckets[si][j]:
                if items.setdefault(story["id"], story["text"]) != story["text"]:
                    raise ValueError("Content hash collision")
        items["description:" + ticker] = "Company ticker " + ticker + "."
        history = config["history_sessions"]
        for j in range(history, t - 1):
            q, d = sessions[j + 1], sessions[j]
            if q >= config["development_target_before"]:
                continue
            if q not in bydate or d not in bydate or not np.isfinite(prices[si, j-history+1:j+1]).all():
                stats["excluded_missing_price_history_or_target"] += 1
                continue
            target_return = float(bydate[q]["Adj Close"]) / float(bydate[d]["Adj Close"]) - 1
            if not np.isfinite(target_return):
                raise ValueError("Invalid target return")
            rows.append({"ticker": ticker, "stock_index": si, "session_index": j,
                         "forecast_date": d, "cutoff_utc": closes[j].isoformat(),
                         "target_date": q, "target_available_utc": closes[j+1].isoformat(),
                         "target": int(target_return > 0), "target_return": target_return})
        audit[ticker] = {"counts": dict(stats), "news_rows_by_year": dict(years), "news_rows_by_month": dict(months),
                         "eligible_bucket_counts_before_cap": daily_counts,
                         "fraction_sessions_with_news": float(np.mean(np.array(daily_counts) > 0)),
                         "fraction_sessions_multiple_news": float(np.mean(np.array(daily_counts) > 1)),
                         "missing_development_price_dates": [d for d in sessions if d not in bydate],
                         "extra_raw_price_dates_in_development": sorted(d for d in bydate if sessions[0] <= d <= sessions[-1] and d not in sessions)}
        print(f"prepared {ticker}: {stats['source_rows']:,} stories, {sum(c>0 for c in daily_counts)}/{t} news sessions", flush=True)
    if len(items) > config["maximum_feature_items"]:
        raise ValueError("Frozen feature-item budget exceeded; preserve audit, do not auto-change cohort")
    listing = sorted(items.items())
    lookup = {key: i for i, (key, _) in enumerate(listing)}
    indices = np.full((n, t, k), -1, np.int32)
    publication = np.zeros((n, t, k), np.float64)
    available_times = np.zeros((n, t, k), np.float64)
    stories = []
    for si, ticker in enumerate(config["tickers"]):
        for j, day in enumerate(buckets[si]):
            for ki, story in enumerate(day):
                indices[si, j, ki] = lookup[story["id"]]
                publication[si, j, ki] = datetime.fromisoformat(story["source_date"]).replace(tzinfo=timezone.utc).timestamp()
                available_times[si, j, ki] = datetime.fromisoformat(story["available_utc"]).timestamp()
                stories.append({"ticker": ticker, "arrival_session": sessions[j], **story})
    rows.sort(key=lambda r: (r["target_date"], r["ticker"]))
    descriptions = np.array([lookup["description:"+s] for s in config["tickers"]], np.int32)
    np.savez(directory / "daily_arrays.npz", prices=prices, news_indices=indices,
             publication_proxy=publication, available_times=available_times, descriptions=descriptions,
             close_seconds=np.array([c.timestamp() for c in closes]))
    write_once(directory / "items.json", listing)
    write_once(directory / "rows.json", rows)
    write_once(directory / "data_audit.json", audit)
    by_identity = {}
    for story in stories:
        by_identity.setdefault(story["id"], []).append((story["ticker"], story["arrival_session"]))
    cross_company = {key: value for key, value in by_identity.items() if len({s for s,d in value}) > 1}
    staggered = {key: value for key, value in cross_company.items() if len({d for s,d in value}) > 1}
    write_once(directory / "duplicate_audit.json", {
        "shared_selected_headline_identities": len(cross_company),
        "shared_identities_with_different_arrival_dates": len(staggered),
        "staggered_examples": list(staggered.items())[:20],
        "scope": "content identity across selected company records; syndicated stories and source vintages unresolved"})
    with (directory / "selected_stories.jsonl").open("x", encoding="utf-8") as stream:
        for story in stories:
            stream.write(json.dumps(story, ensure_ascii=False) + "\n")
    files = ("daily_arrays.npz", "items.json", "rows.json", "data_audit.json", "duplicate_audit.json", "calendar.json", "selected_stories.jsonl")
    result = {"rows": len(rows), "unique_texts": len(items), "tickers": config["tickers"],
              "target_range": [rows[0]["target_date"], rows[-1]["target_date"]],
              "files_sha256": {name: sha256_file(directory/name) for name in files},
              "availability": "unverified date proxy D+2 00UTC; source timestamps not features",
              "label": "next adjacent XNYS session adjusted close > current adjusted close; ties DOWN",
              "preparation_code_sha256": sha256_file(Path(__file__)), "config_sha256": sha256_file(directory/'config.json')}
    write_once(directory / "prepared.json", result)
    return result


def rolling_plans(rows, windows):
    """Build chronological train/valid/evaluation indices with target-boundary purging."""
    plans = []
    for fold, (train_end, valid_end, eval_end) in enumerate(windows):
        train = [i for i, r in enumerate(rows) if r["target_date"] < train_end]
        valid = [i for i, r in enumerate(rows) if train_end <= r["target_date"] < valid_end]
        evaluation = [i for i, r in enumerate(rows) if valid_end <= r["target_date"] < eval_end]
        if not train or not valid or not evaluation:
            raise ValueError("Empty chronological block")
        val_cutoff = min(rows[i]["cutoff_utc"] for i in valid)
        eval_cutoff = min(rows[i]["cutoff_utc"] for i in evaluation)
        purged_train = [i for i in train if rows[i]["target_available_utc"] >= val_cutoff]
        purged_valid = [i for i in valid if rows[i]["target_available_utc"] >= eval_cutoff]
        train = [i for i in train if i not in set(purged_train)]
        valid = [i for i in valid if i not in set(purged_valid)]
        plans.append({"fold": fold, "train": train, "valid": valid, "evaluation": evaluation,
                      "purged_train": purged_train, "purged_valid": purged_valid})
    return plans

"""Timestamp-aware StockNet adaptation; not an exact author-data reproduction.

No training, embeddings or network requests. The principal label is adjusted
close-to-next-adjusted-close direction, without StockNet's movement-band filter.
Fixed 22:00 UTC post-close and 12:00 UTC pre-open cutoffs are deliberately
conservative for the 2014--2015 US trading sessions under study. They are not
exchange-close timestamps or the original StockNet evaluation protocol.
"""

from __future__ import annotations

from bisect import bisect_right
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


SOURCE_REVISION = "330708b5ddc359961078bef469f43f48992fd6e4"
TICKER_DESCRIPTIONS = {
    "AAPL": "Apple Inc., a US consumer electronics and technology company.",
    "AMZN": "Amazon.com Inc., a US online retail and cloud computing company.",
    "BA": "The Boeing Company, a US aerospace and defense company.",
    "BAC": "Bank of America Corporation, a US banking and financial services company.",
    "CVX": "Chevron Corporation, a US energy and petroleum company.",
    "GE": "General Electric Company, a US diversified industrial company in 2014 and 2015.",
    "JNJ": "Johnson & Johnson, a US healthcare and pharmaceutical company.",
    "WMT": "Wal-Mart Stores Inc., a US retail company in 2014 and 2015.",
}
SPLIT_RANGES = {
    "train": ("2014-01-02", "2015-06-30"),
    "valid": ("2015-07-01", "2015-09-30"),
    "test": ("2015-10-01", "2015-12-31"),
}
PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Adj Close", "Volume")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_created_at(value: str) -> datetime:
    """Require an explicit UTC offset and convert the actual timestamp to UTC."""
    if not isinstance(value, str):
        raise ValueError("created_at must be a string")
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y").astimezone(timezone.utc)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timezone-aware created_at: {value!r}") from exc


def _utc(date: str, hour: int) -> datetime:
    return datetime.strptime(f"{date} {hour:02d}", "%Y-%m-%d %H").replace(tzinfo=timezone.utc)


def _iso(time: datetime) -> str:
    return time.isoformat().replace("+00:00", "Z")


def split_for_target(date: str, include_test: bool = False) -> str | None:
    for split, (start, end) in SPLIT_RANGES.items():
        if split == "test" and not include_test:
            continue
        if start <= date <= end:
            return split
    return None


def previous_session(calendar: list[str], date: str) -> str:
    """Find the immediately preceding observed common trading session."""
    position = bisect_right(calendar, date) - 1
    if position < 0 or calendar[position] != date or position == 0:
        raise ValueError("Date is not a common session with an available previous session")
    return calendar[position - 1]


def _read_prices(path: Path, include_test: bool) -> tuple[list[str], dict[str, list[float]]]:
    dates: list[str] = []
    values: dict[str, list[float]] = {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Date", *PRICE_COLUMNS}.issubset(reader.fieldnames):
            raise ValueError(f"Missing price columns: {path}")
        for row in reader:
            date = row["Date"]
            datetime.strptime(date, "%Y-%m-%d")
            dates.append(date)
            # Dev preparation reads future dates as calendar metadata only.
            # Do not even convert any future numeric price cell.
            if not include_test and date >= SPLIT_RANGES["test"][0]:
                continue
            numeric = [float(row[key]) for key in PRICE_COLUMNS]
            if not all(math.isfinite(x) for x in numeric) or any(x <= 0 for x in numeric[:5]) or numeric[5] < 0:
                raise ValueError(f"Invalid price values for {path.name} {date}")
            values[date] = numeric
    if len(dates) != len(set(dates)):
        raise ValueError(f"Duplicate price dates: {path}")
    return sorted(dates), values


def _read_messages(folder: Path, source: Path, include_test: bool) -> tuple[list[dict], dict, list[Path]]:
    messages: list[dict] = []
    mismatches: list[dict] = []
    files = sorted(path for path in folder.iterdir() if path.is_file())
    for path in files:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                timestamp = parse_created_at(row["created_at"])
                # No development statistics or text extraction from future rows.
                if not include_test and timestamp >= _utc(SPLIT_RANGES["test"][0], 0):
                    continue
                tokens = row["text"]
                if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
                    raise ValueError(f"Expected token list: {path}:{number}")
                text = " ".join(" ".join(tokens).split())
                if not text:
                    continue
                relative = path.relative_to(source).as_posix()
                filename_date = path.stem
                if filename_date != timestamp.date().isoformat():
                    mismatches.append({"source_file": relative, "line_number": number,
                                       "created_at": _iso(timestamp), "filename_date": filename_date})
                messages.append({"text": text, "id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                                 "created_at": _iso(timestamp), "source_file": relative,
                                 "line_number": number, "_time": timestamp})
    messages.sort(key=lambda m: (m["_time"], m["source_file"], m["line_number"]))
    return messages, {"filename_timestamp_mismatches": mismatches,
                      "parsed_eligible_nonempty_messages": len(messages)}, files


def _select(messages: list[dict], times: list[datetime], start: datetime, cutoff: datetime, cap: int) -> tuple[list[dict], int, int]:
    eligible = messages[bisect_right(times, start):bisect_right(times, cutoff)]
    # Input ordering gives the earliest timestamp deterministic tie precedence.
    unique: dict[str, dict] = {}
    for row in eligible:
        unique.setdefault(row["text"], row)
    selected = sorted(unique.values(), key=lambda row: (row["id"], row["text"]))[:cap]
    return [{k: v for k, v in row.items() if k != "_time"} for row in selected], len(eligible), len(unique)


def prepare(root: Path, out: Path, include_test: bool = False, *,
            tickers: tuple[str, ...] = tuple(TICKER_DESCRIPTIONS), cap: int = 64) -> tuple[dict[str, list[dict]], dict]:
    """Build new versioned samples and provenance; refuse existing output files.

    ``root`` is the repository or the stocknet-source directory. Returns
    ``({post_close: records, pre_open: records}, manifest)``. The default never
    computes test targets, price floats, test sample counts or test label stats.
    Label-like fields (return/open_up/band_eligible) must never enter features.
    ``include_test=True`` is an explicit later authorization to unlock test data.
    """
    root, out = Path(root), Path(out)
    if not isinstance(include_test, bool):
        raise ValueError("include_test must be explicitly boolean")
    source = root if (root / "price/raw").is_dir() else root / "data/research_v2/stocknet-source"
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Refusing to overwrite existing preparation: {out}")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError("cap must be a positive integer")
    if not tickers or len(set(tickers)) != len(tickers) or any(t not in TICKER_DESCRIPTIONS for t in tickers):
        raise ValueError("Tickers must be unique known companies")
    calendars, prices, messages, hashes, audits = {}, {}, {}, {}, {}
    for ticker in tickers:
        path = source / f"price/raw/{ticker}.csv"
        calendars[ticker], prices[ticker] = _read_prices(path, include_test)
        messages[ticker], audits[ticker], tweet_files = _read_messages(source / f"tweet/preprocessed/{ticker}", source, include_test)
        for file in (path, *tweet_files):
            hashes[file.relative_to(source).as_posix()] = sha256_file(file)
    calendar = sorted(set.intersection(*(set(calendars[ticker]) for ticker in tickers)))
    records: dict[str, list[dict]] = {"post_close": [], "pre_open": []}
    for ticker in tickers:
        ticker_messages = messages[ticker]
        times = [message["_time"] for message in ticker_messages]
        for index in range(1, len(calendar) - 1):
            previous, base, target = calendar[index - 1:index + 2]
            split = split_for_target(target, include_test)
            if split is None:
                continue
            prior_prices, base_prices, target_prices = (prices[ticker][d] for d in (previous, base, target))
            o, h, low, close, adjclose, volume = base_prices
            relative = [o / prior_prices[3] - 1, h / prior_prices[3] - 1,
                        low / prior_prices[3] - 1, close / prior_prices[3] - 1,
                        adjclose / prior_prices[4] - 1, math.log1p(volume)]
            target_return = target_prices[4] / adjclose - 1
            adjusted_target_open = target_prices[0] * target_prices[4] / target_prices[3]
            for schedule in records:
                start, cutoff = ((_utc(previous, 22), _utc(base, 22)) if schedule == "post_close"
                                 else (_utc(base, 22), _utc(target, 12)))
                selected, raw_count, deduplicated_count = _select(ticker_messages, times, start, cutoff, cap)
                records[schedule].append({
                    "id": f"{schedule}_{ticker}_{target}", "ticker": ticker, "split": split,
                    "price_date": base, "forecast_date": cutoff.date().isoformat(),
                    "target_date": target, "target_available_utc": _iso(_utc(target, 22)),
                    "previous_session": previous,
                    "forecast_schedule": schedule, "cutoff_utc": _iso(cutoff),
                    "news_window_start_utc": _iso(start), "market_raw": base_prices,
                    "market_relative": relative, "target_up": int(target_return > 0),
                    "target_return": target_return, "open_up": int(adjusted_target_open > adjclose),
                    "band_eligible": target_return < -.005 or target_return > .0055,
                    "description": TICKER_DESCRIPTIONS[ticker], "headlines": selected,
                    "raw_news_count": raw_count, "deduplicated_news_count": deduplicated_count,
                    "retained_news_count": len(selected),
                })
    for schedule in records:
        records[schedule].sort(key=lambda r: (r["target_date"], r["ticker"]))
    active_splits = ["train", "valid"] + (["test"] if include_test else [])
    manifest = {
        "dataset": "StockNet author-source timestamp-aware adaptation", "source_revision": SOURCE_REVISION,
        "source_directory": str(source.resolve()), "include_test": include_test, "tickers": list(tickers), "cap": cap,
        "split_basis": "target_date", "split_ranges": {s: SPLIT_RANGES[s] for s in active_splits},
        "source_sha256": hashes, "message_audits": audits,
        "common_calendar": calendar, "calendar_basis": "intersection of observed raw-price dates for selected stocks",
        "targets": {"primary": "adjusted close[q] > adjusted close[d]; all movements retained",
                    "secondary_band_flag": "adjusted return < -0.005 or > 0.0055; conditional subset only, no main filtering",
                    "open_up": "adjusted open[q] > adjusted close[d]; label only"},
        "target_availability_guard": "target session 22:00 UTC is a conservative label-availability guard, not exact closing time",
        "selection": "exact whitespace-normalized text dedup, earliest actual timestamp, SHA256 cap after dedup",
        "cutoffs": {"post_close": "(previous_session 22:00 UTC, base_session 22:00 UTC]",
                    "pre_open": "(base_session 22:00 UTC, target_session 12:00 UTC]"},
        "limitations": ["Tweet text is author-preprocessed; no raw headline/news semantic equivalence",
                        "Adjusted prices are source vintages, not established point-in-time adjustments",
                        "Timestamp availability is explicit but collection/selection completeness remains unverified",
                        "Company descriptions are declared static inputs, not author FININ descriptions",
                        "Fixed cutoffs deliberately differ from exact exchange close/open"],
        "summary": {schedule: {split: {"count": sum(r["split"] == split for r in rows),
                                           "up_count": sum(r["target_up"] for r in rows if r["split"] == split),
                                           "empty_news_count": sum(not r["headlines"] for r in rows if r["split"] == split)}
                               for split in active_splits} for schedule, rows in records.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against a concurrent preparation writer.
    for schedule, rows in records.items():
        prefix = "samples" if include_test else "devsamples"
        with (out / f"{prefix}_{schedule}.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    with (out / "provenance.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return records, manifest

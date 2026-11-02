"""Frozen dated-headline packets and local experiment records; labels stay separate."""
from __future__ import annotations

import csv
import io
import json
import math
import random
import re
import sqlite3
import statistics
import uuid
import zipfile
from contextlib import contextmanager
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dateutil.parser import parse as parse_date

from . import RUNTIME_CODE_SHA256, __version__, assert_runtime_current
from .core import _calendar, digest, now
from .gemma import MODEL, SYSTEM

EASTERN = ZoneInfo("America/New_York")
WINDOWS = [5, 10, 20]
SCHEMA = """
CREATE TABLE IF NOT EXISTS studies(
 id TEXT PRIMARY KEY,ticker TEXT NOT NULL,created_at TEXT NOT NULL,spec TEXT NOT NULL,
 status TEXT NOT NULL,error TEXT,pid INTEGER);
CREATE TABLE IF NOT EXISTS predictions(
 id INTEGER PRIMARY KEY,study_id TEXT NOT NULL REFERENCES studies(id),target TEXT NOT NULL,
 window INTEGER NOT NULL,trial INTEGER NOT NULL,packet TEXT NOT NULL,prompt TEXT NOT NULL,
 prompt_hash TEXT NOT NULL,status TEXT NOT NULL,direction TEXT,reason TEXT,raw TEXT,
 model_version TEXT,usage TEXT,started_at TEXT,finished_at TEXT,error TEXT,tokens INTEGER,
 UNIQUE(study_id,target,window,trial));
CREATE TABLE IF NOT EXISTS attempts(
 id INTEGER PRIMARY KEY,prediction_id INTEGER,at TEXT NOT NULL,stage TEXT NOT NULL,detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS requests(
 id INTEGER PRIMARY KEY,at TEXT NOT NULL,input_tokens INTEGER NOT NULL,stage TEXT NOT NULL);
"""


class ExperimentStore:
    def __init__(self, home: Path | str):
        self.home = Path(home).resolve()
        self.root = self.home / "experiments"
        self.root.mkdir(parents=True, exist_ok=True)
        self.db = self.root / "experiments.sqlite3"
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db, timeout=20)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def rows(self, sql, args=()):
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql, args)]

    def execute(self, sql, args=()):
        with self.connect() as db:
            return db.execute(sql, args).lastrowid

    def study(self, study_id):
        rows = self.rows("SELECT * FROM studies WHERE id=?", (study_id,))
        if not rows:
            raise ValueError("Unknown study.")
        return rows[0]

    def attempts(self, prediction_id, stage, detail):
        self.execute("INSERT INTO attempts(prediction_id,at,stage,detail) VALUES(?,?,?,?)",
                     (prediction_id, now(), stage, json.dumps(detail, ensure_ascii=False)))


def session_info(target: str):
    d = date.fromisoformat(target)
    if not 2000 <= d.year <= 2035:
        raise ValueError("Prediction dates must be between 2000 and 2035.")
    cal = _calendar("XNYS")
    if not cal.is_session(pd.Timestamp(d)):
        raise ValueError(f"{target} is not a US trading day. Choose an open market session.")
    return {"open_at": cal.session_open(pd.Timestamp(d)).isoformat(),
            "close_at": cal.session_close(pd.Timestamp(d)).isoformat()}


def trading_dates(since: date, until: date) -> list[str]:
    if until < since or (until - since).days > 1500:
        raise ValueError("Choose an ordered date range of at most 1,500 calendar days.")
    if since.year < 2000 or until.year > 2035:
        raise ValueError("Prediction dates must be between 2000 and 2035.")
    return [str(x.date()) for x in _calendar("XNYS").sessions_in_range(str(since), str(until))]


def normalize_headline(raw_date: str, headline: str, **evidence):
    if not isinstance(raw_date, str) or not isinstance(headline, str):
        raise ValueError("Each row needs a date and headline.")
    raw_date, headline = raw_date.strip(), headline.strip()
    if not raw_date or not headline or len(headline) > 10000:
        raise ValueError("Each row needs a date and a headline of at most 10,000 characters.")
    try:
        # Never invent a missing year/month/day from today's date.
        if not re.search(r"\b\d{4}\b", raw_date):
            raise ValueError()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_date):
            d = date.fromisoformat(raw_date)
        elif re.match(r"^\d{4}-\d{2}-\d{2}T", raw_date):
            a = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            d = (a.astimezone(EASTERN) if a.tzinfo else a).date()
        else:
            a = parse_date(raw_date, default=datetime(1901, 1, 1), dayfirst=True)
            b = parse_date(raw_date, default=datetime(1902, 2, 2), dayfirst=True)
            if a.date() != b.date():
                raise ValueError()
            if a.tzinfo:
                a = a.astimezone(EASTERN)
            d = a.date()
    except (ValueError, OverflowError):
        raise ValueError(f"Unclear date: {raw_date[:60]}. Use YYYY-MM-DD or 10 October 2025.") from None
    pairs = [('"', '"'), ('“', '”'), ("'", "'")]
    if any(headline.startswith(a) and headline.endswith(b) for a, b in pairs):
        headline = headline[1:-1].strip()
    if not headline:
        raise ValueError("A headline cannot be empty.")
    return {"date": d.isoformat(), "headline": headline, **evidence}


def parse_headlines(text: str) -> list[dict]:
    if len(text.encode("utf-8")) > 2_000_000:
        raise ValueError("Import at most 2 MB of dated headlines at a time.")
    records = []
    first = text.lstrip("\ufeff \r\n").splitlines()
    if not first:
        return []
    if "," in first[0] and "headline" in first[0].lower():
        for i, row in enumerate(csv.DictReader(io.StringIO(text.lstrip("\ufeff"))), 2):
            raw = row.get("date") or row.get("published_at") or ""
            try:
                records.append(normalize_headline(raw, row.get("headline") or "", evidence="manual_import"))
            except ValueError as exc:
                raise ValueError(f"CSV row {i}: {exc}") from None
    else:
        for i, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            parts = re.split(r"\s+(?:--|—|–)\s+|\t", line.strip(), maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"Line {i}: use date -- headline, for example 10 October 2025 -- Company announces results.")
            records.append(normalize_headline(*parts, evidence="manual_import"))
    if len(records) > 5000:
        raise ValueError("Import at most 5,000 headlines at a time.")
    return records


def collector_headlines(store, ticker, *, strict=False, scopes=None, job_id=None, since=None, until=None):
    from .query import get_news
    result = get_news(store, ticker, now(), variant="headline", policy="observed" if strict else "historical",
                      scopes=scopes or ["company", "sector"], include_price_recaps=False,
                      job_id=job_id, since=since, until=until)
    records, missing = [], 0
    for r in result["records"]:
        raw = r.get("published_at") or r.get("publication_assertion", {}).get("raw")
        if not raw:
            missing += 1
            continue
        try:
            if r.get("published_at"):
                d = datetime.fromisoformat(r["published_at"].replace("Z", "+00:00")).astimezone(EASTERN).date().isoformat()
            else:
                d = raw[:10]
            records.append(normalize_headline(d, r["headline"], evidence="collector",
                            record_id=r["record_id"], available_at=r["available_at"],
                            timestamp=r.get("published_at"), date_precision=r.get("publication_precision"),
                            source=r.get("source"), source_hash=r.get("raw_hash")))
        except ValueError:
            missing += 1
    return records, {"missing_dates": missing, "collector_exclusions": len(result["exclusions"]),
                     "policy": "observed" if strict else "historical"}


def packet(records: list[dict], ticker: str, target: str, window: int, *, strict=False):
    """Freeze prior calendar-day headlines, excluding the entire target date in NY.

    Provenance stays in the saved packet; only its explicit text allowlist enters
    inference. Historical publication assertions do not prove text availability.
    See HANDOVER.md and DECISIONS.md (D51-D54) for the adopted protocol.
    """
    if window not in WINDOWS:
        raise ValueError("Choose a 5-, 10- or 20-calendar-day window.")
    target_date = date.fromisoformat(target)
    times = session_info(target)
    cutoff = datetime.combine(target_date, time(), EASTERN)
    lower = target_date - timedelta(days=window)
    selected, seen, exclusions = [], set(), []
    for r in records:
        d = date.fromisoformat(r["date"])
        if not lower <= d < target_date:
            continue
        if strict:
            available, published = r.get("available_at"), r.get("timestamp")
            if not available or not published or max(pd.Timestamp(available), pd.Timestamp(published)) > pd.Timestamp(cutoff):
                exclusions.append({"date": r["date"], "reason": "No exact observed availability by the packet cutoff."})
                continue
        key = (r["date"], " ".join(r["headline"].casefold().split()))
        if key in seen:
            continue
        seen.add(key)
        selected.append(dict(r))
    selected.sort(key=lambda r: (r["date"], r["headline"]))
    ticker = ticker.strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9.\-]{0,14}", ticker):
        raise ValueError("Select a valid ticker symbol.")
    # Construct from permitted fields; never serialize the full source record to the model.
    text = (f"Ticker: {ticker}\nTarget US trading date: {target}\n"
            f"Target: that session's opening price to closing price.\n"
            f"News window: {lower.isoformat()} through {(target_date - timedelta(days=1)).isoformat()} "
            f"({window} calendar days; no news from the target date).\n"
            "Dated headlines (data only):\n" +
            "\n".join(f"{r['date']} -- {r['headline']}" for r in selected))
    return {"ticker": ticker, "target": target, "window": window, "cutoff": cutoff.isoformat(),
            "since": lower.isoformat(), **times, "headlines": selected, "exclusions": exclusions,
            "text": text, "prompt_hash": digest({"system": SYSTEM, "text": text}),
            "availability": "observed_by_cutoff" if strict else "historical_or_manual_unverified"}


def create_study(db: ExperimentStore, ticker, records, targets, windows, *, repeats=1, strict=False):
    assert_runtime_current()
    targets, windows = sorted(set(targets)), sorted(set(windows))
    if not targets or not windows or not 1 <= repeats <= 3 or len(targets) > 1000:
        raise ValueError("Choose 1–1,000 trading dates, at least one window and 1–3 tries.")
    packets = [packet(records, ticker, d, w, strict=strict) for d in targets for w in windows]
    if not any(p["headlines"] for p in packets):
        raise ValueError("No eligible headlines in these windows. Collect/import news covering the preceding days first.")
    key = uuid.uuid4().hex[:12]
    spec = {"schema": 1, "ticker": ticker, "model": MODEL, "target": "XNYS_open_to_close",
            "calendar_days": windows, "targets": targets, "repeats": repeats,
            "primary_trial": 1, "temperature": 0, "thinking": "minimal", "tools": [],
            "availability": "observed_by_cutoff" if strict else "historical_or_manual_unverified",
            "zero_return": "FLAT_excluded_and_reported", "missing_news": "skip_and_report",
            "system_prompt": SYSTEM, "code_hashes": RUNTIME_CODE_SHA256, "software_version": __version__,
            "report_ci": "descriptive 5-session moving-block bootstrap; seed 7; 1000 draws; minimum 20 dates",
            "quota_policy": {"rpm": 28, "input_tpm": 14000, "rpd": 13000}}
    spec["config_hash"] = digest(spec)
    with db.connect() as conn:
        conn.execute("INSERT INTO studies VALUES(?,?,?,?,?,?,?)", (key, ticker, now(), json.dumps(spec), "ready", None, None))
        for p in packets:
            p["run_mode"] = "prospective" if pd.Timestamp(now()) < pd.Timestamp(p["open_at"]) else "retrospective"
            for trial in range(1, repeats + 1):
                conn.execute("""INSERT INTO predictions(study_id,target,window,trial,packet,prompt,prompt_hash,status,error)
                    VALUES(?,?,?,?,?,?,?,?,?)""", (key, p["target"], p["window"], trial, json.dumps(p, ensure_ascii=False),
                    p["text"], p["prompt_hash"], "pending" if p["headlines"] else "skipped",
                    None if p["headlines"] else "No headlines in this window."))
    return key


def labels_from_csv(text: str, ticker: str) -> dict:
    rows = csv.DictReader(io.StringIO(text.lstrip("\ufeff")))
    if not {"date", "open", "close"}.issubset(rows.fieldnames or []):
        raise ValueError("Price CSV needs date,open,close columns; ticker and source are optional.")
    result = {}
    for i, row in enumerate(rows, 2):
        if row.get("ticker", ticker).strip().upper() != ticker:
            raise ValueError(f"Price row {i} has a different ticker.")
        d = row["date"].strip()
        session_info(d)
        if pd.Timestamp(now()) < pd.Timestamp(session_info(d)["close_at"]):
            raise ValueError(f"{d} has not closed yet; its outcome cannot be scored.")
        try:
            opening, closing = float(row["open"]), float(row["close"])
        except (TypeError, ValueError):
            raise ValueError(f"Price row {i} needs numeric open/close values.") from None
        if not all(math.isfinite(v) and v > 0 for v in (opening, closing)) or d in result:
            raise ValueError(f"Price row {i} has duplicate dates or invalid prices.")
        result[d] = {"open": opening, "close": closing, "source": row.get("source", "user CSV"),
                     "direction": "UP" if closing > opening else "DOWN" if closing < opening else "FLAT"}
    if not result:
        raise ValueError("Price CSV has no rows.")
    return result


def save_labels(db: ExperimentStore, study_id: str, text: str):
    study = db.study(study_id)
    # Enforce separation: workers never open this directory. Labels do not alter frozen packets.
    labels = labels_from_csv(text, study["ticker"])
    targets = set(json.loads(study["spec"])["targets"])
    labels = {d: row for d, row in labels.items() if d in targets}
    if not labels:
        raise ValueError("The price file contains none of this experiment's prediction dates.")
    folder = db.home / "scoring"
    folder.mkdir(exist_ok=True)
    versions = folder / study_id
    versions.mkdir(exist_ok=True)
    key = digest(text)
    path = versions / (key + ".json")
    data = {"imported_at": now(), "csv_sha256": key, "labels": labels}
    if not path.exists():
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    (versions / "selected.json").write_text(json.dumps({"version": key}), encoding="utf-8")
    return len(labels)


def load_labels(db: ExperimentStore, study_id: str):
    folder = db.home / "scoring" / study_id
    if not (folder / "selected.json").exists():
        return {}
    version = json.loads((folder / "selected.json").read_text())["version"]
    if not re.fullmatch(r"[0-9a-f]{64}", version):
        raise ValueError("Invalid label version.")
    return json.loads((folder / (version + ".json")).read_text())


def block_interval(values):
    if len(values) < 20:
        return None
    rng, n, estimates = random.Random(7), len(values), []
    for _ in range(1000):
        sample = []
        while len(sample) < n:
            start = rng.randrange(n)
            sample.extend(values[(start + i) % n] for i in range(5))
        estimates.append(statistics.mean(sample[:n]))
    estimates.sort()
    return [estimates[24], estimates[974]]


def report(db: ExperimentStore, study_id: str):
    study = db.study(study_id)
    spec = json.loads(study["spec"])
    predictions = db.rows("SELECT * FROM predictions WHERE study_id=? ORDER BY target,window,trial", (study_id,))
    label_record = load_labels(db, study_id)
    labels = label_record.get("labels", {})
    rows, scored = [], {}
    for p in predictions:
        label = labels.get(p["target"], {}).get("direction")
        correct = p["direction"] == label if p["status"] == "completed" and label in ("UP", "DOWN") else None
        rows.append({"date": p["target"], "window_days": p["window"], "try": p["trial"], "status": p["status"],
                     "prediction": p["direction"], "actual": label, "correct": correct, "reason": p["reason"],
                     "error": p["error"], "headlines": len(json.loads(p["packet"])["headlines"])})
        if p["trial"] == 1 and correct is not None:
            scored[(p["window"], p["target"])] = int(correct)
    eligible_sets = [{d for w, d in scored if w == window} for window in spec["calendar_days"]]
    common = sorted(set.intersection(*eligible_sets)) if eligible_sets else []
    summary = []
    for window in spec["calendar_days"]:
        items = [r for r in rows if r["window_days"] == window and r["try"] == 1]
        valid = [r for r in items if r["correct"] is not None]
        shared = [scored[(window, d)] for d in common]
        recalls = [statistics.mean(int(r["prediction"] == lab) for r in valid if r["actual"] == lab)
                   for lab in ("UP", "DOWN") if any(r["actual"] == lab for r in valid)]
        summary.append({"window_days": window, "planned_dates": len(items), "scored_dates": len(valid),
                        "correct": sum(bool(r["correct"]) for r in valid),
                        "accuracy": statistics.mean(r["correct"] for r in valid) if valid else None,
                        "balanced_accuracy": statistics.mean(recalls) if len(recalls) == 2 else None,
                        "always_up_accuracy": statistics.mean(r["actual"] == "UP" for r in valid) if valid else None,
                        "shared_dates": len(shared), "shared_accuracy": statistics.mean(shared) if shared else None,
                        "shared_interval": block_interval(shared),
                        "failed_or_unscored": len(items) - len(valid)})
    differences = []
    for i, a in enumerate(spec["calendar_days"]):
        for b in spec["calendar_days"][i + 1:]:
            diffs = [scored[(a, d)] - scored[(b, d)] for d in common]
            differences.append({"a_days": a, "b_days": b, "shared_dates": len(common),
                                "accuracy_difference": statistics.mean(diffs) if diffs else None,
                                "interval": block_interval(diffs)})
    agreements = []
    for window in spec["calendar_days"]:
        for d in spec["targets"]:
            trials = [r for r in rows if r["window_days"] == window and r["date"] == d]
            if len(trials) > 1 and all(r["status"] == "completed" for r in trials):
                agreements.append(len({r["prediction"] for r in trials}) == 1)
    return {"study_id": study_id, "ticker": study["ticker"], "summary": summary, "rows": rows,
            "paired_differences": differences, "label_version": label_record.get("csv_sha256"),
            "label_balance": {lab: sum(v["direction"] == lab for d, v in labels.items() if d in spec["targets"])
                              for lab in ("UP", "DOWN", "FLAT")},
            "repeat_agreement": statistics.mean(agreements) if agreements else None,
            "primary_trial": 1, "availability": spec["availability"],
            "interval_note": spec["report_ci"]}


def bundle(db: ExperimentStore, study_id: str) -> bytes:
    study = db.study(study_id)
    predictions = db.rows("SELECT * FROM predictions WHERE study_id=? ORDER BY id", (study_id,))
    result = report(db, study_id)
    output = io.BytesIO()
    csv_out = io.StringIO(newline="")
    fields = list(result["rows"][0]) if result["rows"] else []
    writer = csv.DictWriter(csv_out, fieldnames=fields)
    writer.writeheader()
    for row in result["rows"]:
        writer.writerow({k: "'" + v if isinstance(v, str) and v[:1] in "=+-@" else v for k, v in row.items()})
    files = {"results.csv": csv_out.getvalue(), "report.json": json.dumps(result, indent=2),
             "protocol.json": study["spec"],
             "predictions.jsonl": "\n".join(json.dumps(r, ensure_ascii=False) for r in predictions),
             "labels.json": json.dumps(load_labels(db, study_id)),
             "attempts.jsonl": "\n".join(json.dumps(r) for r in db.rows(
                 "SELECT a.* FROM attempts a JOIN predictions p ON p.id=a.prediction_id WHERE p.study_id=? ORDER BY a.id", (study_id,))),
             "README.txt": "Target: BABA/selected ticker US regular-session open to close. News: dated headlines only.\n"
                 "Primary accuracy uses try 1 per date/window. Repeats are not independent market outcomes.\n"
                 "Historical/manual news versions may be unverified. No claim of training decontamination.\n"
                 "No Search/tools are configured. Hosted provider internals and weights are not audited.\n"
                 "Intervals, where available, are descriptive 5-session block-bootstrap intervals, not proof of predictive power.\n"}
    manifest = {"schema": 1, "created_at": now(), "study_id": study_id,
                "files_sha256": {name: digest(content.encode("utf-8")) for name, content in files.items()}}
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
        z.writestr("manifest.json", json.dumps(manifest, indent=2))
    return output.getvalue()

"""Durable observations and immutable extraction versions; local SQLite only."""
from __future__ import annotations

import gzip
import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .core import Settings, digest, now

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info(version INTEGER NOT NULL);
INSERT INTO schema_info SELECT 1 WHERE NOT EXISTS(SELECT 1 FROM schema_info);
CREATE TABLE IF NOT EXISTS jobs(
 id TEXT PRIMARY KEY, request TEXT NOT NULL, status TEXT NOT NULL, created_at TEXT NOT NULL,
 started_at TEXT, finished_at TEXT, heartbeat TEXT, owner_pid INTEGER, cancel INTEGER DEFAULT 0,
 next_run TEXT, summary TEXT DEFAULT '{}', error TEXT);
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY, job_id TEXT, at TEXT NOT NULL, level TEXT NOT NULL,
 stage TEXT NOT NULL, message TEXT NOT NULL, detail TEXT DEFAULT '{}');
CREATE TABLE IF NOT EXISTS source_runs(
 id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, ticker TEXT NOT NULL, source TEXT NOT NULL,
 started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL, coverage TEXT NOT NULL,
 discovered INTEGER DEFAULT 0, detail TEXT, artifact TEXT);
CREATE TABLE IF NOT EXISTS discoveries(
 id INTEGER PRIMARY KEY, job_id TEXT NOT NULL, source_run INTEGER, ticker TEXT NOT NULL,
 url TEXT NOT NULL, candidate TEXT NOT NULL, first_seen TEXT NOT NULL,
 state TEXT DEFAULT 'pending', reason TEXT, headline_ok INTEGER NOT NULL,
 headline_reason TEXT NOT NULL, assessed_at TEXT NOT NULL,
 UNIQUE(job_id,ticker,url,source_run));
CREATE TABLE IF NOT EXISTS documents(id TEXT PRIMARY KEY, url TEXT UNIQUE NOT NULL);
CREATE TABLE IF NOT EXISTS versions(
 id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id),
 ticker TEXT NOT NULL, source TEXT NOT NULL, publisher TEXT NOT NULL, url TEXT NOT NULL,
 title TEXT NOT NULL, body TEXT NOT NULL, paragraphs TEXT NOT NULL,
 raw_hash TEXT NOT NULL, text_hash TEXT NOT NULL, extracted_at TEXT NOT NULL,
 fetched_at TEXT NOT NULL, first_seen TEXT NOT NULL, published TEXT NOT NULL,
 provider_time TEXT NOT NULL, extractor TEXT NOT NULL, kind TEXT NOT NULL,
 accepted INTEGER NOT NULL, reason TEXT NOT NULL, metadata TEXT NOT NULL,
 UNIQUE(document_id,ticker,raw_hash,extractor));
CREATE TABLE IF NOT EXISTS annotations(
 id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, record_id TEXT NOT NULL,
 reviewer TEXT NOT NULL, label TEXT NOT NULL, note TEXT, at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS fetch_log(
 id INTEGER PRIMARY KEY, job_id TEXT, url TEXT NOT NULL, at TEXT NOT NULL,
 status INTEGER, artifact TEXT, detail TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS worker_lease(
 name TEXT PRIMARY KEY, pid INTEGER NOT NULL, token TEXT NOT NULL, heartbeat TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS versions_time ON versions(ticker,extracted_at);
CREATE INDEX IF NOT EXISTS discoveries_time ON discoveries(ticker,first_seen);
CREATE INDEX IF NOT EXISTS events_job ON events(job_id,id);
CREATE TABLE IF NOT EXISTS audits(
 id TEXT PRIMARY KEY, created_at TEXT NOT NULL, spec TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_reviews(
 audit_id TEXT NOT NULL REFERENCES audits(id), record_id TEXT NOT NULL, reviewer TEXT NOT NULL,
 relevance TEXT NOT NULL, text_quality TEXT NOT NULL, timing TEXT NOT NULL, note TEXT NOT NULL,
 at TEXT NOT NULL, PRIMARY KEY(audit_id,record_id,reviewer));
CREATE TABLE IF NOT EXISTS content_quarantines(
 id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, raw_hash TEXT NOT NULL, at TEXT NOT NULL,
 reason TEXT NOT NULL, UNIQUE(ticker,raw_hash,reason));
CREATE INDEX IF NOT EXISTS quarantines_hash ON content_quarantines(ticker,raw_hash,at);
CREATE TABLE IF NOT EXISTS headline_quarantines(
 id INTEGER PRIMARY KEY, ticker TEXT NOT NULL, discovery_id INTEGER NOT NULL REFERENCES discoveries(id),
 at TEXT NOT NULL, reason TEXT NOT NULL, UNIQUE(ticker,discovery_id,reason));
CREATE INDEX IF NOT EXISTS headline_quarantines_time ON headline_quarantines(ticker,at);
"""


class Store:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.artifacts = settings.root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        with self.connect() as db:
            # Older Windows Python builds bundle SQLite affected by the WAL race.
            v = sqlite3.sqlite_version_info
            safe_wal = v >= (3, 51, 3) or (3, 50, 7) <= v < (3, 51, 0) or (3, 44, 6) <= v < (3, 45, 0)
            db.execute("PRAGMA journal_mode=" + ("WAL" if safe_wal else "DELETE"))
            db.executescript(SCHEMA)
            if db.execute("SELECT version FROM schema_info").fetchone()[0] != 1:
                raise RuntimeError("Unsupported database schema; preserve data and use the matching collector release.")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.settings.db, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA busy_timeout=20000")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def rows(self, sql: str, args=()) -> list[dict]:
        with self.connect() as db:
            return [dict(r) for r in db.execute(sql, args)]

    def execute(self, sql: str, args=()) -> int:
        with self.connect() as db:
            return db.execute(sql, args).lastrowid

    def event(self, job: str | None, stage: str, message: str, level: str = "info", **detail):
        self.execute("INSERT INTO events(job_id,at,level,stage,message,detail) VALUES(?,?,?,?,?,?)",
                     (job, now(), level, stage, message, json.dumps(detail)))

    def archive(self, body: bytes) -> str:
        if shutil.disk_usage(self.settings.root).free < self.settings.min_free_bytes:
            raise OSError("Insufficient disk space. Collection stopped; existing evidence was preserved.")
        key = digest(body)
        path = self.artifacts / key[:2] / (key + ".gz")
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
            try:
                tmp.write_bytes(gzip.compress(body, mtime=0))
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        return key

    def read_artifact(self, key: str) -> bytes:
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("Invalid artifact identifier.")
        body = gzip.decompress((self.artifacts / key[:2] / (key + ".gz")).read_bytes())
        if digest(body) != key:
            raise ValueError("Artifact checksum mismatch: preserve this evidence and restore a verified backup.")
        return body

    def create_job(self, request: dict) -> str:
        key = uuid.uuid4().hex[:12]
        self.execute("INSERT INTO jobs(id,request,status,created_at) VALUES(?,?,?,?)",
                     (key, json.dumps(request), "queued", now()))
        self.event(key, "queue", "Collection queued.", tickers=request["tickers"])
        return key

    def save_version(self, data: dict) -> str:
        document_id = digest(data["url"])[:24]
        data["document_id"] = document_id
        data["id"] = digest({"document": document_id, "ticker": data["ticker"], "raw": data["raw_hash"], "extractor": data["extractor"]})
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO documents VALUES(?,?)", (document_id, data["url"]))
            columns = list(data)
            values = [json.dumps(data[c], ensure_ascii=False) if isinstance(data[c], (dict, list)) else data[c] for c in columns]
            db.execute(f"INSERT OR IGNORE INTO versions({','.join(columns)}) VALUES({','.join('?' for _ in columns)})", values)
        return data["id"]

    def job(self, job_id: str) -> dict:
        rows = self.rows("SELECT * FROM jobs WHERE id=?", (job_id,))
        if not rows:
            raise ValueError(f"Unknown job: {job_id}")
        return rows[0]

    def checkpoint(self, job_id: str):
        self.execute("UPDATE jobs SET heartbeat=? WHERE id=?", (now(), job_id))
        if self.job(job_id)["cancel"]:
            raise InterruptedError("Stopped by user. Pending items can be resumed.")

    def backup(self, destination: Path):
        destination = destination.resolve()
        if destination == self.settings.root or self.settings.root in destination.parents:
            raise ValueError("Choose a backup folder outside the active data directory.")
        destination.mkdir(parents=True, exist_ok=False)
        with self.connect() as db, sqlite3.connect(destination / "collector.sqlite3") as backup:
            db.backup(backup)
        shutil.copytree(self.artifacts, destination / "artifacts")
        for name in ("companies.json", "sources.json"):
            custom = self.settings.root / name
            if custom.exists():
                shutil.copy2(custom, destination / custom.name)

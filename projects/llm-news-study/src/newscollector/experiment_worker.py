"""One sequential prediction worker with durable checkpoints and conservative quotas."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import psutil

from . import RUNTIME_CODE_SHA256, assert_runtime_current
from .core import now
from .experiments import ExperimentStore
from .gemma import GemmaClient, ProviderError, credential, parse_prediction


def alive(pid):
    if not pid:
        return False
    try:
        proc = psutil.Process(pid)
        return proc.is_running() and "newscollector.experiment_worker" in " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False


def start(db: ExperimentStore, study_id: str):
    assert_runtime_current()
    study = db.study(study_id)
    if json.loads(study["spec"])["code_hashes"] != RUNTIME_CODE_SHA256:
        raise ValueError("Software changed since this study was frozen. Create a new study with the current code.")
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for active in conn.execute("SELECT * FROM studies WHERE pid IS NOT NULL OR status IN ('running','starting')"):
            if alive(active["pid"]):
                raise ValueError("A prediction study is already running. Pause or stop it first.")
            conn.execute("UPDATE studies SET status='paused',pid=NULL,error=? WHERE id=?",
                         ("Previous worker ended. Resume to retry unfinished requests; an interrupted provider call may have been processed.", active["id"]))
        if study["status"] == "completed":
            raise ValueError("This study is complete. Create a new study for a separate run.")
        conn.execute("UPDATE predictions SET status='pending' WHERE study_id=? AND status IN ('running','failed')", (study_id,))
        conn.execute("UPDATE studies SET status='starting',error=NULL WHERE id=?", (study_id,))
        try:
            kwargs = {"cwd": os.getcwd(), "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
            process = subprocess.Popen([sys.executable, "-m", "newscollector.experiment_worker", "--home", str(db.home), "--study", study_id], **kwargs)
            conn.execute("UPDATE studies SET pid=? WHERE id=?", (process.pid, study_id))
        except OSError:
            conn.execute("UPDATE studies SET status='paused',error=? WHERE id=?", ("Could not start the prediction worker. Try Resume.", study_id))
            raise ValueError("Could not start the prediction worker.") from None


def pause(db, study_id, *, stop=False):
    db.execute("UPDATE studies SET status=? WHERE id=? AND status IN ('running','starting','paused','ready')",
               ("stopped" if stop else "paused", study_id))


def wait_for_budget(db, study_id, stage, tokens, *, sleep=time.sleep):
    if tokens > 14000:
        raise ProviderError("This packet exceeds the 14,000-token safety budget. No headlines were silently removed. Create a smaller packet.")
    while True:
        if db.study(study_id)["status"] != "running":
            return False
        moment = datetime.now(UTC)
        since = (moment - timedelta(seconds=62)).isoformat()
        recent = db.rows("SELECT * FROM requests WHERE at>=? ORDER BY at", (since,))
        pacific = moment.astimezone(ZoneInfo("America/Los_Angeles"))
        daily_start = pacific.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(UTC).isoformat()
        count = db.rows("SELECT COUNT(*) n FROM requests WHERE at>=?", (daily_start,))[0]["n"]
        if count >= 13000:
            raise ProviderError("Daily request safety allowance reached. Resume after midnight Pacific time.", 429)
        if len(recent) < 28 and sum(r["input_tokens"] for r in recent) + tokens <= 14000:
            # Count metadata/countTokens calls too; input counts conservatively reserve both counting and generation.
            db.execute("INSERT INTO requests(at,input_tokens,stage) VALUES(?,?,?)", (now(), tokens, stage))
            return True
        db.execute("UPDATE studies SET error=? WHERE id=?", ("Waiting for the per-minute request/token allowance. Progress is saved.", study_id))
        sleep(1)


def work(db: ExperimentStore, study_id: str, *, client_factory=GemmaClient, sleep=time.sleep):
    study = db.study(study_id)
    if study["status"] == "completed":
        return
    if study["status"] in ("paused", "stopped"):
        # Honor a stop/pause clicked while the child process was still starting.
        db.execute("UPDATE studies SET pid=NULL WHERE id=?", (study_id,))
        return
    spec = json.loads(study["spec"])
    if spec["code_hashes"] != RUNTIME_CODE_SHA256:
        db.execute("UPDATE studies SET status='paused',error=? WHERE id=?", ("Software changed. Create a new study.", study_id))
        return
    db.execute("UPDATE studies SET status='running',pid=?,error=NULL WHERE id=?", (os.getpid(), study_id))
    client = None
    try:
        client = client_factory(credential(db.home))
        if not wait_for_budget(db, study_id, "model_check", 0, sleep=sleep):
            return
        metadata = client.check()
        previous_meta = spec.get("provider_model_record")
        if previous_meta and previous_meta != metadata:
            raise ProviderError("Google's model metadata changed since the first request. Create a separate study.")
        if not previous_meta:
            spec["provider_model_record"] = metadata
            db.execute("UPDATE studies SET spec=? WHERE id=?", (json.dumps(spec), study_id))
        for p in db.rows("SELECT * FROM predictions WHERE study_id=? AND status='pending' ORDER BY target,window,trial", (study_id,)):
            assert_runtime_current()
            if db.study(study_id)["status"] != "running":
                return
            frozen = json.loads(p["packet"])
            if frozen["run_mode"] == "prospective" and pd.Timestamp(now()) >= pd.Timestamp(frozen["open_at"]):
                db.execute("UPDATE predictions SET status='skipped',error=? WHERE id=?", ("Prediction was not completed before the session opened.", p["id"]))
                continue
            db.execute("UPDATE predictions SET status='running',started_at=?,error=NULL WHERE id=?", (now(), p["id"]))
            db.attempts(p["id"], "start", {"prompt_hash": p["prompt_hash"], "model": spec["model"], "tools": []})
            raw = None
            try:
                tokens = p["tokens"]
                if tokens is None:
                    # UTF-8 bytes are a conservative reservation for the countTokens request, capped to the limit.
                    estimate = min(14000, len((spec["system_prompt"] + p["prompt"]).encode("utf-8")) + 128)
                    if not wait_for_budget(db, study_id, "count_tokens", estimate, sleep=sleep):
                        db.execute("UPDATE predictions SET status='pending' WHERE id=?", (p["id"],))
                        return
                    tokens = client.count(p["prompt"])
                    db.execute("UPDATE predictions SET tokens=? WHERE id=?", (tokens, p["id"]))
                    db.attempts(p["id"], "token_count", {"tokens": tokens})
                if not wait_for_budget(db, study_id, "generate", tokens, sleep=sleep):
                    db.execute("UPDATE predictions SET status='pending' WHERE id=?", (p["id"],))
                    return
                db.execute("UPDATE studies SET error=NULL WHERE id=?", (study_id,))
                raw = client.predict(p["prompt"])
                # Save raw response before parsing: malformed answers remain auditable.
                db.execute("UPDATE predictions SET raw=?,model_version=?,usage=?,finished_at=? WHERE id=?",
                           (raw["raw_text"], raw.get("model_version"), json.dumps(raw.get("usage", {})), now(), p["id"]))
                db.attempts(p["id"], "response", raw)
                result = parse_prediction(raw["raw_text"])
                if frozen["run_mode"] == "prospective" and pd.Timestamp(now()) >= pd.Timestamp(frozen["open_at"]):
                    raise ProviderError("Response arrived after the target session opened; excluded as late.")
                db.execute("UPDATE predictions SET status='completed',direction=?,reason=? WHERE id=?",
                           (result["direction"], result["reason"], p["id"]))
            except (ProviderError, ValueError) as exc:
                db.attempts(p["id"], "error", {"message": str(exc), "http_status": getattr(exc, "status", None)})
                db.execute("UPDATE predictions SET status='failed',error=?,finished_at=? WHERE id=?", (str(exc), now(), p["id"]))
                db.execute("UPDATE studies SET status='paused',error=? WHERE id=?", (str(exc), study_id))
                return
        if db.study(study_id)["status"] == "running":
            db.execute("UPDATE studies SET status='completed',error=NULL WHERE id=?", (study_id,))
    except (ProviderError, ValueError) as exc:
        db.execute("UPDATE studies SET status='paused',error=? WHERE id=?", (str(exc), study_id))
    except Exception:
        # Provider exceptions can contain request headers. Never serialize them.
        db.execute("UPDATE studies SET status='paused',error=? WHERE id=?", ("Worker interrupted by an unexpected error. Resume unfinished requests; saved outputs remain available.", study_id))
    finally:
        if client:
            client.close()
        db.execute("UPDATE studies SET pid=NULL WHERE id=?", (study_id,))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", required=True)
    parser.add_argument("--study", required=True)
    args = parser.parse_args()
    work(ExperimentStore(args.home), args.study)

"""Explicit stages for the bounded no-neural-training temporal study."""
import argparse
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.temporal_pooling_study import execute
from src.encoder_comparison import write


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "train", "analyze"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/temporal_pooling_study_2026-10-13")
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    token = time.time_ns()
    command_path = args.output / f"command_{args.stage}_{token}.json"
    command = {"stage": args.stage, "raw_argv": sys.argv.copy(), "cwd": str(Path.cwd()),
               "python_executable": sys.executable, "output": str(args.output),
               "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running",
               "offline_environment": {name: os.environ[name] for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}}
    write(command_path, command)
    started = time.monotonic()
    try:
        result = execute(ROOT, args.output, args.stage)
        command["status"] = "completed"
        print(f"Completed temporal study stage {args.stage}", flush=True)
    except BaseException:
        command["status"] = "failed"
        command["traceback"] = traceback.format_exc()
        args.output.mkdir(parents=True, exist_ok=True)
        write(args.output / f"failure_{args.stage}_{time.time_ns()}.json", {"traceback": traceback.format_exc()})
        raise
    finally:
        command["finished_utc"] = datetime.now(timezone.utc).isoformat()
        command["seconds"] = time.monotonic() - started
        write(command_path, command)

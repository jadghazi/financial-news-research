"""Fresh CMIN study stages; imports do not download, fit or rewrite artifacts.

See docs/code_guide.md#pipeline-entry-points for the code map, contracts, and execution boundaries.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cmin_panel_data import acquire, prepare, read, write_once


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["acquire", "prepare", "cache", "run", "inspect"])
    parser.add_argument("--config", default="configs/cmin_panel_v1.json")
    parser.add_argument("--data-directory", help="Audited existing cohort; run stage only")
    parser.add_argument("--study-directory", help="Distinct output for an explicitly configured later study")
    parser.add_argument("--protocol", default="docs/cmin_panel_protocol_2026-10-14.md")
    args = parser.parse_args()
    config = read(args.config)
    directory = Path("artifacts") / config["id"]
    if args.data_directory or args.study_directory:
        if args.stage != "run" or not args.data_directory or not args.study_directory:
            parser.error("Later studies require run, --data-directory and --study-directory together")
        directory = Path(args.study_directory)
    directory.mkdir(parents=True,exist_ok=True)
    command = directory/f"command_{args.stage}_{time.time_ns()}"
    write_once(command.with_suffix(".started.json"),{"argv":sys.argv,"utc":datetime.now(timezone.utc).isoformat()})
    try:
        if args.stage == "acquire":
            result = acquire(config, "artifacts/feasibility/public_reference_2026-10-14/cmin_source_manifest.json", directory)
        elif args.stage == "prepare":
            result = prepare(config, directory)
        elif args.stage == "cache":
            from src.cmin_panel_experiment import build_cache
            result = build_cache(config, directory)
        elif args.stage == "run":
            from src.cmin_panel_experiment import run
            result = run(config, Path(args.data_directory) if args.data_directory else directory,
                         study_out=args.study_directory,protocol_path=args.protocol)
        else:
            result = read(directory / "summary.json")
        write_once(command.with_suffix(".finished.json"),{"status":"completed","utc":datetime.now(timezone.utc).isoformat()})
        display={"stage":args.stage,"status":"complete",**{k:result[k] for k in
              ("rows","unique_texts","items","completed_neural_fits","evaluation_company_days","bytes") if k in result}}
        if "arms" in result:
            display["mean_seed_log_loss"]={arm:value["mean_seed_log_loss"] for arm,value in result["arms"].items()}
            display["paired_comparisons"]=result["paired_comparisons"]
        print(json.dumps(display,indent=2))
    except Exception:
        write_once(command.with_suffix(".failed.json"),{"status":"failed","traceback":traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()

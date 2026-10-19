"""Replay the deterministic source-URL date audit; no inference or training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cmin_panel_data import read, write_once


def audit(directory):
    directory=Path(directory)
    groups=defaultdict(list)
    with (directory/"selected_stories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            story=json.loads(line)
            if story.get("source_link"):
                groups[story["source_link"]].append({key:story[key] for key in
                    ("ticker","source_date","arrival_session","id")})
    conflicts={url:records for url,records in groups.items() if len({r["source_date"] for r in records})>1}
    result={"exact_url_groups_with_disagreeing_source_dates":len(conflicts),"groups":conflicts,
            "interpretation":"Observed source-date disagreement or updating; historical original versions and publication clock unresolved. No claim of genuine publication-age decay."}
    path=directory/"source_date_disagreement_audit.json"
    if path.exists():
        if read(path)!=result:
            raise ValueError("Existing source-date audit differs; preserve and version the correction")
    else:
        write_once(path,result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--directory",default="artifacts/cmin_panel_2026-10-14_v1")
    args=parser.parse_args()
    result=audit(args.directory)
    print(json.dumps({"conflicting_source_date_url_groups":result["exact_url_groups_with_disagreeing_source_dates"]}))

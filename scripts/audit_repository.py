"""Offline audit of the Git sharing package; standard library only.

Reads indexed files and current working-tree bytes. No downloads, model imports,
training, or experiment writes. Prints JSON; redirect it to save an audit.
Historical completion manifests may reference deliberately untracked local files.

See docs/code_guide.md#pipeline-entry-points for the code map, contracts, and execution boundaries.
"""

import ast
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args]).decode("utf-8")


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8-sig"))


def digest(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def audit():
    tracked = set(git("ls-files", "-z").split("\0")) - {""}
    errors = []
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_commit": git("rev-parse", "HEAD").strip(),
        "scope": "Git-indexed paths, current working-tree bytes; offline sharing audit, not model replay or a financial experiment",
        "limitations": [
            "No external URL availability or upstream source-clock verification",
            "No neural checkpoint replay; full checkpoints/data/caches are not in Git",
            "Historical completion manifests include omitted workstation-only files",
            "CSV score checks verify arithmetic and matching rows, not generalization or absence of leakage",
        ],
    }

    required = ["README.md", "DECISIONS.md", "AGENTS.md", "docs/repository_guide.md", "docs/code_guide.md",
                "docs/cmin_pipeline.md", "docs/cmin_runbook.md", "docs/laptop_sync.md",
                "reports/cmin_timing_2026-10-19.md"]
    for name in required:
        if name not in tracked:
            errors.append(f"Required entry point is not tracked: {name}")

    counts, sizes, syntax_count, json_count = Counter(), Counter(), 0, 0
    secret_pattern = re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|sk-(?:proj-)?[A-Za-z0-9_-]{35,}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
    forbidden_parts = {".venv", "__pycache__", ".pytest_cache", "node_modules", "tmp"}
    forbidden_suffixes = {".pyc", ".pyo", ".pt", ".npy", ".npz", ".safetensors", ".aux", ".log"}
    large_files = []
    internal_imports_checked = 0
    guide_backlinks_checked = 0
    for name in sorted(tracked):
        path = ROOT / name
        if not path.is_file():
            errors.append(f"Tracked file missing: {name}")
            continue
        size = path.stat().st_size
        counts[name.split("/")[0]] += 1
        sizes[name.split("/")[0]] += size
        if forbidden_parts.intersection(path.relative_to(ROOT).parts) or path.suffix in forbidden_suffixes:
            errors.append(f"Generated/local-only file is tracked: {name}")
        if name.startswith(("data/raw/", "data/processed/", "data/research_v2/")) or (path.name.startswith(".env") and path.name != ".env.example"):
            errors.append(f"Raw data or credentials file is tracked: {name}")
        if size > 1024 * 1024:
            large_files.append({"path": name, "bytes": size})
        if size > 10 * 1024 * 1024:
            errors.append(f"Unexpected file above sharing budget of 10 MiB: {name}")
        if path.suffix in {".py", ".md", ".json", ".toml", ".txt", ".tex", ".csv", ".html"}:
            try:
                text = path.read_text(encoding="utf-8-sig")
                if secret_pattern.search(text):
                    errors.append(f"Possible credential detected; inspect file without printing its value: {name}")
                if path.suffix == ".py":
                    tree = ast.parse(text, filename=name)
                    syntax_count += 1
                    if name.startswith(("src/", "scripts/", "tests/")):
                        for node in ast.walk(tree):
                            modules = ([node.module] if isinstance(node, ast.ImportFrom) and node.module else
                                       [entry.name for entry in node.names] if isinstance(node, ast.Import) else [])
                            for module in modules:
                                if module == "src" or module.startswith("src."):
                                    target = module.replace(".", "/")
                                    internal_imports_checked += 1
                                    if target + ".py" not in tracked and target + "/__init__.py" not in tracked:
                                        errors.append(f"Missing internal import: {name} -> {module}")
                        if name.startswith(("src/", "scripts/")):
                            guide_backlinks_checked += 1
                            if "docs/code_guide.md" not in (ast.get_docstring(tree) or ""):
                                errors.append(f"Missing module-docstring code-guide reference: {name}")
                elif path.suffix == ".json":
                    json.loads(text)
                    json_count += 1
            except (ValueError, SyntaxError, UnicodeError) as exc:
                errors.append(f"Invalid text/JSON/Python: {name} ({type(exc).__name__})")
    result["inventory"] = {"tracked_files": len(tracked), "bytes": sum(sizes.values()),
        "directories": {key: {"files": counts[key], "bytes": sizes[key]} for key in sorted(counts)},
        "files_above_1_mib": large_files, "python_files_parsed": syntax_count, "json_files_parsed": json_count,
        "credential_scan": "high-confidence token/private-key patterns only; not an exhaustive secret audit"}

    link_count = 0
    # Frozen artifact notes can refer to the full local run; primary docs must link to Git-visible evidence.
    docs = sorted(p for p in tracked if p.endswith(".md") and not p.startswith("artifacts/"))
    for name in docs:
        if not (ROOT / name).exists():
            continue
        text = (ROOT / name).read_text(encoding="utf-8-sig")
        for match in re.finditer(r"\[[^\]\n]+\]\(([^)]+)\)", text):
            target = match.group(1).strip().strip("<>")
            if target.startswith(("https:", "http:", "mailto:", "#")):
                continue
            resolved = ((ROOT / name).parent / unquote(urlsplit(target).path)).resolve()
            try:
                relative = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                errors.append(f"Nonportable documentation link: {name}")
                continue
            link_count += 1
            if relative not in tracked and not any(p.startswith(relative.rstrip("/") + "/") for p in tracked):
                errors.append(f"Documentation target absent from Git: {name} -> {relative}")
    result["documentation"] = {"markdown_files_checked": len(docs), "local_links_checked": link_count,
        "internal_imports_checked": internal_imports_checked, "module_guide_backlinks_checked": guide_backlinks_checked,
        "scope": "Markdown outside frozen artifacts; existence in Git, not fragment/anchor validation"}

    manifests = []
    for name in sorted(p for p in tracked if p.startswith("reports/provenance/research_") and p.endswith(".json")):
        manifest = read(name)
        entries = manifest.get("files", [])
        for entry in entries:
            path = entry["path"]
            if path not in tracked:
                errors.append(f"Manifest file not tracked: {path}")
            elif digest(path) != entry["sha256"] or (ROOT / path).stat().st_size != entry["bytes"]:
                errors.append(f"Sharing-manifest byte mismatch: {path}")
        manifests.append({"path": name, "files_checked": len(entries)})
    result["sharing_manifests"] = manifests

    pdf_checks = []
    for name in sorted(p for p in tracked if p.endswith(".build.json") and p.startswith("output/pdf/")):
        build = read(name)
        prefix = name.removesuffix(".build.json")
        pdf, tex = prefix + ".pdf", prefix + ".tex"
        if pdf not in tracked or digest(pdf) != build.get("pdf_sha256"):
            errors.append(f"PDF build hash mismatch: {pdf}")
        expected_tex = build.get("tex_sha256") or build.get("sources_sha256", {}).get(tex)
        if expected_tex and (tex not in tracked or digest(tex) != expected_tex):
            errors.append(f"LaTeX build-source hash mismatch: {tex}")
        pdf_checks.append({"pdf": pdf, "pdf_hash_matches": pdf in tracked and digest(pdf) == build.get("pdf_sha256"),
                           "latex_hash_checked": bool(expected_tex), "recorded_pages": build.get("pages")})
    for name in (p for p in tracked if p.endswith(".pdf")):
        if not (ROOT / name).read_bytes().startswith(b"%PDF-"):
            errors.append(f"Invalid PDF header: {name}")
    result["pdf_builds"] = pdf_checks

    reference_rows = {}
    aggregates = defaultdict(lambda: defaultdict(list))
    fits, max_loss_error, max_accuracy_error = [], 0.0, 0.0
    prediction_paths = sorted(p for p in tracked if p.startswith("artifacts/cmin_") and p.endswith("/predictions.csv"))
    for name in prediction_paths:
        path = ROOT / name
        summary_name = path.with_name("summary.json").relative_to(ROOT).as_posix()
        summary = read(summary_name)
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        keys, losses, correct = [], [], []
        for row in rows:
            y, z = int(row["target"]), float(row["logit"])
            if y not in (0, 1) or not math.isfinite(z):
                errors.append(f"Invalid target/logit: {name}")
                continue
            if row["forecast_date"] >= row["target_date"]:
                errors.append(f"Nonfuture target date: {name}")
            keys.append(tuple(row[k] for k in ("ticker", "forecast_date", "target_date", "cutoff_utc", "target")))
            losses.append(max(z, 0.0) + math.log1p(math.exp(-abs(z))) - y * z)
            correct.append(int((z >= 0) == y))
        if not losses:
            errors.append(f"Empty predictions: {name}")
            continue
        if len(set(keys)) != len(keys):
            errors.append(f"Duplicate forecast rows: {name}")
        fold = summary["fold"]
        if fold in reference_rows and keys != reference_rows[fold]:
            errors.append(f"CMIN comparison rows/targets not matched: {name}")
        reference_rows[fold] = keys
        loss, accuracy = math.fsum(losses) / len(losses), sum(correct) / len(correct)
        # The archived timing runner used a different, retained summary schema.
        expected = summary["evaluation_network"] if "evaluation_network" in summary else summary["raw"]
        loss_error, accuracy_error = abs(loss - expected["log_loss"]), abs(accuracy - expected["accuracy"])
        max_loss_error, max_accuracy_error = max(max_loss_error, loss_error), max(max_accuracy_error, accuracy_error)
        if loss_error > 1e-10 or accuracy_error > 1e-12 or expected["count"] != len(rows):
            errors.append(f"Saved raw-network metric mismatch: {name}")
        study = path.parents[2].relative_to(ROOT).as_posix()
        arm = summary.get("arm", path.parent.name.rsplit("_", 1)[0])
        aggregates[(study, arm)][summary["seed"]].extend(losses)
        fits.append({"path": name, "company_days": len(rows), "log_loss": loss, "accuracy": accuracy})

    mean_losses = {}
    for (study, arm), seeds in sorted(aggregates.items()):
        mean_loss = math.fsum(math.fsum(values) / len(values) for values in seeds.values()) / len(seeds)
        mean_losses[(study, arm)] = mean_loss
        summary_path = study + "/summary.json"
        if summary_path in tracked:
            arms = read(summary_path).get("arms", {})
            if arm in arms and abs(mean_loss - arms[arm]["mean_seed_log_loss"]) > 1e-10:
                errors.append(f"Aggregate arm loss mismatch: {study}/{arm}")

    timing = read("artifacts/cmin_timing_2026-10-19_v1/analysis.json")["metrics"]
    metric_sources = {
        "delay2": ("artifacts/cmin_panel_2026-10-14_v1", "finin"),
        "price_only": ("artifacts/cmin_panel_2026-10-14_v1", "price_only"),
        **{policy: (f"artifacts/cmin_timing_2026-10-19_v1/{policy}", "finin") for policy in ("date0", "delay1", "clock_ny")},
    }
    for label, source in metric_sources.items():
        if source not in mean_losses or abs(mean_losses[source] - timing[label]["mean_seed_log_loss"]) > 1e-10:
            errors.append(f"Timing analysis aggregate mismatch: {label}")
    older = read("artifacts/cmin_older_news_2026-10-14_v1/analysis/summary.json")["metrics"]
    for label, source in {
        "all_news": ("artifacts/cmin_decay_48h_2026-10-14_v1", "pooled_uniform"),
        "recent_news": ("artifacts/cmin_older_news_2026-10-14_v1/recent", "pooled_uniform"),
        "price_only": ("artifacts/cmin_older_news_2026-10-14_v1/price", "price_only"),
    }.items():
        if source not in mean_losses or abs(mean_losses[source] - older[label]["mean_seed_log_loss"]) > 1e-10:
            errors.append(f"Older-news analysis aggregate mismatch: {label}")
    result["prediction_checks"] = {"fits_checked": len(fits), "stored_csv_rows": sum(v["company_days"] for v in fits),
        "matched_evaluation_company_days": sum(len(rows) for rows in reference_rows.values()),
        "maximum_log_loss_error": max_loss_error, "maximum_accuracy_error": max_accuracy_error,
        "aggregate_mean_seed_losses": [{"study": study, "arm": arm, "log_loss": value}
                                        for (study, arm), value in sorted(mean_losses.items())]}

    result["errors"] = errors
    result["status"] = "pass" if not errors else "fail"
    return result


if __name__ == "__main__":
    output = audit()
    print(json.dumps(output, indent=2, ensure_ascii=True))
    sys.exit(0 if output["status"] == "pass" else 1)

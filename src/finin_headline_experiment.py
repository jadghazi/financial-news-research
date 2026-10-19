"""Bounded 2023 DEVELOPMENT-only rolling FININ headline controls.

Only raw-BCE selection blocks reach the fitting engine. A fold's evaluation
dates never select that fold's epochs, parameters, transforms or fallbacks.
Past dates can become later folds' training/selection observations under the
fixed rolling schedule. Histories retain the same
five-session eligible centers across all four three-day models. Publication and
price availability remain source metadata assertions rather than verified vintages.
"""
from __future__ import annotations

import csv
import copy
from datetime import datetime, timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import subprocess
import time
import traceback

import numpy as np
import torch
from torch import nn

from src.encoder_comparison import metrics
from src.finin_history_data import HistoryCohort, HistoryDataset, _time, fit_history_scaler
from src.models.research_finin import ResearchFININ
from src.research_training import fixed_budget_fit, isolated_diagnostics, predict_logits
from src.reproduction_v2_data import sha256_file
from src.reproduction_v2_features import ENCODERS, SENTIMENT

ARMS = ("full_concat", "mean_concat", "matched_no_news", "price_history_mlp")
SEEDS = (17, 42, 73)
LINEAR_ARMS = ("linear_prices", "linear_prices_coverage", "linear_prices_coverage_news")
BOUNDARY = "2024-01-01"
TRAINING_START = "2022-09-20"
WINDOWS = (("2022-12-31", "2023-01-01", "2023-02-01", "2023-04-01"),
           ("2023-04-30", "2023-05-01", "2023-06-01", "2023-08-01"),
           ("2023-06-30", "2023-07-01", "2023-08-01", "2023-10-01"))
COLLECTION_FIELDS = ("collection_status", "missing_collection_dates", "collection_missing",
                     "observed_empty_news", "missing_snapshot_dates", "source_window_missing_snapshots",
                     "news_count_precap", "news_count_retained", "availability_policy",
                     "collection_status_basis", "source_collection_missing", "source_collection_complete",
                     "news_observed_empty", "collection_missing_dates")
SOURCES = ("src/finin_headline_experiment.py", "tests/test_finin_headline_experiment.py",
           "src/finin_history_data.py", "src/models/research_finin.py",
           "src/models/paper_finin.py", "src/research_training.py",
           "src/encoder_comparison.py", "src/evaluation.py", "src/training.py",
           "src/reproduction_v2_data.py")
SOURCES += ("src/finin_headline_controls.py", "src/financial_headline_cache.py", "src/reproduction_v2_features.py")


def _write(path, value):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite evidence: {path}")
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_frozen_features(features):
    """Require pinned identities in both requested and actual encoder metadata.

    Content hashes are verified by FrozenFeatures and model-file provenance by
    the pinned cache builder; width alone never identifies a text encoder.
    """
    manifest = features.manifest
    identity = manifest.get("identity", {})
    if features.text.ndim != 2 or features.text.shape[1] != 768:
        raise ValueError("Pinned frozen RoBERTa CLS width768 required")
    if identity.get("encoder") != ENCODERS["roberta"] or identity.get("sentiment") != SENTIMENT:
        raise ValueError("Requested encoder/sentiment identity is not the exact pinned configuration")
    if identity.get("pooling") != "last_hidden_state[:,0,:]" or identity.get("l2_normalized") is not False:
        raise ValueError("Frozen raw CLS pooling required; no hidden normalization changes")
    for kind, expected in (("text", ENCODERS["roberta"]), ("sentiment", SENTIMENT)):
        actual = manifest.get("states", {}).get(kind, {}).get("model_provenance", {})
        if actual.get("model") != expected["name"] or actual.get("revision") != expected["revision"]:
            raise ValueError(f"Actual {kind} encoder provenance differs from the exact pinned model/revision")
    return True


def _git_identity(root):
    """Record tracked dirty-diff identity without exposing its contents."""
    try:
        options = {"cwd": root, "capture_output": True, "check": True}
        revision = subprocess.run(["git", "rev-parse", "HEAD"], **options).stdout.decode("ascii").strip()
        status = subprocess.run(["git", "status", "--porcelain=v1", "--untracked-files=all"], **options).stdout
        diff = subprocess.run(["git", "diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD"], **options).stdout
        return {"revision": revision, "dirty": bool(status), "status_sha256": hashlib.sha256(status).hexdigest(),
                "tracked_dirty_diff_sha256": hashlib.sha256(diff).hexdigest(),
                "status_lines": status.decode("utf-8", errors="replace").splitlines(),
                "scope": "Tracked HEAD diff plus status identity; exact study source files separately snapshotted/hashed"}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "reason": type(exc).__name__}


def rolling_plans(centers):
    """Fixed target-date blocks; labels must strictly precede next fit cutoff.

    Equality is excluded too, matching fit_history_scaler's strict cutoff guard.
    The recorded purge lists make this conservative boundary choice explicit.
    """
    if not centers or any(r["split"] not in ("train", "valid") or not r["target_date"] < BOUNDARY for r in centers):
        raise ValueError("Only nonempty development centers before 2024 allowed")
    keys = [(r["ticker"], r["target_date"]) for r in centers]
    if len(set(keys)) != len(keys) or len({r["ticker"] for r in centers}) != 1:
        raise ValueError("Require one market and one center per target date")
    plans = []
    for fold, (train_end, selection_start, evaluation_start, evaluation_end) in enumerate(WINDOWS):
        candidates = [i for i, r in enumerate(centers) if TRAINING_START <= r["target_date"] <= train_end]
        selection_candidates = [i for i, r in enumerate(centers) if selection_start <= r["target_date"] < evaluation_start]
        evaluation = [i for i, r in enumerate(centers) if evaluation_start <= r["target_date"] < evaluation_end]
        if not candidates or not selection_candidates or not evaluation:
            raise ValueError(f"Empty fixed train/selection/evaluation block: {fold}")
        first_selection = min(_time(centers[i]["cutoff_utc"]) for i in selection_candidates)
        first_evaluation = min(_time(centers[i]["cutoff_utc"]) for i in evaluation)
        train = [i for i in candidates if _time(centers[i]["target_available_utc"]) < first_selection]
        selection = [i for i in selection_candidates if _time(centers[i]["target_available_utc"]) < first_evaluation]
        if not train or not selection:
            raise ValueError("Availability purge leaves empty training or selection")
        plan = {"fold": fold, "train_indices": train, "selection_indices": selection,
                "evaluation_indices": evaluation,
                "training_purged_indices": [i for i in candidates if i not in train],
                "selection_purged_indices": [i for i in selection_candidates if i not in selection],
                "fit_cutoff_utc": first_selection.isoformat(), "evaluation_cutoff_utc": first_evaluation.isoformat(),
                "target_boundaries": {"train_through": train_end, "selection_from": selection_start,
                                      "evaluation_from": evaluation_start, "evaluation_before": evaluation_end},
                "availability_rule": "target_available_utc strictly before next block's first forecast cutoff"}
        for role in ("train", "selection", "evaluation"):
            chosen = [centers[i] for i in plan[f"{role}_indices"]]
            plan[f"{role}_ids"] = [r["id"] for r in chosen]
            plan[f"{role}_target_dates"] = [r["target_date"] for r in chosen]
        plans.append(plan)
    evdates = [d for p in plans for d in p["evaluation_target_dates"]]
    if len(evdates) != len(set(evdates)):
        raise ValueError("Outer evaluation dates overlap")
    return plans


def fit_text_transform(cohort, features, indices, *, fit_cutoff_utc, mode):
    """Unique training-history headline CLS center and ONE scalar RMS scale.

    Descriptions are transformed consistently but never fit their statistics.
    No sentiment transform, PCA or per-coordinate standardization is introduced.
    """
    if mode not in ("none", "center_rms") or fit_cutoff_utc is None:
        raise ValueError("Explicit fixed transform mode and fitting cutoff required")
    cutoff = _time(fit_cutoff_utc)
    indices = list(indices)
    if not indices or len(indices) != len(set(indices)) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or not 0 <= i < len(cohort) for i in indices):
        raise ValueError("Unique valid training center indices required")
    identifiers = set()
    for i in indices:
        center = cohort.centers[i]
        if _time(center["target_available_utc"]) >= cutoff:
            raise ValueError("Transform fitting label unavailable at cutoff")
        for row in cohort.histories[i]:
            if _time(row["cutoff_utc"]) >= cutoff:
                raise ValueError("Transform fitting history unavailable at cutoff")
            identifiers.update(message["id"] for message in row["headlines"])
    ids = sorted(identifiers)
    if not ids:
        raise ValueError("No fitting headlines for frozen representation transform")
    values = np.asarray(features.vectors(ids), dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Invalid fitting CLS matrix")
    mean = values.mean(axis=0) if mode == "center_rms" else np.zeros(values.shape[1])
    rms = float(np.sqrt(np.mean((values - mean) ** 2))) if mode == "center_rms" else 1.
    if not np.isfinite(rms) or rms <= 1e-12:
        if mode == "center_rms":
            raise ValueError("Degenerate training headline RMS")
    return {"mode": mode, "mean": mean.tolist(), "rms": rms, "fit_cutoff_utc": fit_cutoff_utc,
            "unique_headline_ids": ids, "training_center_ids": [cohort.centers[i]["id"] for i in indices],
            "fit_values_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
            "definition": "unique training-history headlines, vector center and global scalar RMS; descriptions excluded from fitting"}


class TransformFeatures:
    """Read-only adapter; cached arrays and backing feature object are untouched."""
    def __init__(self, features, transform):
        self.features, self.transform, self.text = features, transform, features.text
        self.mean = np.asarray(transform["mean"], dtype=np.float64)
        self.rms = float(transform["rms"])

    def vectors(self, ids):
        return ((np.asarray(self.features.vectors(ids), dtype=np.float64) - self.mean) / self.rms).astype(np.float32)

    def sentiments(self, ids):
        return np.array(self.features.sentiments(ids), dtype=np.float32, copy=True)


class PriceHistoryMLP(nn.Module):
    """Small comparator: all three price days plus fixed description projection."""
    def __init__(self, text_width=768, width=32, history_days=3, dropout=.1):
        super().__init__()
        self.history_days = history_days
        self.description = nn.Linear(text_width, width)
        self.predictor = nn.Sequential(nn.Linear(6 * history_days + width, width), nn.ReLU(),
                                       nn.Dropout(dropout), nn.Linear(width, 1))

    def forward(self, batch):
        price, description = batch["market_features"], batch["market_description"]
        if price.ndim != 3 or price.shape[1:] != (self.history_days, 6):
            raise ValueError("Price-history comparator received wrong geometry")
        values = torch.cat((price.flatten(1), self.description(description)), dim=1)
        return self.predictor(values).squeeze(-1), price.new_zeros((price.shape[0], self.history_days, 1))


def model_factory(arm):
    if arm not in ARMS:
        raise ValueError("Unknown fixed experiment arm")
    if arm == "price_history_mlp":
        return PriceHistoryMLP()
    return ResearchFININ(text_width=768, width=32, attention_width=32, attention_heads=1,
                         history_days=3, mlp_layers=2, dropout=.1, attention_style="plain",
                         interaction=arm != "mean_concat", query_weighting=arm != "mean_concat",
                         news_available=arm != "matched_no_news", temporal_aggregation="concat")


def _source_identity(cohort_dir, features, cache_dir, protocol):
    root = Path(__file__).resolve().parents[1]
    result = {"source_sha256": {p: sha256_file(root / p) for p in SOURCES},
              "cohort_sha256": {name: sha256_file(cohort_dir / name) for name in ("provenance.json", "devsamples_post_close.jsonl")},
              "protocol_sha256": sha256_file(protocol), "feature_manifest": features.manifest,
              "amendment_sha256": {p.name: sha256_file(p) for p in sorted(protocol.parent.glob("AMENDMENT*.md"))},
              "git": _git_identity(root),
              "environment": {"python": sys.version, "executable": sys.executable,
                              "torch": str(torch.__version__), "numpy": np.__version__,
                              "platform": platform.platform(), "device": "cpu", "threads": 4}}
    wrapper_source = Path(inspect.getfile(type(features))).resolve()
    if not wrapper_source.is_file():
        raise ValueError("Feature adapter must have an inspectable source module")
    result["feature_wrapper_source"] = {"path": str(wrapper_source), "sha256": sha256_file(wrapper_source)}
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        result["cache_sha256"] = {p.name: sha256_file(p) for p in sorted(cache_dir.iterdir()) if p.is_file()}
    return result


def _prepare(cohort_dir, features, out, *, text_transform, cache_dir):
    validate_frozen_features(features)
    if text_transform not in ("none", "center_rms"):
        raise ValueError("Declare transform switch before any fits")
    if not (out / "PROTOCOL.md").exists():
        raise ValueError("Frozen protocol must exist before preparation")
    cohort = HistoryCohort.from_directory(cohort_dir, "post_close", history_days=3,
                                         common_cohort_max_history=5, development_target_before=BOUNDARY)
    source_rows = [json.loads(line) for line in (cohort_dir / "devsamples_post_close.jsonl").read_text(encoding="utf-8").splitlines()]
    for center, source_index in zip(cohort.centers, cohort.center_source_indices):
        source = source_rows[source_index]
        if source["id"] != center["id"]:
            raise ValueError("Center/source collection metadata identity differs")
        center["collection_metadata"] = {key: source[key] for key in COLLECTION_FIELDS if key in source}
    for center, history_metadata in zip(cohort.centers, history_collection_metadata(cohort, source_rows)):
        center["collection_metadata"].update(history_metadata)
    cohort.provenance["collection_metadata_rule"] = "Explicit source flags/counts preserved; empty mask alone does not establish observed absence of news"
    plans = rolling_plans(cohort.centers)
    prepared = []
    for plan in plans:
        item = dict(plan)
        item["source_splits"] = {role: [cohort.centers[i]["split"] for i in item[f"{role}_indices"]] for role in ("train", "selection", "evaluation")}
        item["fold_ownership"] = {role: item[f"{role}_ids"] for role in ("train", "selection", "evaluation")}
        training_copy = fold_training_cohort(cohort, item["train_indices"])
        item["market_scaler"] = fit_history_scaler(training_copy, item["train_indices"], fit_cutoff_utc=item["fit_cutoff_utc"])
        item["text_transform"] = fit_text_transform(cohort, features, item["train_indices"],
                                                   fit_cutoff_utc=item["fit_cutoff_utc"], mode=text_transform)
        prepared.append(item)
    config = {"arms": list(ARMS), "seeds": list(SEEDS), "history_days": 3, "common_cohort_history_days": 5,
              "text_width": 768, "width": 32, "attention_style": "plain", "text_transform": text_transform,
              "training_target_from": TRAINING_START,
              "target_windows": [list(window) for window in WINDOWS],
              "optimizer": "adamw", "learning_rate": .001, "weight_decay": .01, "epochs": 30,
              "batch_size": 16, "clip": 1., "dropout": .1, "fits": 36, "linear_fits": 9,
              "linear_arms": list(LINEAR_ARMS), "hyperparameter_candidates": 1,
              "linear_configuration": {"l2": .01, "dtype": "float64", "optimizer": "LBFGS",
                                       "max_iter": 1000, "tolerance_grad": 1e-9, "convergence_max_abs_gradient": 1e-5,
                                       "objective": "meanBCE + .01/2 sum(weights squared); intercept unpenalized",
                                       "selection": "Fixed configuration; no validation selection"},
              "linear_news_control": "Secondary: one pooled mean across all active T3 frozen CLS vectors and the same mean sentiment, zeros if empty, added to prices+coverage. High-dimensional 768-vector input and fixed penalty differ from smaller controls; no PCA or penalty search.",
              "selection": "minimum raw BCE on declared selection dates; earliest exact tie",
              "matched_no_news_transform": "Shares training-news-fitted text transform for matched geometry, applied to fixed description; headline values and masks inactive",
              "price_comparator": "Same history and description, smaller capacity; not a parameter-matched ablation",
              "evaluation": "DEVELOPMENT rolling blocks; no untouched-test or tradable-performance claim"}
    config["complete_history_sensitivity"] = "One predeclared paired subset: three source collection flags explicitly known and no missing snapshots; same IDs across all arms; no refitting"
    identity = _source_identity(cohort_dir, features, cache_dir, out / "PROTOCOL.md")
    return cohort, prepared, config, identity


def fold_training_cohort(cohort, train_indices):
    """Explicit per-fold ownership copy, preserving original source metadata."""
    selected = set(train_indices)
    result = copy.copy(cohort)
    result.centers = [dict(row, source_split=row["split"], split="train" if i in selected else "valid")
                      for i, row in enumerate(cohort.centers)]
    return result


def history_collection_metadata(cohort, source_rows):
    """Map ONLY explicit three-input-day collection flags; never infer from labels.

    Missing source flags make sensitivity eligibility unavailable, rather than
    declaring an empty headline day observed. Metadata never becomes a feature.
    """
    source = {}
    for row in source_rows:
        key = (row["ticker"], row["forecast_schedule"], row["price_date"])
        if key in source:
            raise ValueError("Duplicate collection metadata source session")
        source[key] = row
    output = []
    for history in cohort.histories:
        if len(history) != 3:
            raise ValueError("Complete-history sensitivity requires exactly three input days")
        days, missing_dates, flags = [], set(), []
        for row in history:
            key = (row["ticker"], row["forecast_schedule"], row["price_date"])
            if key not in source:
                raise ValueError("History session lacks original source metadata")
            original = source[key]
            flag = original.get("collection_missing")
            missing = original.get("missing_collection_dates")
            known = isinstance(flag, bool) and isinstance(missing, list) and all(isinstance(day, str) for day in missing)
            if known and missing and flag is False:
                raise ValueError("Complete collection flag contradicts missing snapshot dates")
            flags.append(flag if known else None)
            if known:
                missing_dates.update(missing)
            days.append(row["price_date"])
        available = all(flag is not None for flag in flags)
        output.append({"history_collection_flags_available": available,
                       "history_collection_missing": any(flags) if available else None,
                       "history_missing_collection_dates": sorted(missing_dates),
                       "history_collection_price_dates": days,
                       "history_collection_scope": "Three explicit source proxy collection windows; no verified point-in-time availability"})
    return output


def prepare(cohort_dir, features, out, *, text_transform="center_rms", cache_dir=None):
    cohort_dir, out = Path(cohort_dir), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "identity.json").exists() or (out / "plans.json").exists():
        raise ValueError("Preparation already exists; no unknown resume")
    unexpected = [p.name for p in out.iterdir() if not p.is_file() or not
                  (p.name == "PROTOCOL.md" or (p.name.upper().startswith("AMENDMENT") and p.suffix == ".md")
                   or (p.name.startswith("command_") and p.suffix == ".json"))]
    if unexpected:
        raise ValueError(f"Untracked nonempty namespace: {unexpected}")
    torch.set_num_threads(4)
    cohort, plans, config, identity = _prepare(cohort_dir, features, out, text_transform=text_transform, cache_dir=cache_dir)
    _write(out / "identity.json", identity)
    _write(out / "configuration.json", config)
    _write(out / "plans.json", plans)
    _write(out / "cohort.json", {"centers": cohort.centers, "center_source_indices": cohort.center_source_indices,
                                "provenance": cohort.provenance})
    folder = out / "source_snapshot"
    folder.mkdir()
    root = Path(__file__).resolve().parents[1]
    for source in SOURCES:
        (folder / source.replace("/", "__")).write_bytes((root / source).read_bytes())
    (folder / "feature_wrapper.py").write_bytes(Path(identity["feature_wrapper_source"]["path"]).read_bytes())
    _write(out / "prepared_hashes.json", {name: sha256_file(out / name) for name in
                                         ("identity.json", "configuration.json", "plans.json", "cohort.json")})
    return {"configuration": config, "fold_counts": [{role: len(p[f"{role}_indices"]) for role in ("train", "selection", "evaluation")} for p in plans]}


def _predictions(path, rows, logits):
    if path.exists() or len(rows) != len(logits):
        raise ValueError("Prediction overwrite or row count mismatch")
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("id", "ticker", "target_date", "cutoff_utc", "target_up", "logit", "collection_metadata"))
        writer.writeheader()
        for row, logit in zip(rows, logits):
            writer.writerow({k: row[k] for k in ("id", "ticker", "target_date", "cutoff_utc", "target_up")} | {
                "logit": float(logit), "collection_metadata": json.dumps(row.get("collection_metadata", {}), sort_keys=True)})


def fit_control_arm(fit_function, dataset, arm):
    """Explicit information switches; no implicit coverage/news defaults."""
    if arm not in LINEAR_ARMS:
        raise ValueError("Unknown fixed linear control arm")
    return fit_function(dataset, coverage=arm != "linear_prices", news=arm == "linear_prices_coverage_news")


def run(cohort_dir, features, out, *, text_transform="center_rms", cache_dir=None):
    """Execute only after root source/protocol review; no implicit prepare/resume."""
    cohort_dir, out = Path(cohort_dir), Path(out)
    torch.set_num_threads(4)
    command = {"started_utc": datetime.now(timezone.utc).isoformat(), "argv": sys.argv.copy(),
               "cohort_dir": str(cohort_dir.resolve()), "output": str(out.resolve()),
               "stage": "train", "status": "started"}
    _write(out / f"command_train_{time.time_ns()}.json", command)
    started = time.monotonic()
    try:
        for name, digest in json.loads((out / "prepared_hashes.json").read_text()).items():
            if sha256_file(out / name) != digest:
                raise ValueError("Prepared experiment evidence changed")
        cohort, plans, config, identity = _prepare(cohort_dir, features, out, text_transform=text_transform, cache_dir=cache_dir)
        if identity != json.loads((out / "identity.json").read_text()) or plans != json.loads((out / "plans.json").read_text()) or config != json.loads((out / "configuration.json").read_text()):
            raise ValueError("Frozen source, config or fold preparation changed")
        fits = out / "fits"
        if fits.exists():
            raise ValueError("Attempted/completed fits retained; no implicit resume")
        fits.mkdir()
        from src.finin_headline_controls import fit as fit_control, predict as predict_control
        records, control_records = [], []
        for plan in plans:
            adapter = TransformFeatures(features, plan["text_transform"])
            datasets = {role: HistoryDataset(cohort, adapter, plan["market_scaler"], indices=plan[f"{role}_indices"], max_headlines=64)
                        for role in ("train", "selection", "evaluation")}
            for control in LINEAR_ARMS:
                folder = fits / f"f{plan['fold']}_{control}"
                folder.mkdir()
                _write(folder / "configuration.json", {"arm": control, "fold": plan["fold"],
                        "selection": "None; declared fixed hyperparameters, training-only preprocessing",
                        "plan_sha256": sha256_file(out / "plans.json")})
                fitted = fit_control_arm(fit_control, datasets["train"], control)
                _write(folder / "model.json", fitted)
                scores = {role: predict_control(fitted, datasets[role]) for role in ("train", "selection", "evaluation")}
                measured = {}
                for role, logits in scores.items():
                    _predictions(folder / f"{role}.csv", datasets[role].rows, logits)
                    measured[role] = metrics(np.asarray([r["target_up"] for r in datasets[role].rows]), logits)
                record = {"fold": plan["fold"], "arm": control, "metrics": measured,
                          "artifact_sha256": {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file()}}
                _write(folder / "summary.json", record)
                control_records.append(record)
            for arm in ARMS:
                for seed in SEEDS:
                    folder = fits / f"f{plan['fold']}_{arm}_{seed}"
                    folder.mkdir()
                    _write(folder / "configuration.json", config | {"arm": arm, "seed": seed, "fold": plan["fold"], "plan_sha256": sha256_file(out / "plans.json")})
                    fit = fixed_budget_fit(lambda: model_factory(arm), {"train": datasets["train"], "valid": datasets["selection"]},
                                           seed, epochs=30, batch_size=16, learning_rate=.001, clip=1.,
                                           optimizer_name="adamw", weight_decay=.01)
                    gradients = {name: float(p.grad.norm()) if p.grad is not None else None for name, p in fit["model"].named_parameters()}
                    changes = {name: float((value-fit["initial_state"][name]).double().norm())
                               for name, value in fit["selected_state"].items() if value.is_floating_point()}
                    with isolated_diagnostics(fit["model"]):
                        initial_model = model_factory(arm)
                        initial_model.load_state_dict(fit["initial_state"])
                        initial_logits = predict_logits(initial_model, datasets["train"])
                    train_y = np.asarray([r["target_up"] for r in datasets["train"].rows])
                    for role in ("initial", "selected", "final"):
                        torch.save(fit[f"{role}_state"], folder / f"{role}.pt")
                    _write(folder / "history.json", fit["history"])
                    np.savez_compressed(folder / "epoch_logits.npz",
                                        train=np.stack([p["logits"]["train"] for p in fit["epoch_predictions"]]),
                                        selection=np.stack([p["logits"]["valid"] for p in fit["epoch_predictions"]]))
                    logits = predict_logits(fit["model"], datasets["evaluation"])
                    y = np.asarray([r["target_up"] for r in datasets["evaluation"].rows])
                    prior = fit["training_prior"]
                    constant = np.full(len(y), np.log(prior / (1-prior)))
                    _predictions(folder / "evaluation.csv", datasets["evaluation"].rows, logits)
                    _predictions(folder / "constant.csv", datasets["evaluation"].rows, constant)
                    summary = {"fold": plan["fold"], "arm": arm, "seed": seed, "selected_epoch": fit["selected_epoch"],
                               "initial_train_metrics": metrics(train_y, initial_logits),
                               "selected_train_metrics": metrics(train_y, fit["selected_logits"]["train"]),
                               "selected_selection_metrics": metrics(np.asarray([r["target_up"] for r in datasets["selection"].rows]), fit["selected_logits"]["valid"]),
                               "last_optimizer_batch_gradient_norms": gradients,
                               "gradient_scope": "Final epoch last optimizer batch, clipped gradients; checkpoint subsequently restored to selected epoch",
                               "selected_parameter_change_norms": changes,
                               "training_prior": prior, "selection_constant": fit["constant_validation"],
                               "network_metrics": metrics(y, logits), "constant_metrics": metrics(y, constant),
                               "parameters": sum(p.numel() for p in fit["model"].parameters()),
                               "artifact_sha256": {p.name: sha256_file(p) for p in folder.iterdir() if p.is_file()},
                               "fallback": "Not applied; raw selected network and constant reported separately"}
                    _write(folder / "summary.json", summary)
                    records.append(summary)
                    print(f"FININ headline f{plan['fold']} {arm} seed{seed}: BCE{summary['network_metrics']['log_loss']:.6f}", flush=True)
        _write(out / "results.json", records)
        _write(out / "controls_results.json", control_records)
        _write(out / f"command_completed_{time.time_ns()}.json", command | {"status": "completed", "seconds": time.monotonic()-started})
        return records
    except BaseException:
        _write(out / f"failure_{time.time_ns()}.json", command | {"status": "failed", "seconds": time.monotonic()-started, "traceback": traceback.format_exc()})
        raise


def _interval(delta, fold_sizes):
    if not len(delta) or sum(fold_sizes) != len(delta) or any(n < 0 for n in fold_sizes):
        raise ValueError("Nonempty aligned within-fold dated deltas required")
    rng = np.random.default_rng(20261013)
    offsets = np.cumsum([0] + fold_sizes)
    values = []
    for _ in range(2000):
        total = 0.
        for begin, end in zip(offsets[:-1], offsets[1:]):
            n = end-begin
            if n == 0:
                continue
            idx = ((rng.integers(0, n, size=(int(np.ceil(n/5)), 1)) + np.arange(5)) % n).ravel()[:n]
            total += delta[begin+idx].sum()
        values.append(total/len(delta))
    return {"mean_delta": float(delta.mean()), "ci95": np.quantile(values, [.025, .975]).tolist(),
            "block_dates": 5, "replicates": 2000, "seed": 20261013,
            "scope": "Conditional descriptive dated losses; seeds averaged per date, saved fits fixed, no seed independence"}


def _subset_metrics(y, logits):
    measured = metrics(y, logits)
    if not np.isfinite(measured["auc"]):
        measured["auc"] = None
        measured["auc_status"] = "Undefined: subset lacks both classes"
    return measured


def _masked_draw(delta, fold_sizes, complete_mask, rng, block=5):
    """Draw FULL timeline blocks, then retain masked observations within them."""
    offsets = np.cumsum([0] + fold_sizes)
    total, retained = 0., 0
    for begin, end in zip(offsets[:-1], offsets[1:]):
        n = end - begin
        if n == 0:
            continue
        idx = ((rng.integers(0, n, size=(int(np.ceil(n/block)), 1)) + np.arange(block)) % n).ravel()[:n] + begin
        chosen = idx[complete_mask[idx]]
        total += float(delta[chosen].sum())
        retained += len(chosen)
    return total, retained


def _masked_interval(delta, fold_sizes, complete_mask):
    """Conditional sensitivity CI preserving omitted dates in the sampling clock.

    Empty draws are counted and retried, up to 20,000 attempts. The random
    denominator counts only complete-history observations inside sampled blocks.
    A bounded failure is reported instead of fabricating missing replicates.
    """
    delta, mask = np.asarray(delta, dtype=float), np.asarray(complete_mask, dtype=bool)
    if delta.ndim != 1 or mask.shape != delta.shape or sum(fold_sizes) != len(delta) or any(n < 0 for n in fold_sizes) or not mask.any() or not np.isfinite(delta).all():
        raise ValueError("Finite full-timeline deltas and nonempty aligned complete-history mask required")
    rng = np.random.default_rng(20261013)
    values, counts, empty, attempts = [], [], 0, 0
    while len(values) < 2000 and attempts < 20000:
        total, count = _masked_draw(delta, fold_sizes, mask, rng)
        attempts += 1
        if count == 0:
            empty += 1
            continue
        values.append(total/count)
        counts.append(count)
    return {"mean_delta": float(delta[mask].mean()),
            "ci95": np.quantile(values, [.025, .975]).tolist() if len(values) == 2000 else None,
            "status": "available" if len(values) == 2000 else "insufficient nonempty bootstrap draws",
            "block_dates": 5, "replicates": len(values), "requested_replicates": 2000,
            "attempts": attempts, "empty_attempts": empty, "seed": 20261013,
            "original_timeline_counts_by_fold": list(fold_sizes),
            "original_eligible_count": int(mask.sum()),
            "sampled_eligible_count": {"min": min(counts) if counts else None,
                                       "median": float(np.median(counts)) if counts else None,
                                       "max": max(counts) if counts else None},
            "scope": "Full within-fold forecast timeline blocks sampled before complete-history filtering; seeds averaged per date; conditional on nonempty draws and saved fits; no independent replication"}


def analyze(out):
    """Recompute saved prediction evidence, never select settings from evaluation."""
    out = Path(out)
    plans = json.loads((out / "plans.json").read_text())
    summaries = json.loads((out / "results.json").read_text())
    controls = json.loads((out / "controls_results.json").read_text())
    control_lookup = {(r["fold"], r["arm"]): r for r in controls}
    if len(control_lookup) != 9:
        raise ValueError("Require all nine fixed linear controls")
    lookup = {(s["fold"], s["arm"], s["seed"]): s for s in summaries}
    if len(lookup) != 36:
        raise ValueError("Require all 36 fixed fits")
    pooled = {arm: {seed: [] for seed in SEEDS} for arm in ARMS}
    constant_logits = []
    control_pooled, control_losses = {arm: [] for arm in LINEAR_ARMS}, {arm: [] for arm in LINEAR_ARMS}
    labels, losses, per_fold = [], {arm: [] for arm in ARMS}, []
    sensitivity_masks, sensitivity_ids = [], []
    for plan in plans:
        fold_summary = {"fold": plan["fold"], "arms": {}}
        reference_rows = None
        for arm in ARMS:
            arm_loss, arm_metrics = [], []
            for seed in SEEDS:
                saved = lookup[plan["fold"], arm, seed]
                folder = out / "fits" / f"f{plan['fold']}_{arm}_{seed}"
                for name, digest in saved["artifact_sha256"].items():
                    if sha256_file(folder / name) != digest:
                        raise ValueError("Saved experiment artifact changed")
                with (folder / "evaluation.csv").open(encoding="utf-8", newline="") as stream:
                    predictions = list(csv.DictReader(stream))
                ids = [r["id"] for r in predictions]
                if ids != plan["evaluation_ids"]:
                    raise ValueError("Evaluation predictions not aligned to frozen plan")
                metadata = [{k: r[k] for k in ("id", "ticker", "target_date", "cutoff_utc", "target_up", "collection_metadata")} for r in predictions]
                if reference_rows is None:
                    reference_rows = metadata
                    y = np.asarray([float(r["target_up"]) for r in predictions])
                    labels.extend(y.tolist())
                    collection_metadata = [json.loads(r["collection_metadata"]) for r in predictions]
                    complete = np.asarray([m.get("history_collection_flags_available") is True and
                                           m.get("history_collection_missing") is False for m in collection_metadata], dtype=bool)
                    sensitivity_masks.append(complete)
                    sensitivity_ids.append([r["id"] for r, keep in zip(predictions, complete) if keep])
                    fold_summary["complete_history_sensitivity"] = {
                        "eligible_count": int(complete.sum()), "total_count": len(complete),
                        "unknown_collection_history_count": sum(m.get("history_collection_flags_available") is not True for m in collection_metadata),
                        "eligible_ids": sensitivity_ids[-1]}
                    with (folder / "constant.csv").open(encoding="utf-8", newline="") as stream:
                        constant_predictions = list(csv.DictReader(stream))
                    if [r["id"] for r in constant_predictions] != plan["evaluation_ids"]:
                        raise ValueError("Constant baseline dates differ")
                    constants = np.asarray([float(r["logit"]) for r in constant_predictions])
                    constant_logits.extend(constants.tolist())
                    fold_summary["constant_metrics"] = metrics(y, constants)
                elif metadata != reference_rows:
                    raise ValueError("Evaluation dates or targets differ across arms/seeds")
                logits = np.asarray([float(r["logit"]) for r in predictions])
                measured = metrics(y, logits)
                for key, value in measured.items():
                    expected = saved["network_metrics"][key]
                    if (isinstance(value, dict) and value != expected) or (not isinstance(value, dict) and not np.isclose(value, expected, atol=1e-12, rtol=1e-12)):
                        raise ValueError("Saved prediction metric mismatch")
                pooled[arm][seed].extend(logits.tolist())
                arm_loss.append(np.logaddexp(0, logits) - y*logits)
                arm_metrics.append({"seed": seed, "metrics": measured, "selected_epoch": saved["selected_epoch"]})
            losses[arm].append(np.mean(arm_loss, axis=0))
            fold_summary["arms"][arm] = arm_metrics
        for control in LINEAR_ARMS:
            record = control_lookup[plan["fold"], control]
            folder = out / "fits" / f"f{plan['fold']}_{control}"
            for name, digest in record["artifact_sha256"].items():
                if sha256_file(folder / name) != digest:
                    raise ValueError("Linear control evidence changed")
            with (folder / "evaluation.csv").open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            metadata = [{k: r[k] for k in ("id", "ticker", "target_date", "cutoff_utc", "target_up", "collection_metadata")} for r in predictions]
            if metadata != reference_rows:
                raise ValueError("Linear control dates or targets differ")
            logits = np.asarray([float(r["logit"]) for r in predictions])
            measured = metrics(y, logits)
            expected = record["metrics"]["evaluation"]
            for key, value in measured.items():
                if (isinstance(value, dict) and value != expected[key]) or (not isinstance(value, dict) and not np.isclose(value, expected[key], atol=1e-12, rtol=1e-12)):
                    raise ValueError("Linear saved metric differs")
            control_pooled[control].extend(logits.tolist())
            control_losses[control].append(np.logaddexp(0, logits) - y*logits)
            fold_summary[control] = measured
        per_fold.append(fold_summary)
    y = np.asarray(labels)
    contrasts = {}
    for other in ("matched_no_news", "mean_concat"):
        deltas = [a-b for a, b in zip(losses["full_concat"], losses[other])]
        contrasts[f"full_minus_{other}"] = _interval(np.concatenate(deltas), [len(d) for d in deltas]) | {
            "fold_mean_delta": [float(d.mean()) for d in deltas],
            "fold_intervals": [_interval(d, [len(d)]) for d in deltas]}
    for control in ("linear_prices_coverage", "linear_prices_coverage_news"):
        deltas = [a-b for a, b in zip(losses["full_concat"], control_losses[control])]
        contrasts[f"full_minus_{control}_secondary"] = _interval(np.concatenate(deltas), [len(d) for d in deltas]) | {
            "fold_mean_delta": [float(d.mean()) for d in deltas],
            "fold_intervals": [_interval(d, [len(d)]) for d in deltas], "status": "Secondary exploratory capacity-unmatched comparison"}
    direction_fields = ("accuracy", "balanced_accuracy", "up_recall", "down_recall", "confusion", "predicted_down")
    directions = {name: {key: value for key, value in metrics(y, np.full(len(y), logit)).items() if key in direction_fields}
                  for name, logit in (("always_up", 1.), ("always_down", -1.))}
    sensitivity = {"definition": "Exactly three history collection windows explicitly known, none missing; all comparators use identical IDs; no refitting",
                   "per_fold_counts": [int(mask.sum()) for mask in sensitivity_masks],
                   "eligible_ids_by_fold": sensitivity_ids,
                   "scope": "Snapshot collection proxy subset of the same development fits; not independent replication or a historical availability guarantee. Resampling uses five dates on each full within-fold forecast timeline, then filters the sampled observations by complete-history metadata."}
    sensitivity_count = sum(sensitivity["per_fold_counts"])
    if sensitivity_count:
        sensitivity["status"] = "available"
        sensitivity["paired_comparisons"] = {}
        full_mask = np.concatenate(sensitivity_masks)
        for other in ("matched_no_news", "mean_concat"):
            deltas = [a-b for a, b in zip(losses["full_concat"], losses[other])]
            sensitivity["paired_comparisons"][f"full_minus_{other}"] = _masked_interval(np.concatenate(deltas), [len(d) for d in deltas], full_mask) | {
                "fold_mean_delta": [float(d[mask].mean()) if mask.any() else None for d, mask in zip(deltas, sensitivity_masks)],
                "fold_intervals": [_masked_interval(d, [len(d)], mask) if mask.any() else None for d, mask in zip(deltas, sensitivity_masks)]}
        pooled_mask = np.concatenate(sensitivity_masks)
        sensitivity["pooled_metrics"] = {arm: [{"seed": seed,
            "metrics": _subset_metrics(y[pooled_mask], np.asarray(pooled[arm][seed])[pooled_mask])} for seed in SEEDS] for arm in ARMS}
        sensitivity["linear_metrics"] = {arm: _subset_metrics(y[pooled_mask], np.asarray(control_pooled[arm])[pooled_mask]) for arm in LINEAR_ARMS}
        sensitivity["constant_metrics"] = _subset_metrics(y[pooled_mask], np.asarray(constant_logits)[pooled_mask])
    else:
        sensitivity["status"] = "unavailable; no eligible complete three-day history with explicit flags"
    result = {"per_fold": per_fold, "pooled": {arm: [{"seed": seed, "metrics": metrics(y, np.asarray(pooled[arm][seed]))} for seed in SEEDS] for arm in ARMS},
              "paired_comparisons": contrasts, "unique_evaluation_dates": len(y),
              "constant_pooled_metrics": metrics(y, np.asarray(constant_logits)),
              "constant_direction_references": directions,
              "linear_pooled_metrics": {arm: metrics(y, np.asarray(control_pooled[arm])) for arm in LINEAR_ARMS},
              "complete_history_sensitivity": sensitivity,
              "scope": "2023 DEVELOPMENT rolling evidence, raw seed models; no ensemble, hyperparameter search or untouched test"}
    _write(out / "analysis.json", result)
    return result

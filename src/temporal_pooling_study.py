"""Bounded frozen-feature news lookback/decay study; no neural training.

News slots are the three actual SPY sessions preceding the forecast, retaining
the legacy one-session delay. All arms share current prices and all-slot coverage.
This retrospective control tests predictive weighting, not information diffusion.
"""
from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from src import encoder_comparison as ec
from src.data.examples import select_headlines
from src.ml_study_data import StudyCohort, chronological_plan
from src.ml_training_study import conditional_interval
from src.pipeline_diagnosis import fit_linear, linear_predict, solver_gate, write_predictions
from src.training_data import PreparedDataset

ARMS = ["prices", "prices_coverage", "latest", "uniform", "decay"]
PENALTIES = [10., 1., .1, .01]
AGES = np.array([2., 1., 0.])
SOURCE_FILES = [
    "src/temporal_pooling_study.py", "scripts/15_test_temporal_pooling.py",
    "tests/test_temporal_pooling_study.py", "src/ml_study_data.py",
    "src/ml_training_study.py", "src/research_training.py",
    "src/pipeline_diagnosis.py", "src/encoder_comparison.py",
    "src/training_data.py", "src/features/cache.py", "src/data/prices.py",
    "src/data/examples.py", "src/data/nifty.py", "src/evaluation.py",
    "data/processed/train.jsonl", "data/processed/valid.jsonl",
    "data/raw/nifty/train.jsonl", "data/raw/nifty/valid.jsonl",
    "data/nifty_manifest.json", "data/raw/spy/daily.json", "data/spy_manifest.json",
    "artifacts/features/manifest.json", "artifacts/features/headline_index.json",
    "artifacts/features/text.npy", "artifacts/features/sentiment.npy",
    "artifacts/features/market_description.npy",
]


def temporal_weights(available, mode):
    """Return row-wise weights; latest always means the fixed final slot."""
    available = np.asarray(available, dtype=bool)
    if available.ndim != 2 or available.shape[1] != 3:
        raise ValueError("Availability must have three chronological slots")
    if mode == "latest":
        weights = np.zeros(available.shape, dtype=np.float64)
        weights[:, 2] = available[:, 2]
        return weights
    if mode not in ("uniform", "decay"):
        raise ValueError("Unknown temporal weighting")
    base = np.ones(3) if mode == "uniform" else np.exp2(-AGES)
    weights = available * base
    mass = weights.sum(axis=1, keepdims=True)
    return np.divide(weights, mass, out=np.zeros_like(weights), where=mass > 0)


def pool_features(daily, available, mode):
    daily = np.asarray(daily, dtype=np.float64)
    if daily.ndim != 3 or daily.shape[:2] != np.shape(available) or not np.isfinite(daily).all():
        raise ValueError("Daily features must be finite and align with availability")
    return np.einsum("nd,ndf->nf", temporal_weights(available, mode), daily)


def design(arrays, arm):
    if arm == "prices":
        return arrays["prices"]
    common = np.concatenate([arrays["prices"], arrays["coverage"]], axis=1)
    if arm == "prices_coverage":
        return common
    if arm not in ("latest", "uniform", "decay"):
        raise ValueError("Unknown study arm")
    content = np.concatenate([arrays["daily_text"], arrays["daily_sentiment"]], axis=2)
    return np.concatenate([common, pool_features(content, arrays["available"], arm)], axis=1)


def _identity(root, out):
    paths = {name: ec.sha(root / name) for name in SOURCE_FILES}
    protocol = out / "PROTOCOL.md"
    if not protocol.exists():
        raise ValueError("Root must write PROTOCOL.md before preparation")
    return {"source_sha256": paths, "protocol_sha256": ec.sha(protocol),
            "environment": {"python": sys.version, "python_executable": sys.executable,
                            "torch": str(torch.__version__), "numpy": np.__version__,
                            "platform": platform.platform(), "processor": platform.processor(),
                            "available_logical_cpus": os.cpu_count(), "cpu_threads": 4,
                            "device": "cpu", "cuda_used": False},
            "scope": "Train/valid only; legacy cache signature also checks historical test inputs"}


def _assert_frozen(root, out):
    expected = ec.read(out / "source_identity.json")
    if _identity(root, out) != expected:
        raise ValueError("Frozen source/protocol identity changed; use a new namespace")
    prepared = ec.read(out / "prepared_manifest.json")
    for name, digest in prepared["artifact_sha256"].items():
        if ec.sha(out / name) != digest:
            raise ValueError(f"Prepared evidence changed: {name}")


def prepare(root, out):
    """Freeze features, exact date exclusions and original/intersected split plans."""
    root, out = Path(root), Path(out)
    torch.set_num_threads(4)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "source_identity.json").exists() or (out / "features.npz").exists():
        raise ValueError("Preparation/partial preparation retained; use a new namespace")
    identity = _identity(root, out)
    ec.write(out / "source_identity.json", identity)
    snapshots = out / "source_snapshot"
    snapshots.mkdir()
    for name in SOURCE_FILES:
        if name.endswith(".py"):
            (snapshots / name.replace("/", "__")).write_bytes((root / name).read_bytes())
    cohort = StudyCohort(root)
    cache = PreparedDataset(root, "train")
    sources = {
        split: {r["date"]: r for r in map(json.loads, (root / f"data/raw/nifty/{split}.jsonl").read_text(encoding="utf-8").splitlines())}
        for split in ("train", "valid")
    }
    positions = {day: index for index, day in enumerate(cohort.sessions)}
    records, exclusions = [], []
    prices, coverage, text, sentiment, available = [], [], [], [], []
    for original_index, row in enumerate(cohort.rows):
        p = positions[row["forecast_date"]]
        if p < 3:
            raise ValueError("Insufficient market calendar for lookback")
        dates = cohort.sessions[p - 3:p]
        source = sources[row["split"]]
        if dates[0] < min(source) or dates[-1] > max(source):
            exclusions.append({"original_index": original_index, "id": row["id"], "news_dates": dates,
                               "reason": "three-slot warmup would cross raw source split boundary"})
            continue
        dt = np.zeros((3, 384), dtype=np.float64)
        ds = np.zeros((3, 3), dtype=np.float64)
        present, cov, ids = [], [], []
        for slot, day in enumerate(dates):
            if day not in source:
                present.append(False); cov.extend([0., 0., 0.]); ids.append([])
                continue
            heads = select_headlines(source[day]["news"])
            selected = [cache.index[h["id"]] for h in heads]
            dt[slot] = np.asarray(cache.text[selected], dtype=np.float64).mean(axis=0)
            ds[slot] = np.asarray(cache.sentiment[selected], dtype=np.float64).mean(axis=0)
            count = len({h.strip() for h in source[day]["news"].splitlines() if h.strip()})
            present.append(True)
            cov.extend([1., np.log1p(count), len(heads) / count])
            ids.append([h["id"] for h in heads])
        if ids[2] != [h["id"] for h in row["headlines"]] or dates[2] != row["news_date"]:
            raise ValueError("Latest slot differs from original one-day inputs")
        count_latest = len(row["headlines"])
        expected_text = cohort._vectors[original_index, :count_latest].astype(np.float64).mean(axis=0)
        expected_sentiment = cohort._scores[original_index, :count_latest].astype(np.float64).mean(axis=0)
        if not np.array_equal(dt[2], expected_text) or not np.array_equal(ds[2], expected_sentiment):
            raise ValueError("Latest daily means differ from independently aligned cohort")
        record = {k: row[k] for k in ("id", "split", "news_date", "forecast_date", "target_date", "target_up")}
        record.update(original_index=original_index, news_dates=dates, ages=AGES.tolist(), headline_ids=ids,
                      available=present, market_features_raw=cohort.raw_market[original_index].tolist())
        records.append(record); prices.append(cohort.raw_market[original_index]); coverage.append(cov)
        text.append(dt); sentiment.append(ds); available.append(present)
    counts = {split: sum(r["split"] == split for r in records) for split in ("train", "valid")}
    if counts != {"train": 1473, "valid": 313}:
        raise ValueError(f"Prespecified warmup cohort differs: {counts}")
    arrays = {"prices": np.asarray(prices), "coverage": np.asarray(coverage),
              "daily_text": np.asarray(text), "daily_sentiment": np.asarray(sentiment),
              "available": np.asarray(available, dtype=bool)}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite prepared features")
    np.savez_compressed(out / "features.npz", **arrays)
    (out / "cohort.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    original_plans = chronological_plan(cohort.rows, cohort.sessions)
    index_map = {r["original_index"]: i for i, r in enumerate(records)}
    plans = []
    for original in original_plans:
        plan = dict(original)
        plan["original_indices"] = {k: original[k] for k in original if k.endswith("_indices")}
        plan["warmup_removed_indices"] = {}
        for key in plan["original_indices"]:
            plan[key] = [index_map[i] for i in original[key] if i in index_map]
            plan["warmup_removed_indices"][key] = [i for i in original[key] if i not in index_map]
        for key, cutoff in (("outer_train_indices", plan["outer_cutoff"]), ("inner_fit_indices", plan["inner_cutoff"])):
            if any(records[i]["target_date"] > cutoff for i in plan[key]):
                raise ValueError("Target availability violated after intersection")
        plans.append(plan)
    ec.write(out / "folds.json", plans)
    manifest = {"counts": counts, "exclusions": exclusions, "models": ARMS, "penalties": PENALTIES,
                "coverage_fields": [f"age{int(age)}_{field}" for age in AGES for field in ("available", "log_unique_count", "retained_fraction")],
                "missing_days": int((~arrays["available"]).sum()),
                "availability_semantics": "Raw source date bucket with nonempty selected headlines; absent dates are empty slots. Empty present source buckets raise an error.",
                "integrity_gates": {"latest_means_match_original_cohort": True,
                                    "raw_markets_labels_calendar_match_source": True,
                                    "all_arms_share_prices_and_coverage": True},
                "weights": {name: temporal_weights(np.ones((1, 3), bool), name)[0].tolist() for name in ("latest", "uniform", "decay")},
                "artifact_sha256": {name: ec.sha(out / name) for name in ("features.npz", "cohort.jsonl", "folds.json")},
                "limitation": "Three news days, one price day; finite fixed decay, no intraday availability assurance"}
    ec.write(out / "prepared_manifest.json", manifest)
    return manifest


def _load(root, out):
    _assert_frozen(root, out)
    rows = [json.loads(line) for line in (out / "cohort.jsonl").read_text(encoding="utf-8").splitlines()]
    with np.load(out / "features.npz") as source:
        arrays = {key: source[key].copy() for key in source.files}
    return rows, arrays, ec.read(out / "folds.json")


def train(root, out):
    """60 inner plus 15 outer convex fits, no outer-driven model selection."""
    root, out = Path(root), Path(out)
    torch.set_num_threads(4)
    rows, arrays, plans = _load(root, out)
    directory = out / "fits"
    if directory.exists():
        raise ValueError("Partial/completed fits retained; use a new attempt namespace")
    directory.mkdir()
    torch.set_num_threads(4)
    solver_gate(directory)
    y = np.asarray([r["target_up"] for r in rows], dtype=float)
    folds = []
    for plan in plans:
        number = plan["fold"]
        folder = directory / f"fold_{number}"
        folder.mkdir()
        tr, ev, inner_tr, inner_ev = (np.asarray(plan[k], dtype=int) for k in
                                    ("outer_train_indices", "outer_eval_indices", "inner_fit_indices", "inner_valid_indices"))
        if not 0 < y[tr].mean() < 1 or not 0 < y[inner_tr].mean() < 1:
            raise ValueError("Fitting prefix lacks both target classes")
        prior = float(y[tr].mean())
        const = np.full(len(ev), np.log(prior / (1 - prior)))
        write_predictions(folder / "constant_predictions.csv", [rows[i] for i in ev], const)
        result = {"fold": number, "prior": prior, "constant": ec.metrics(y[ev], const), "models": {}}
        for arm in ARMS:
            modeldir = folder / arm
            modeldir.mkdir()
            x = design(arrays, arm)
            candidates, best, chosen = [], float("inf"), None
            for penalty in PENALTIES:
                model = fit_linear(x[inner_tr], y[inner_tr], penalty)
                logits = linear_predict(model, x[inner_ev])
                metrics = ec.metrics(y[inner_ev], logits)
                ec.write(modeldir / f"inner_model_{penalty}.json", model)
                write_predictions(modeldir / f"inner_predictions_{penalty}.csv", [rows[i] for i in inner_ev], logits)
                candidates.append({"penalty": penalty, "metrics": metrics, "gradient_max": model["gradient_max"]})
                if metrics["log_loss"] < best:
                    best, chosen = metrics["log_loss"], penalty
            model = fit_linear(x[tr], y[tr], chosen)
            logits = linear_predict(model, x[ev])
            ec.write(modeldir / "outer_model.json", model)
            write_predictions(modeldir / "outer_predictions.csv", [rows[i] for i in ev], logits)
            summary = {"arm": arm, "fold": number, "selected_penalty": chosen,
                       "inner_candidates": candidates, "feature_count": x.shape[1],
                       "train_metrics": ec.metrics(y[tr], linear_predict(model, x[tr])),
                       "outer_metrics": ec.metrics(y[ev], logits),
                       "artifact_sha256": {p.name: ec.sha(p) for p in modeldir.iterdir() if p.is_file()}}
            ec.write(modeldir / "summary.json", summary)
            result["models"][arm] = summary
            print(f"temporal fold{number} {arm}: BCE{summary['outer_metrics']['log_loss']:.6f}, lambda{chosen}", flush=True)
        result["constant_predictions_sha256"] = ec.sha(folder / "constant_predictions.csv")
        folds.append(result)
        ec.write(directory / "partial_results.json", folds)
    results = {"folds": folds, "inner_fit_count": 60, "outer_fit_count": 15,
               "solver_gate_extra_fit": 1, "scope": "Retrospective reused train/validation periods; not independent replication"}
    ec.write(directory / "results.json", results)
    return results


def _conditional_interval(block_deltas, block_length):
    """Within-fold circular resampling; conditional on saved fits/selection."""
    return conditional_interval(np.concatenate(block_deltas), [len(delta) for delta in block_deltas], block_length)["ci95"]


def metrics_match(actual, expected):
    """Compare full saved metrics, including nested integer confusion counts."""
    if isinstance(actual, dict) or isinstance(expected, dict):
        return (isinstance(actual, dict) and isinstance(expected, dict)
                and actual.keys() == expected.keys()
                and all(metrics_match(actual[key], expected[key]) for key in actual))
    if isinstance(actual, (int, np.integer)) and isinstance(expected, (int, np.integer)):
        return actual == expected
    if isinstance(actual, (float, int, np.number)) and isinstance(expected, (float, int, np.number)):
        return bool(np.isfinite(actual) and np.isfinite(expected)
                    and np.isclose(actual, expected, atol=1e-12, rtol=1e-12))
    return actual == expected


def analyze(root, out):
    import csv
    root, out = Path(root), Path(out)
    torch.set_num_threads(4)
    rows, arrays, plans = _load(root, out)
    if (out / "analysis.json").exists():
        raise ValueError("Completed analysis cannot be overwritten")
    results = ec.read(out / "fits/results.json")
    if len(plans) != 3 or len(results["folds"]) != 3:
        raise ValueError("Prespecified three-fold evidence incomplete")
    pooled = {name: [] for name in ["constant"] + ARMS}
    losses = {name: [] for name in pooled}
    labels = []
    for plan, result in zip(plans, results["folds"]):
        if plan["fold"] != result["fold"]:
            raise ValueError("Fold result identity differs")
        ev = plan["outer_eval_indices"]
        y = np.asarray([rows[i]["target_up"] for i in ev], dtype=float)
        labels.extend(y.tolist())
        folder = out / f"fits/fold_{plan['fold']}"
        for name in pooled:
            modeldir = folder / name
            path = folder / "constant_predictions.csv" if name == "constant" else modeldir / "outer_predictions.csv"
            if name == "constant":
                if ec.sha(path) != result["constant_predictions_sha256"]:
                    raise ValueError("Constant output changed")
            else:
                for filename, digest in result["models"][name]["artifact_sha256"].items():
                    if ec.sha(modeldir / filename) != digest:
                        raise ValueError("Saved fit/prediction changed")
            with path.open(encoding="utf-8", newline="") as stream:
                predictions = list(csv.DictReader(stream))
            if len(predictions) != len(ev):
                raise ValueError("Outer prediction count differs")
            for prediction, i in zip(predictions, ev):
                if any(str(rows[i][key]) != prediction[key] for key in ("forecast_date", "target_date", "target_up")):
                    raise ValueError("Saved prediction/date/label differs")
            logits = np.asarray([float(p["logit"]) for p in predictions])
            if name != "constant":
                model = ec.read(modeldir / "outer_model.json")
                replay = linear_predict(model, design(arrays, name)[ev])
                if not np.allclose(logits, replay, atol=1e-12, rtol=1e-12):
                    raise ValueError("Outer saved linear prediction replay differs")
            expected = result["constant"] if name == "constant" else result["models"][name]["outer_metrics"]
            measured = ec.metrics(y, logits)
            if not metrics_match(measured, expected):
                raise ValueError("Saved outer metric differs")
            pooled[name].extend(logits.tolist())
            losses[name].append(np.logaddexp(0., logits) - y * logits)
    y = np.asarray(labels)
    contrasts = [("coverage_vs_prices", "prices_coverage", "prices"),
                 ("uniform_vs_latest", "uniform", "latest"),
                 ("decay_vs_uniform", "decay", "uniform"),
                 ("decay_vs_latest", "decay", "latest")]
    contrasts += [(f"{name}_vs_coverage", name, "prices_coverage") for name in ("latest", "uniform", "decay")]
    contrasts += [(f"{name}_vs_constant", name, "constant") for name in ("latest", "uniform", "decay")]
    comparisons = {}
    for title, a, b in contrasts:
        deltas = [aa - bb for aa, bb in zip(losses[a], losses[b])]
        comparisons[title] = {"bce_delta": float(np.concatenate(deltas).mean()),
                              "fold_bce_delta": [float(delta.mean()) for delta in deltas],
                              "conditional_95_ci": {str(block): _conditional_interval(deltas, block) for block in (5, 20)},
                              "sign": "Negative favors first named arm"}
    summary = {"outer_prediction_count": len(y),
               "pooled_metrics": {name: ec.metrics(y, np.asarray(logits)) for name, logits in pooled.items()},
               "comparisons": comparisons,
               "selection": {name: [fold["models"][name]["selected_penalty"] for fold in results["folds"]] for name in ARMS},
               "uncertainty_scope": "Within-fold blocks of consecutive observed rows, not market-day blocks; saved fits/selection fixed, reused history, ten exploratory comparisons",
               "artifact_sha256": {"fits/results.json": ec.sha(out / "fits/results.json")}}
    ec.write(out / "analysis.json", summary)
    return summary


def execute(root, out, stage):
    if stage not in ("prepare", "train", "analyze"):
        raise ValueError("Unknown temporal study stage")
    return {"prepare": prepare, "train": train, "analyze": analyze}[stage](Path(root), Path(out))

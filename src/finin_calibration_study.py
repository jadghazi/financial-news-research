"""Exploratory DEVELOPMENT-only intercept recalibration of saved trajectories.

No neural training/inference, October--December test rows or final predictions.
Chronological 24/16/24 validation-date blocks are repartitioned retrospectively;
strict label-availability guards apply at selection/calibration boundaries.
This cannot turn an inspected development period into independent confirmation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.encoder_comparison import metrics
from src.reproduction_v2_data import TICKER_DESCRIPTIONS, sha256_file
from src.reproduction_v2_probe import bootstrap_difference, partition, read_rows, write


TICKERS = list(TICKER_DESCRIPTIONS)
CANDIDATES = ["full_raw", "mean_raw", "full_pca32", "mean_pca32", "matched_no_news"]
SEEDS = [17, 42, 73]


def chronological_blocks(rows: list[dict]) -> dict[str, np.ndarray]:
    """Partition exactly 64 development dates; purge by actual cutoff guards."""
    dates = sorted({row["target_date"] for row in rows})
    if len(dates) != 64 or any(row["split"] != "valid" or row["target_date"] >= "2015-10-01" for row in rows):
        raise ValueError("Require exactly 64 development validation dates, no test rows")
    selection_dates, calibration_dates, evaluation_dates = set(dates[:24]), set(dates[24:40]), set(dates[40:])
    calibration_start = min(row["cutoff_utc"] for row in rows if row["target_date"] in calibration_dates)
    evaluation_start = min(row["cutoff_utc"] for row in rows if row["target_date"] in evaluation_dates)
    masks = {
        "selection": [row["target_date"] in selection_dates and row["target_available_utc"] < calibration_start for row in rows],
        "calibration": [row["target_date"] in calibration_dates and row["target_available_utc"] < evaluation_start for row in rows],
        "evaluation": [row["target_date"] in evaluation_dates for row in rows],
        "selection_purged": [row["target_date"] in selection_dates and row["target_available_utc"] >= calibration_start for row in rows],
        "calibration_purged": [row["target_date"] in calibration_dates and row["target_available_utc"] >= evaluation_start for row in rows],
    }
    blocks = {name: np.flatnonzero(mask) for name, mask in masks.items()}
    if any(len(blocks[name]) == 0 for name in ("selection", "calibration", "evaluation")):
        raise ValueError("Chronological diagnostic block is empty")
    return blocks


def raw_bce(y, logits):
    y, logits = np.asarray(y, float), np.asarray(logits, float)
    return float(np.mean(np.logaddexp(0, logits) - y * logits))


def fit_offsets(logits, y, tickers, mode: str, penalty: float = .1) -> dict:
    """Fixed-slope intercept fit: mean BCE + .1/2*mean(centered stock delta²)."""
    logits, y = np.asarray(logits, float), np.asarray(y, float)
    tickers = list(tickers)
    if mode not in {"none", "global", "stock"} or logits.shape != y.shape or logits.ndim != 1 or len(logits) != len(tickers):
        raise ValueError("Invalid offset fitting geometry or mode")
    if not len(y) or not np.isfinite(logits).all() or not np.isin(y, [0., 1.]).all() or len(np.unique(y)) != 2:
        raise ValueError("Require finite logits and both binary classes")
    if any(ticker not in TICKERS for ticker in tickers) or not np.isfinite(penalty) or penalty <= 0:
        raise ValueError("Unknown ticker or invalid fixed penalty")
    if mode == "none":
        return {"mode": mode, "alpha": 0., "stock_delta": {ticker: 0. for ticker in TICKERS},
                "penalty": penalty, "objective": raw_bce(y, logits), "gradient_max": None, "iterations": 0, "converged": True}
    values, labels = torch.tensor(logits, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    indices = torch.tensor([TICKERS.index(ticker) for ticker in tickers])
    alpha = torch.nn.Parameter(torch.zeros(1, dtype=torch.float64))
    raw_delta = torch.nn.Parameter(torch.zeros(len(TICKERS), dtype=torch.float64))
    parameters = [alpha, raw_delta] if mode == "stock" else [alpha]
    optimizer = torch.optim.LBFGS(parameters, max_iter=500, tolerance_grad=1e-10,
                                 tolerance_change=1e-14, line_search_fn="strong_wolfe")
    calls = 0
    def closure():
        nonlocal calls
        calls += 1; optimizer.zero_grad(set_to_none=True)
        delta = raw_delta - raw_delta.mean() if mode == "stock" else torch.zeros_like(raw_delta)
        predicted = values + alpha + delta[indices]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(predicted, labels) + penalty / 2 * delta.square().mean()
        if not torch.isfinite(loss): raise ValueError("Nonfinite offset objective")
        loss.backward(); return loss
    optimizer.step(closure)
    objective = float(closure().detach())
    gradient = max(float(parameter.grad.abs().max()) for parameter in parameters)
    if gradient > 1e-5: raise ValueError(f"Offset optimizer failed convergence gate: {gradient}")
    delta = raw_delta.detach().numpy(); delta -= delta.mean()
    return {"mode": mode, "alpha": float(alpha.detach()[0]), "stock_delta": dict(zip(TICKERS, delta.tolist())),
            "penalty": penalty, "objective": objective, "gradient_max": gradient, "closure_calls": calls,
            "iterations": int(optimizer.state[alpha].get("n_iter", 0)), "converged": True,
            "fitting_count": len(y), "fitting_up": int(y.sum())}


def apply_offsets(logits, tickers, fitted):
    return np.asarray(logits, float) + fitted["alpha"] + np.array([fitted["stock_delta"][ticker] for ticker in tickers])


def evaluated(rows, indices, logits):
    subset = [rows[i] for i in indices]; y = np.array([r["target_up"] for r in subset])
    result = metrics(y, logits)
    per_ticker = {}
    for ticker in TICKERS:
        chosen = np.array([r["ticker"] == ticker for r in subset])
        if len(np.unique(y[chosen])) == 2:
            per_ticker[ticker] = metrics(y[chosen], np.asarray(logits)[chosen])
    result["by_ticker"] = per_ticker
    result["mean_within_ticker_auc"] = float(np.mean([values["auc"] for values in per_ticker.values()])) if per_ticker else None
    return result


def _predictions(path, rows, indices, raw, changed):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "ticker", "target_date", "cutoff_utc", "target_up", "raw_logit", "offset_logit"])
        writer.writeheader()
        for i, before, after in zip(indices, raw, changed):
            writer.writerow({key: rows[i][key] for key in ("id", "ticker", "target_date", "cutoff_utc", "target_up")} |
                            {"raw_logit": float(before), "offset_logit": float(after)})


def run(source: Path, out: Path) -> dict:
    """Execute bounded scalar fits using only saved development trajectories."""
    source, out = Path(source), Path(out)
    if out.exists(): raise FileExistsError("Preserve previous calibration diagnostic")
    names = [name for name in CANDIDATES] + ["full_raw_fixed30", "matched_no_news_fixed30", "zero_logit_reference"]
    config = {"scope": "Retrospective previously inspected DEVELOPMENT dates; no independent confirmation, no final/test access",
              "schedules": ["post_close", "pre_open"], "candidates": names, "seeds": SEEDS,
              "partition": "first24 dates epoch selection, next16 calibration, last24 evaluation; strict label availability purges boundaries",
              "epoch_selection": "per seed, minimum raw BCE in first eligible selection block; earliest exact tie; epoch30 sensitivities specified",
              "ensemble": "arithmetic mean of three seed logits", "offset_modes": ["none", "global", "stock"],
              "offset_model": "sigmoid(z+alpha+delta_stock); slope1, sum(delta)=0; mean BCE+.1/2*mean(delta²), alpha unpenalized",
              "optimizer": "float64 LBFGS max500, convergence maximum gradient<=1e-5", "threads": 2,
              "reference": "zero logits calibrated by same data/objective; global or stock prevalence offset",
              "uncertainty": "paired circular five-date blocks, 2000 resamples; 24 evaluation dates, exploratory, no search correction",
              "limitations": ["Repartitioned dates were previously inspected", "Checkpoints come from fixed training trajectories; no retraining",
                              "Only intercept shifts tested; same-stock ranking must remain invariant", "24 evaluation dates provide limited uncertainty information"],
              "code_sha256": sha256_file(Path(__file__)),
              "dependencies": {name: sha256_file(Path(name)) for name in ("src/reproduction_v2_probe.py", "src/reproduction_v2_data.py", "src/encoder_comparison.py", "src/evaluation.py")},
              "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    inputs = {}
    for schedule in config["schedules"]:
        for name in ("preprocessing.json",):
            path = source / "neural_roberta" / schedule / name; inputs[str(path)] = sha256_file(path)
        path = source / f"stocknet_dev/devsamples_{schedule}.jsonl"; inputs[str(path)] = sha256_file(path)
        for candidate in CANDIDATES:
            for seed in SEEDS:
                run_folder = (source / f"matched_no_news/{schedule}/no_news_{seed}" if candidate == "matched_no_news"
                              else source / f"neural_roberta/{schedule}/{candidate}_{seed}")
                summary = json.loads((run_folder / "summary.json").read_text())
                for filename in ("epoch_predictions.npz", "summary.json"):
                    path = run_folder / filename; inputs[str(path)] = sha256_file(path)
                if inputs[str(run_folder / "epoch_predictions.npz")] != summary["artifact_sha256"]["epoch_predictions.npz"]:
                    raise ValueError("Saved development trajectory differs from source run")
    config["saved_source_sha256"] = inputs
    out.mkdir(parents=True)
    write(out / "protocol_before_fits.json", config)
    (out / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    torch.set_num_threads(2); started = time.monotonic(); all_results = {}
    for schedule in config["schedules"]:
        full_rows = read_rows(source, schedule)
        _, valid, _, _ = partition(full_rows)
        preprocessing = json.loads((source / f"neural_roberta/{schedule}/preprocessing.json").read_text())
        if valid.tolist() != preprocessing["validation_indices"]: raise ValueError("Saved prediction indices differ from data chronology")
        rows = [full_rows[i] for i in valid]; blocks = chronological_blocks(rows)
        y = np.array([r["target_up"] for r in rows]); tickers = [r["ticker"] for r in rows]
        calibration, evaluation = blocks["calibration"], blocks["evaluation"]
        folder = out / schedule; folder.mkdir()
        write(folder / "partitions.json", {name: {"indices": indices.tolist(), "ids": [rows[i]["id"] for i in indices],
                                                  "dates": sorted({rows[i]["target_date"] for i in indices})} for name, indices in blocks.items()})
        trajectories, epoch_selection, seed_logits = {}, {}, {}
        for candidate in CANDIDATES:
            predictions = []; sensitivity = []; selected = {}
            for seed in SEEDS:
                run_folder = (source / f"matched_no_news/{schedule}/no_news_{seed}" if candidate == "matched_no_news"
                              else source / f"neural_roberta/{schedule}/{candidate}_{seed}")
                with np.load(run_folder / "epoch_predictions.npz") as saved:
                    logits = saved["valid"].copy()  # train trajectories never needed
                if logits.shape != (30, len(rows)) or not np.isfinite(logits).all(): raise ValueError("Invalid saved validation geometry")
                selection_losses = [raw_bce(y[blocks["selection"]], epoch_logits[blocks["selection"]]) for epoch_logits in logits]
                epoch = int(np.argmin(selection_losses))
                predictions.append(logits[epoch]); sensitivity.append(logits[29])
                selected[str(seed)] = {"epoch": epoch + 1, "selection_bce": selection_losses[epoch], "all_epoch_selection_bce": selection_losses}
            trajectories[candidate] = np.mean(predictions, axis=0); epoch_selection[candidate] = selected
            seed_logits[candidate] = np.array(predictions)
            if candidate in {"full_raw", "matched_no_news"}:
                trajectories[candidate + "_fixed30"] = np.mean(sensitivity, axis=0)
                seed_logits[candidate + "_fixed30"] = np.array(sensitivity)
        trajectories["zero_logit_reference"] = np.zeros(len(rows))
        write(folder / "selected_epochs.json", epoch_selection)
        np.savez_compressed(folder / "selected_seed_logits.npz", **seed_logits)
        write(folder / "validation_order.json", [row["id"] for row in rows])
        results, scores = {}, {}
        for candidate in names:
            raw = trajectories[candidate]; result = {}; scores[candidate] = {}
            candidate_folder = folder / candidate; candidate_folder.mkdir()
            for mode in config["offset_modes"]:
                fit = fit_offsets(raw[calibration], y[calibration], [tickers[i] for i in calibration], mode)
                changed = apply_offsets(raw, tickers, fit)
                metrics_before = evaluated(rows, evaluation, raw[evaluation]); metrics_after = evaluated(rows, evaluation, changed[evaluation])
                if metrics_before["by_ticker"].keys() != metrics_after["by_ticker"].keys(): raise ValueError("Offset changes ticker evaluation population")
                for ticker in metrics_before["by_ticker"]:
                    if abs(metrics_before["by_ticker"][ticker]["auc"] - metrics_after["by_ticker"][ticker]["auc"]) > 1e-12:
                        raise ValueError("Intercept calibration changed within-ticker ranking")
                scores[candidate][mode] = changed[evaluation]
                _predictions(candidate_folder / f"{mode}_evaluation_predictions.csv", rows, evaluation, raw[evaluation], changed[evaluation])
                _predictions(candidate_folder / f"{mode}_calibration_predictions.csv", rows, calibration, raw[calibration], changed[calibration])
                result[mode] = {"offset_fit": fit, "calibration": evaluated(rows, calibration, changed[calibration]),
                                "evaluation": metrics_after, "within_stock_auc_invariant": True,
                                "minus_uncalibrated_ci": bootstrap_difference(rows, evaluation, changed[evaluation], raw[evaluation])}
            results[candidate] = result
            print(f"calibration {schedule} {candidate}: BCE none {result['none']['evaluation']['log_loss']:.4f}, global {result['global']['evaluation']['log_loss']:.4f}, stock {result['stock']['evaluation']['log_loss']:.4f}; down {result['none']['evaluation']['predicted_down']}->{result['stock']['evaluation']['predicted_down']}", flush=True)
        for candidate in names:
            for mode in config["offset_modes"]:
                reference_mode = mode if mode != "none" else "global"
                results[candidate][mode]["minus_same_calibration_prevalence_ci"] = bootstrap_difference(rows, evaluation, scores[candidate][mode], scores["zero_logit_reference"][reference_mode])
                if candidate not in {"matched_no_news", "matched_no_news_fixed30", "zero_logit_reference"}:
                    reference = "matched_no_news_fixed30" if candidate.endswith("_fixed30") else "matched_no_news"
                    results[candidate][mode]["minus_matched_no_news_ci"] = bootstrap_difference(rows, evaluation, scores[candidate][mode], scores[reference][mode])
        write(folder / "results.json", results); all_results[schedule] = results
    result = {"configuration": config, "results": all_results, "seconds": time.monotonic() - started,
              "files_sha256": {str(path.relative_to(out)): sha256_file(path) for path in out.rglob("*") if path.is_file()}}
    write(out / "results.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--source", type=Path, required=True); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); run(args.source, args.out)

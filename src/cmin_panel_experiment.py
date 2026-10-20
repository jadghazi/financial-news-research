"""Fixed-budget company-panel FININ baseline and reusable professor-task controls.

No encoder fine-tuning, outcome-selected stocks, test selection or implicit
ablation sweep. Default run is exactly 3 arms x 2 seeds x 2 development folds.

See docs/code_guide.md#tensor-assembly-and-experiment-runner for the code map, contracts, and execution boundaries.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from src.cmin_panel_data import read, write_once, rolling_plans
from src.evaluation import classification_metrics
from src.models.research_finin import ResearchFININ
from src.runtime_utils import sha256_file as _sha256_file
from src.frozen_features import ENCODERS, SENTIMENT, _cache_kind, _verify_models
from src.research_training import fixed_budget_fit, predict_logits


def sha256_file(path):
    return _sha256_file(Path(path))


def build_cache(config, directory):
    """Verify prepared identities and create/reuse pinned offline frozen features."""
    directory = Path(directory)
    prepared = read(directory / "prepared.json")
    for name, expected in prepared["files_sha256"].items():
        if sha256_file(directory/name) != expected:
            raise ValueError("Prepared data changed before encoding")
    items = read(directory / "items.json")
    if len(items) > config["maximum_feature_items"]:
        raise ValueError("Encoding item budget exceeded")
    out = directory / "features"
    out.mkdir(exist_ok=True)
    identity = {"encoder": ENCODERS["roberta"], "sentiment": SENTIMENT,
                "pooling": "last_hidden_state[:,0,:]", "l2_normalized": False,
                "max_tokens": 64, "batch_size": 16, "device": "cpu", "threads": 4,
                "sentiment_order": ["positive", "neutral", "negative"], "dtype": "float32",
                "items_sha256": sha256_file(directory/"items.json"),
                "prepared_sha256": sha256_file(directory/"prepared.json"),
                "encoder_code_sha256": sha256_file("src/frozen_features.py")}
    write_once(out / "identity.json", identity)
    write_once(out / "index.json", [key for key, _ in items])
    if (out / "manifest.json").exists():
        manifest = read(out / "manifest.json")
        for name, expected in manifest["files_sha256"].items():
            if sha256_file(out/name) != expected:
                raise ValueError("Completed frozen feature cache changed")
        for state in manifest["states"].values():
            _verify_models(state["model_provenance"])
        return {"items": len(items), "status": "verified existing cache"}
    states = {kind: _cache_kind(kind, items, out, identity, "roberta", 16, 64, False, 100)
              for kind in ("text", "sentiment")}
    manifest = {"identity": identity, "items": len(items), "states": states,
                "files_sha256": {name: sha256_file(out/name) for name in
                 ("identity.json", "index.json", "text.npy", "sentiment.npy", "text_state.json", "sentiment_state.json")}}
    write_once(out / "manifest.json", manifest)
    return {"items": len(items), "status": "complete", "benchmarks": {k: v["benchmark"] for k, v in states.items()}}


def fit_scaler(rows, train_indices, arrays, text, history):
    """Fit price/text transforms using only unique days/texts in training histories."""
    if not train_indices:
        raise ValueError("Empty scaling population")
    unique_days = set()
    for i in train_indices:
        row = rows[i]
        unique_days.update((row["stock_index"], j) for j in range(row["session_index"]-history+1, row["session_index"]+1))
    unique_days = sorted(unique_days)
    p = np.stack([arrays["prices"][s, d] for s, d in unique_days]).astype(np.float64)
    price_mean, price_std = p.mean(0), p.std(0).clip(min=1e-6)
    ids = sorted({int(i) for s, d in unique_days for i in arrays["news_indices"][s, d] if i >= 0})
    if not ids:
        raise ValueError("No training text for news comparison")
    values = np.asarray(text[ids], dtype=np.float64)
    text_mean = values.mean(0)
    text_rms = float(np.sqrt(np.mean((values-text_mean)**2)).clip(min=1e-6))
    return {"price_mean": price_mean.tolist(), "price_std": price_std.tolist(),
            "text_mean": text_mean.tolist(), "text_rms": text_rms,
            "training_day_keys": unique_days, "training_text_indices": ids,
            "maximum_training_target": max(rows[i]["target_date"] for i in train_indices)}


class PanelDataset(Dataset):
    """Lazy bounded tensors; age restrictions preserve complete price history.

    news_age_sessions counts sessions since the declared arrival, not the
    unknown exact publication. headline_age_hours is a source-date proxy.
    Zeroing a text/sentiment channel retains the mask/coverage information.
    """
    def __init__(self, rows, indices, arrays, text, sentiment, scaler, history=5,
                 news_age_sessions=None, text_available=True):
        self.rows = [rows[i] for i in indices]
        self.arrays, self.text, self.sentiment = arrays, text, sentiment
        self.history, self.news_age_sessions = history, news_age_sessions
        self.text_available = text_available
        self.pm = np.array(scaler["price_mean"], np.float32)
        self.ps = np.array(scaler["price_std"], np.float32)
        self.tm = np.array(scaler["text_mean"], np.float32)
        self.tr = scaler["text_rms"]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        si, j = row["stock_index"], row["session_index"]
        start = j-self.history+1
        ids = self.arrays["news_indices"][si, start:j+1]
        mask = ids >= 0
        if self.news_age_sessions is not None:
            lower, upper = self.news_age_sessions
            if not 0 <= lower < upper <= self.history:
                raise ValueError("Age range must be inside fixed price history")
            ages = np.arange(self.history-1, -1, -1)[:, None]
            mask = mask & (ages >= lower) & (ages < upper)
        safe_ids = ids.clip(min=0)
        headlines = (self.text[safe_ids]-self.tm)/self.tr
        headlines = np.where(mask[..., None], headlines, 0).astype(np.float32)
        if not self.text_available:
            headlines.fill(0)
        sentiment = np.where(mask[..., None], self.sentiment[safe_ids], 0).astype(np.float32)
        prices = (self.arrays["prices"][si, start:j+1]-self.pm)/self.ps
        desc = (self.text[self.arrays["descriptions"][si]]-self.tm)/self.tr
        cutoff = self.arrays["close_seconds"][j]
        available = self.arrays["available_times"][si, start:j+1]
        if (available[mask] > cutoff).any():
            raise ValueError("Unavailable news in batch")
        ages = np.where(mask, (cutoff-self.arrays["publication_proxy"][si, start:j+1])/3600, 0).astype(np.float32)
        return {"market_features": torch.from_numpy(prices.copy()),
                "market_description": torch.from_numpy(desc.copy()),
                "headline_vectors": torch.from_numpy(headlines), "sentiment": torch.from_numpy(sentiment),
                "mask": torch.from_numpy(mask.copy()), "headline_age_hours": torch.from_numpy(ages),
                "target": torch.tensor(row["target"], dtype=torch.float32)}


def make_model(config, arm, prior):
    """Construct one declared control arm; initialize its output bias from train prevalence."""
    choices = {
        "price_only": {"news_available": False},
        "mean_news": {"interaction": False, "query_weighting": False},
        "finin": {},
        "no_interaction": {"interaction": False},
        "no_query": {"query_weighting": False},
        "no_sentiment": {"sentiment_available": False},
        "sentiment_only": {},
        "news_only": {"prices_available": False},
        "no_direct_market": {"direct_market_available": False},
        "pooled_uniform": {"temporal_aggregation": "pooled"},
        "pooled_decay": {"temporal_aggregation": "pooled", "half_life_hours": config.get("half_life_hours")},
    }
    if arm not in choices or (arm == "pooled_decay" and config.get("half_life_hours") is None):
        raise ValueError("Unknown arm or undeclared decay half-life")
    args = {"text_width": 768, "width": config["width"], "attention_width": config["width"],
            "attention_heads": 1, "history_days": config["history_sessions"], "mlp_layers": 2,
            "dropout": config["dropout"], "attention_style": config["attention_style"],
            "temporal_aggregation": config["temporal_aggregation"], "initial_probability": prior}
    args.update(choices[arm])
    return ResearchFININ(**args)


def structurally_used_module_parameters(model):
    """Module upper bound; includes zero-input columns, not exact active capacity."""
    prefixes = ["text_projection", "market_numbers", "market_fusion", "predictor"]
    if model.news_available:
        prefixes += ["news_numbers", "news_fusion"]
        if model.interaction:
            prefixes += ["self_query", "self_key", "self_value", "self_output", "normalization"]
        if model.query_weighting:
            prefixes += ["market_query", "news_key"]
    return sum(p.numel() for name,p in model.named_parameters() if name.split(".")[0] in prefixes)


def score(targets, logits):
    """Score raw logits without saturating loss; AUC is undefined for a single class."""
    y, z = np.asarray(targets, dtype=float), np.asarray(logits, dtype=float)
    if not np.isfinite(z).all():
        raise ValueError("Nonfinite predictions")
    p = 1/(1+np.exp(-np.clip(z, -700, 700)))
    result = classification_metrics(y.tolist(), p.tolist())
    c = result["confusion"]
    tp, tn, fp, fn = (c[k] for k in ("true_up", "true_down", "false_up", "false_down"))
    denominator = np.sqrt(float((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
    result.update(log_loss=float(np.mean(np.logaddexp(0,z)-y*z)),
                  brier=float(np.mean((p-y)**2)), mcc=float((tp*tn-fp*fn)/denominator) if denominator else 0.,
                  prevalence=float(y.mean()), predicted_up=int((p>=.5).sum()),
                  probability_mean=float(p.mean()), probability_std=float(p.std()))
    order = np.argsort(z)
    _, first, count = np.unique(z[order], return_index=True, return_counts=True)
    positive = np.add.reduceat(y[order], first)
    negative = count-positive
    pairs = positive.sum()*negative.sum()
    result["auc"] = float(np.sum(positive*(np.cumsum(negative)-.5*negative))/pairs) if pairs else None
    return result


def linear_prices(datasets):
    """Fixed L2=1 logistic price/identity control, train-only transforms."""
    def matrix(dataset):
        x = []
        for i, row in enumerate(dataset.rows):
            identity = np.zeros(len(dataset.arrays["descriptions"]))
            identity[row["stock_index"]] = 1
            x.append(np.r_[1., dataset[i]["market_features"].numpy().ravel(), identity])
        return np.array(x, dtype=np.float64)
    x = {role: matrix(ds) for role, ds in datasets.items()}
    y = np.array([r["target"] for r in datasets["train"].rows], dtype=float)
    beta = np.zeros(x["train"].shape[1]); beta[0] = np.log(y.mean()/(1-y.mean()))
    penalty = np.ones(len(beta)); penalty[0] = 0
    def objective(b):
        z = x["train"]@b
        return float(np.sum(np.logaddexp(0,z)-y*z)+.5*np.sum(penalty*b*b))
    history = []
    for step in range(80):
        z = x["train"]@beta; p = 1/(1+np.exp(-np.clip(z,-50,50)))
        gradient = x["train"].T@(p-y)+penalty*beta
        hessian = x["train"].T@((p*(1-p))[:,None]*x["train"])+np.diag(penalty)
        delta = np.linalg.solve(hessian, gradient)
        before = objective(beta); rate = 1.
        while objective(beta-rate*delta) > before and rate > 2**-24:
            rate *= .5
        beta -= rate*delta
        history.append({"step": step+1, "objective": objective(beta), "gradient_max": float(np.abs(gradient).max())})
        if np.abs(gradient).max() < 1e-6:
            break
    return {"beta": beta.tolist(), "history": history, "converged": history[-1]["gradient_max"] < 1e-6,
            "logits": {role: values@beta for role, values in x.items()}}


def save_predictions(path, dataset, logits):
    """Save aligned company/date/target records and raw network logits, not an ensemble."""
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["ticker", "forecast_date", "target_date", "cutoff_utc", "target", "logit"])
        writer.writeheader()
        for row, z in zip(dataset.rows, logits):
            writer.writerow({k: row[k] for k in writer.fieldnames if k != "logit"} | {"logit": float(z)})


def paired_uncertainty(fold_records, first="finin", second="price_only", samples=2000):
    """Conditional-on-fits paired five-date circular blocks within each fold."""
    rng = np.random.default_rng(20261014)
    blocks, raw = [], []
    for fold in fold_records:
        y = np.array([r["target"] for r in fold["rows"]])
        def losses(arm):
            return np.mean([np.logaddexp(0,z)-y*z for z in fold["logits"][arm]], axis=0)
        diff = losses(first)-losses(second)
        dates = np.array([r["target_date"] for r in fold["rows"]])
        day_values = np.array([[diff[dates==d].sum(), int((dates==d).sum())] for d in sorted(set(dates))])
        blocks.append(day_values); raw.extend(diff.tolist())
    estimates = []
    for _ in range(samples):
        total, count = 0., 0.
        for days in blocks:
            starts = rng.integers(0,len(days),size=(len(days)+4)//5)
            indices = ((starts[:,None]+np.arange(5)) % len(days)).ravel()[:len(days)]
            summed = days[indices].sum(0); total += summed[0]; count += summed[1]
        estimates.append(total/count)
    return {"first": first, "second": second, "delta_log_loss": float(np.mean(raw)),
            "conditional_95_interval": np.quantile(estimates,[.025,.975]).tolist(),
            "method": "2,000 paired circular five-date blocks per fold; stocks retained together; mean seed losses, not ensemble",
            "scope": "conditional on fitted models; does not include training uncertainty or establish causality"}


def run(config, directory, study_out=None, protocol_path="docs/cmin_panel_protocol_2026-10-14.md"):
    """Execute only declared fits in an immutable, versioned study namespace.

    ``directory`` supplies audited cohort data/cache; ``study_out`` separates
    later studies. Exact code/protocol/data identities are retained. A cleaned
    source tree cannot masquerade as a completed historical run; use the old
    checkout for that execution. See docs/code_guide.md#saved-studies-and-historical-source.
    """
    directory = Path(directory)
    cohort_config = read(directory/"config.json")
    if study_out is None and config != cohort_config:
        raise ValueError("Frozen baseline configuration changed")
    if study_out is not None:
        # Explicit later studies reuse the audited sources/features in a new
        # output namespace. No automatic sweep or repeated source downloads.
        for key in ("source_commit","tickers","history_sessions","headlines_per_arrival_session",
                    "availability_delay_calendar_days","development_target_before"):
            if config[key] != cohort_config[key]:
                raise ValueError(f"Later study changes cohort geometry/timeline: {key}")
        if Path(study_out).resolve() == directory.resolve():
            raise ValueError("Later study must have a distinct output namespace")
    if not Path(protocol_path).is_file():
        raise ValueError("A saved study protocol is required before fitting")
    if len(config["arms"])*len(config["seeds"])*len(config["windows"]) > config["maximum_neural_fits"]:
        raise ValueError("Declared fit budget exceeded")
    prepared = read(directory/"prepared.json")
    for name, expected in prepared["files_sha256"].items():
        if sha256_file(directory/name) != expected:
            raise ValueError("Prepared data changed")
    feature_manifest = read(directory/"features"/"manifest.json")
    feature_identity = feature_manifest["identity"]
    if (feature_identity["encoder"] != ENCODERS["roberta"] or feature_identity["sentiment"] != SENTIMENT
            or feature_identity["pooling"] != "last_hidden_state[:,0,:]" or feature_identity["l2_normalized"] is not False):
        raise ValueError("Frozen encoder identity differs from declared FININ adaptation")
    for kind, expected in (("text",ENCODERS["roberta"]),("sentiment",SENTIMENT)):
        actual = feature_manifest["states"][kind]["model_provenance"]
        if actual["model"] != expected["name"] or actual["revision"] != expected["revision"]:
            raise ValueError("Actual cached model differs from its requested identity")
        _verify_models(actual)
    for name, expected in feature_manifest["files_sha256"].items():
        if sha256_file(directory/"features"/name) != expected:
            raise ValueError("Feature cache changed")
    rows = read(directory/"rows.json")
    if any(r["target_date"] >= config["development_target_before"] for r in rows):
        raise ValueError("Reserved outcomes entered development")
    arrays = dict(np.load(directory/"daily_arrays.npz"))
    text = np.load(directory/"features"/"text.npy", mmap_mode="r")
    sentiment = np.load(directory/"features"/"sentiment.npy", mmap_mode="r")
    if text.shape != (len(read(directory/"items.json")),768) or sentiment.shape != (len(text),3):
        raise ValueError("Frozen array geometry mismatch")
    plans = rolling_plans(rows, config["windows"])
    cohort_directory = directory
    directory = Path(study_out) if study_out is not None else directory
    directory.mkdir(parents=True,exist_ok=True)
    write_once(directory/"config.json",config)
    write_once(directory/"plans.json", plans)
    sources = ["src/cmin_panel_data.py", "src/cmin_panel_experiment.py", "scripts/16_cmin_panel.py",
               "src/models/research_finin.py", "src/models/paper_finin.py", "src/research_training.py",
               "src/frozen_features.py", "src/evaluation.py", "src/runtime_utils.py"]
    source_hashes = {p: sha256_file(p) for p in sources}
    identity = {"config": config, "sources": source_hashes, "cohort_directory":str(cohort_directory.resolve()),
                "prepared_sha256": sha256_file(cohort_directory/"prepared.json"),
                "feature_manifest_sha256": sha256_file(cohort_directory/"features"/"manifest.json"),
                "source_date_disagreement_audit_sha256":sha256_file(cohort_directory/"source_date_disagreement_audit.json"),
                "protocol_sha256":sha256_file(protocol_path)}
    protocol_copy = directory/"PROTOCOL.md"
    if protocol_copy.exists() and sha256_file(protocol_copy) != sha256_file(protocol_path):
        raise ValueError("Stored protocol differs")
    if not protocol_copy.exists():
        protocol_copy.write_bytes(Path(protocol_path).read_bytes())
    write_once(directory/"study_identity.json", identity)
    snapshot = directory/"source_snapshot"; snapshot.mkdir(exist_ok=True)
    for source in sources:
        dest = snapshot/source.replace("/","__")
        if dest.exists() and sha256_file(dest) != source_hashes[source]:
            raise ValueError("Study source snapshot changed")
        if not dest.exists():
            dest.write_bytes(Path(source).read_bytes())
    if (directory/"summary.json").exists():
        completion = read(directory/"completion.json")
        for name, expected in completion.items():
            if sha256_file(directory/name) != expected:
                raise ValueError("Completed study evidence changed")
        for fit in read(directory/"summary.json")["fits"]:
            folder = directory/f"fold_{fit['fold']}"/f"{fit['arm']}_{fit['seed']}"
            for name, expected in fit["files_sha256"].items():
                if sha256_file(folder/name) != expected:
                    raise ValueError("Completed fit evidence changed")
        return read(directory/"summary.json")
    torch.set_num_threads(4)
    environment = {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "device": "cpu", "threads": 4,
                   "git_revision": subprocess.run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip(),
                   "pip_freeze": subprocess.run([sys.executable,"-m","pip","freeze"],capture_output=True,text=True).stdout.splitlines()}
    write_once(directory/"environment.json", environment)
    command_path = directory/f"command_run_{time.time_ns()}.json"
    write_once(command_path, {"argv": sys.argv, "started_utc": datetime.now(timezone.utc).isoformat()})
    fold_records, fit_summaries, baselines = [], [], []
    neural_seconds = 0.
    comparisons_requested = config.get("comparison_pairs",[["finin",other] for other in ("price_only","mean_news")
                                            if "finin" in config["arms"] and other in config["arms"]])
    if any(first not in config["arms"] or second not in config["arms"] for first,second in comparisons_requested):
        raise ValueError("Comparison references an untrained arm")
    try:
        for plan in plans:
            fold = plan["fold"]; folder = directory/f"fold_{fold}"; folder.mkdir(exist_ok=True)
            scaler = fit_scaler(rows, plan["train"], arrays, text, config["history_sessions"])
            write_once(folder/"scaler.json", scaler)
            datasets = {role: PanelDataset(rows, plan[role], arrays, text, sentiment, scaler,
                        config["history_sessions"]) for role in ("train","valid","evaluation")}
            targets = {role: np.array([r["target"] for r in ds.rows],dtype=float) for role, ds in datasets.items()}
            prior = float(targets["train"].mean())
            constant = {name: np.full(len(targets["evaluation"]),z) for name,z in
                        (("training_prior",float(np.log(prior/(1-prior)))),("equal_probability",0.))}
            linear_path = folder/"linear_prices_identity.json"
            if linear_path.exists():
                linear = read(linear_path); constant["linear_prices_identity"] = np.array(linear["evaluation_logits"])
            else:
                linear = linear_prices(datasets)
                constant["linear_prices_identity"] = linear["logits"]["evaluation"]
                write_once(linear_path, {"beta":linear["beta"],"history":linear["history"],"converged":linear["converged"],
                                         "evaluation_logits": constant["linear_prices_identity"].tolist(),
                                         "evaluation": score(targets["evaluation"],constant["linear_prices_identity"])})
            baselines.append({"fold":fold,"counts":{k:len(v) for k,v in datasets.items()},
                              "metrics":{name:score(targets["evaluation"],z) for name,z in constant.items()}})
            record = {"fold":fold,"rows":datasets["evaluation"].rows,"logits":{arm:[] for arm in config["arms"]}}
            for arm in config["arms"]:
                arm_datasets = {role: PanelDataset(rows,plan[role],arrays,text,sentiment,scaler,
                                config["history_sessions"],news_age_sessions=config.get("news_age_sessions"),
                                text_available=arm != "sentiment_only") for role in datasets}
                for seed in config["seeds"]:
                    out = folder/f"{arm}_{seed}"
                    if (out/"summary.json").exists():
                        summary = read(out/"summary.json")
                        for filename, expected in summary["files_sha256"].items():
                            if sha256_file(out/filename) != expected:
                                raise ValueError("Completed fit evidence changed")
                        logits = np.load(out/"selected_logits.npz")["evaluation"]
                        neural_seconds += summary["seconds"]
                    else:
                        if out.exists():
                            raise ValueError(f"Incomplete fit retained; no automatic repeated attempt: {out}")
                        if neural_seconds >= config["maximum_total_neural_seconds"]:
                            raise RuntimeError("Neural time budget exhausted; no new fits")
                        out.mkdir()
                        write_once(out/"attempt.json",{"fold":fold,"arm":arm,"seed":seed,"started_utc":datetime.now(timezone.utc).isoformat()})
                        print(f"fit fold={fold} arm={arm} seed={seed} train={len(datasets['train'])}",flush=True)
                        started = time.monotonic()
                        def factory():
                            model = make_model(config,arm,prior)
                            torch.save(model.state_dict(),out/"initial.pt")
                            return model
                        def progress(record):
                            epoch = record["history"]["epoch"]
                            write_once(out/f"epoch_{epoch:02d}.json",{
                                "history":record["history"],"selected_epoch":record["selected_epoch"]})
                            np.savez(out/f"epoch_{epoch:02d}_logits.npz",**record["logits"])
                            torch.save(record["selected_state"],out/f"epoch_{epoch:02d}_selected.pt")
                        progress_epoch = 0
                        def budget_guard(model, _datasets):
                            nonlocal progress_epoch
                            progress_epoch += 1
                            elapsed_now = time.monotonic()-started
                            with (out/"progress.jsonl").open("a",encoding="utf-8") as stream:
                                stream.write(json.dumps({"epoch":progress_epoch,"seconds":elapsed_now})+"\n")
                            if neural_seconds+elapsed_now >= config["maximum_total_neural_seconds"]:
                                torch.save(model.state_dict(),out/"budget_stop.pt")
                                raise RuntimeError("Neural time budget reached at epoch boundary; partial fit preserved")
                        fit = fixed_budget_fit(factory,
                              {k:arm_datasets[k] for k in ("train","valid")},seed,
                              epochs=config["epochs"],batch_size=config["batch_size"],learning_rate=config["learning_rate"],
                              optimizer_name="adamw",weight_decay=config["weight_decay"],extra_diagnostic=budget_guard,
                              progress_callback=progress)
                        elapsed = time.monotonic()-started; neural_seconds += elapsed
                        logits = predict_logits(fit["model"], arm_datasets["evaluation"])
                        for name, state in (("initial",fit["initial_state"]),("selected",fit["selected_state"]),("final",fit["final_state"])):
                            if name == "initial":
                                stored = torch.load(out/"initial.pt",weights_only=True)
                                if any(not torch.equal(stored[key],value) for key,value in state.items()):
                                    raise ValueError("Initial state evidence mismatch")
                            else:
                                torch.save(state,out/(name+".pt"))
                        write_once(out/"history.json",fit["history"])
                        epoch_arrays = {f"{role}_{epoch['epoch']}":z for epoch in fit["epoch_predictions"] for role,z in epoch["logits"].items()}
                        np.savez(out/"epoch_logits.npz",**epoch_arrays)
                        np.savez(out/"selected_logits.npz",evaluation=logits,**fit["selected_logits"])
                        save_predictions(out/"predictions.csv",datasets["evaluation"],logits)
                        # Inspect an outcome-independent first evaluation row per company.
                        trace_indices = [next(i for i,r in enumerate(datasets["evaluation"].rows) if r["ticker"]==ticker)
                                         for ticker in config["tickers"] if any(r["ticker"]==ticker for r in datasets["evaluation"].rows)]
                        samples = torch.utils.data.Subset(arm_datasets["evaluation"],trace_indices)
                        batch = next(iter(DataLoader(samples,batch_size=len(trace_indices),shuffle=False)))
                        fit["model"].eval()
                        with torch.inference_mode():
                            trace = fit["model"].trace(batch)
                        np.savez(out/"trace.npz",**{k:v.numpy() for k,v in trace.items()},mask=batch["mask"].numpy(),
                                 headline_age_hours=batch["headline_age_hours"].numpy())
                        write_once(out/"trace_rows.json",[datasets["evaluation"].rows[i] for i in trace_indices])
                        system = logits if not fit["fallback_to_constant"] else np.full(len(logits),np.log(prior/(1-prior)))
                        summary = {"fold":fold,"arm":arm,"seed":seed,"selected_epoch":fit["selected_epoch"],
                                   "seconds":elapsed,"parameter_count":sum(p.numel() for p in fit["model"].parameters()),
                                   "structurally_used_module_parameters":structurally_used_module_parameters(fit["model"]),
                                   "capacity_scope":"module upper bound; includes zero-input columns; arms have different active capacity",
                                   "training_prior":prior,"constant_fallback_selected_on_validation":fit["fallback_to_constant"],
                                   "evaluation_network":score(targets["evaluation"],logits),
                                   "evaluation_selected_system":score(targets["evaluation"],system),
                                   "files_sha256":{p.name:sha256_file(p) for p in out.iterdir() if p.is_file()}}
                        write_once(out/"summary.json",summary)
                    fit_summaries.append(summary); record["logits"][arm].append(logits)
                    print(f"done fold={fold} {arm}/{seed} LL={summary['evaluation_network']['log_loss']:.4f}",flush=True)
            fold_records.append(record)
        comparisons = [paired_uncertainty(fold_records,first,second) for first,second in comparisons_requested]
        combined = {}
        y = np.array([r["target"] for f in fold_records for r in f["rows"]])
        for arm in config["arms"]:
            values = [score(y,np.concatenate([f["logits"][arm][seed_index] for f in fold_records])) for seed_index in range(len(config["seeds"]))]
            combined[arm] = {"mean_seed_log_loss":float(np.mean([v["log_loss"] for v in values])),"seed_metrics":values}
        summary = {"id":config["id"],"completed_neural_fits":len(fit_summaries),"total_neural_seconds":neural_seconds,
                   "evaluation_company_days":len(y),"evaluation_unique_dates":len(set(r["target_date"] for f in fold_records for r in f["rows"])),
                   "baselines":baselines,"arms":combined,"fits":fit_summaries,"paired_comparisons":comparisons,
                   "scope":"two fixed retrospective development periods, conditional timing; no final holdout or professor ablation sweep opened"}
        write_once(directory/"summary.json",summary)
        completed_files = ["summary.json","plans.json","study_identity.json","environment.json","PROTOCOL.md"]
        completed_files += [f"fold_{f['fold']}/{f['arm']}_{f['seed']}/summary.json" for f in fit_summaries]
        write_once(directory/"completion.json",{name:sha256_file(directory/name) for name in completed_files})
        return summary
    except Exception:
        write_once(directory/f"failure_{time.time_ns()}.json",{"traceback":traceback.format_exc(),
                  "completed_fits_this_invocation":len(fit_summaries),"neural_seconds":neural_seconds})
        raise

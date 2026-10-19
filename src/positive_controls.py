"""Fixed-budget synthetic learnability controls; no financial forecast claims.

Only ``run`` trains. Importing this module neither downloads nor modifies caches.
Original training inputs form the calibration/reference library. Statistics are
defined before synthetic generation. The newly generated synthetic holdout uses
disjoint sampled source/nuisance pools; it is not an untouched financial period.
It is inspected at initialization and after the fixed final epoch, never for
optimization or checkpoint selection.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.encoder_comparison import count_params, infer, metrics, read, sha, write
from src.models.finin import ReducedFININ
from src.training import set_seed
from src.training_data import PreparedDataset

SEEDS = [17, 42, 73]
GENERATION_SEEDS = {"train": 110013, "holdout": 220013}
CONFIG = {
    "schema": 1, "training_count": 512, "holdout_count": 256,
    "epochs": 30, "batch_size": 16, "learning_rate": .001,
    "optimizer": "torch.optim.Adam defaults", "clip_norm": 1.,
    "width": 64, "dropout": .1, "threads": 4, "device": "cpu",
    "seeds": SEEDS, "generation_seeds": GENERATION_SEEDS,
    "source_partition_seed": 330013, "negative_label_seed": 440013,
    "text_quantile_intervals": [[.1, .2], [.8, .9]],
    "text_residual_multiplier_max": .1,
    "price_feature_index": 0, "price_sentiment_quantiles": [.1, .9],
    "mask_count_range": [1, 16], "checkpoint_rule": "fixed final epoch 30",
    "positive_success": {"holdout_balanced_accuracy_min": .9, "holdout_bce_max": .35},
    "negative_control": "independently shuffled balanced labels; report without pass threshold",
    "signal_removal": "frozen epoch30 holdout evaluation; price feature0 replaced with synthetic-training mean; text/sentiment replaced with per-slot masked synthetic-training means; zero padded slots",
    "diagnostic_rng": "every infer call isolated by torch.random.fork_rng(devices=[])",
    "budget": "full price/text/sentiment x 3 seeds; mean_pool text x 3; full random-label text x seed17",
}
FIELDS = ["market_features", "market_description", "headline_vectors", "sentiment", "mask", "target"]


class SyntheticDataset(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays["target"])

    def __getitem__(self, index):
        return {k: torch.from_numpy(v[index].copy()) if v[index].ndim else torch.tensor(v[index].item(), dtype=torch.float32)
                for k, v in self.arrays.items() if k in FIELDS} | {"row_index": index}


def _identity(root):
    names = ["src/positive_controls.py", "src/encoder_comparison.py", "src/training.py", "src/training_data.py",
             "src/models/finin.py", "src/models/attention.py", "src/models/encoders.py",
             "src/features/cache.py", "src/evaluation.py", "data/processed/train.jsonl",
             "artifacts/features/manifest.json", "artifacts/features/headline_index.json",
             "artifacts/features/text.npy", "artifacts/features/sentiment.npy", "artifacts/features/market_description.npy"]
    return {"config": CONFIG, "files_sha256": {n: sha(root/n) for n in names},
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}


def _partition(indices, rng):
    permuted = rng.permutation(np.asarray(indices, dtype=np.int64))
    cut = max(1, min(len(permuted)-1, int(len(permuted)*2/3)))
    if len(permuted) < 2:
        raise ValueError("Insufficient source values for disjoint train/holdout pools")
    return {"train": permuted[:cut], "holdout": permuted[cut:]}


def _diagnostic_infer(model, dataset):
    # Even a nonshuffled DataLoader may sample an iterator base seed. Preserve
    # the optimizer/dropout RNG sequence around every diagnostic evaluation.
    with torch.random.fork_rng(devices=[]):
        return infer(model, dataset)


def _removed_channel(condition, datasets):
    arrays = {f: a.copy() for f, a in datasets["holdout"].arrays.items()}
    train = datasets["train"].arrays
    if condition == "price":
        mean = np.asarray(train["market_features"][:, 0].mean(dtype=np.float64), dtype=np.float32)
        arrays["market_features"][:, 0] = mean
        np.testing.assert_array_equal(arrays["market_features"][:, 1:], datasets["holdout"].arrays["market_features"][:, 1:])
        channel = "market_features"
    else:
        channel = "headline_vectors" if condition == "text" else "sentiment"
        mask = train["mask"][:, :, None]
        counts = mask.sum(0)
        mean = np.divide((train[channel].astype(np.float64)*mask).sum(0), counts,
                         out=np.zeros_like(train[channel][0], dtype=np.float64), where=counts>0).astype(np.float32)
        arrays[channel] = np.broadcast_to(mean, arrays[channel].shape).copy()
        arrays[channel][~arrays["mask"]] = 0
    for field in arrays:
        if field != channel:
            np.testing.assert_array_equal(arrays[field], datasets["holdout"].arrays[field])
    return SyntheticDataset(arrays), mean, channel


def _construct(root, out):
    # PreparedDataset verifies existing content and generating signatures. Its
    # signature check reads all legacy split identities; statistics below use
    # exclusively training features, never their market labels.
    ds = PreparedDataset(root, "train")
    indices = np.asarray(sorted({ds.index[h["id"]] for r in ds.rows for h in r["headlines"]}), dtype=np.int64)
    text = np.asarray(ds.text[indices], dtype=np.float64)
    sentiment = np.asarray(ds.sentiment[indices], dtype=np.float64)
    prices = np.asarray([r["market_features"] for r in ds.rows], dtype=np.float64)
    if not all(np.isfinite(x).all() for x in [text, sentiment, prices]):
        raise ValueError("Nonfinite source features")
    mu = text.mean(0); centered = text-mu
    covariance = centered.T @ centered / (len(centered)-1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, -1]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    scores = centered @ direction
    endpoints = np.quantile(scores, [.1, .2, .8, .9])
    if not endpoints[1] < endpoints[2]:
        raise ValueError("Text quantile rule is not separable")
    # Envelope covers BOTH labels, so nuisance scaling is label independent.
    signal_endpoints = np.stack([mu+endpoints[0]*direction, mu+endpoints[-1]*direction])
    noise_budget = np.minimum(signal_endpoints.min(0)-text.min(0), text.max(0)-signal_endpoints.max(0))
    text_bounds_valid = not np.any(noise_budget < -1e-10)
    noise_budget = np.maximum(noise_budget, 0)
    contrast = sentiment[:, 0]-sentiment[:, 2]
    sentiment_cuts = np.quantile(contrast, [.1, .9])
    price_cuts = np.quantile(prices[:, 0], [.1, .9])
    construction_failures = {}
    if not text_bounds_valid:
        construction_failures["text"] = "Prespecified PC signal lies outside observed coordinate bounds; amend protocol separately"
    if not sentiment_cuts[0] < sentiment_cuts[1]:
        construction_failures["sentiment"] = "Prespecified sentiment rule is not separable"
    if not price_cuts[0] < price_cuts[1]:
        construction_failures["price"] = "Prespecified price rule is not separable"
    rng = np.random.default_rng(CONFIG["source_partition_seed"])
    residual_pools = _partition(np.arange(len(text)), rng)
    sentiment_pools = {c: _partition(np.flatnonzero(contrast <= sentiment_cuts[0] if c == 0 else contrast >= sentiment_cuts[1]), rng)
                       for c in [0, 1]} if "sentiment" not in construction_failures else {}
    price_pools = {c: _partition(np.flatnonzero(prices[:, 0] <= price_cuts[0] if c == 0 else prices[:, 0] >= price_cuts[1]), rng)
                   for c in [0, 1]} if "price" not in construction_failures else {}
    datasets = {}; source_disjointness = {}
    for condition in ["price", "text", "sentiment"]:
        directory = out/"datasets"/condition; directory.mkdir(parents=True)
        if condition in construction_failures:
            write(directory/"failure.json", {"condition": condition, "error": construction_failures[condition], "status": "construction_failed"})
            continue
        datasets[condition] = {}
        sampled_sources = {}; sampled_counts = {}
        for split, n in [("train", 512), ("holdout", 256)]:
            r = np.random.default_rng(GENERATION_SEEDS[split])
            labels = r.permutation(np.tile(np.array([0, 1], dtype=np.float32), n//2))
            counts = r.integers(1, 17, size=n)
            mask = np.arange(16)[None, :] < counts[:, None]
            arrays = {"market_features": np.tile(prices.mean(0), (n, 1)).astype(np.float32),
                      "market_description": np.tile(ds.description, (n, 1)),
                      "headline_vectors": np.tile(mu, (n, 16, 1)).astype(np.float32),
                      "sentiment": np.tile(sentiment.mean(0), (n, 16, 1)).astype(np.float32),
                      "mask": mask, "target": labels}
            provenance = []
            for i, c in enumerate(labels.astype(int)):
                count = counts[i]; record = {"row_index": i, "signal_class": int(c), "headline_count": int(count)}
                if condition == "text":
                    nuisance_indices = r.choice(residual_pools[split], count, replace=True)
                    residual = centered[nuisance_indices] - scores[nuisance_indices, None]*direction
                    ratio = np.divide(noise_budget, np.abs(residual), out=np.full_like(residual, np.inf), where=np.abs(residual)>0)
                    multiplier = np.minimum(.1, ratio.min(1))
                    coefficients = r.uniform(endpoints[0 if c == 0 else 2], endpoints[1 if c == 0 else 3], size=count)
                    vectors = mu + coefficients[:, None]*direction + multiplier[:, None]*residual
                    arrays["headline_vectors"][i, :count] = vectors.astype(np.float32)
                    record.update(nuisance_cache_indices=indices[nuisance_indices].tolist(), pc_coefficients=coefficients.tolist(), residual_multipliers=multiplier.tolist())
                elif condition == "sentiment":
                    drawn = r.choice(sentiment_pools[c][split], count, replace=True)
                    arrays["sentiment"][i, :count] = sentiment[drawn].astype(np.float32)
                    record["sentiment_cache_indices"] = indices[drawn].tolist()
                else:
                    drawn = int(r.choice(price_pools[c][split]))
                    arrays["market_features"][i, 0] = prices[drawn, 0]
                    record["price_training_row_index"] = drawn
                provenance.append(record)
            source_field = {"price": "price_training_row_index", "text": "nuisance_cache_indices", "sentiment": "sentiment_cache_indices"}[condition]
            sampled = [source for record in provenance for source in
                       ([record[source_field]] if condition == "price" else record[source_field])]
            sampled_sources[split] = set(sampled); sampled_counts[split] = len(sampled)
            for field in ["headline_vectors", "sentiment"]:
                arrays[field][~mask] = 0
            if condition == "text":
                projected = (arrays["headline_vectors"].astype(np.float64)-mu) @ direction
                rule_scores = (projected*mask).sum(1)/counts
                threshold = float((endpoints[1]+endpoints[2])/2)
                active = arrays["headline_vectors"][mask]
                if np.any(active < text.min(0)-1e-6) or np.any(active > text.max(0)+1e-6):
                    raise ValueError("Synthetic text coordinate left observed training bounds")
            elif condition == "sentiment":
                rule_scores = ((arrays["sentiment"][:, :, 0]-arrays["sentiment"][:, :, 2])*mask).sum(1)/counts
                threshold = float(sentiment_cuts.mean())
            else:
                rule_scores = arrays["market_features"][:, 0]; threshold = float(price_cuts.mean())
            if not np.array_equal((rule_scores > threshold).astype(np.float32), labels):
                raise ValueError("Constructed channel rule does not exactly recover labels")
            np.savez_compressed(directory/f"{split}.npz", **arrays)
            write(directory/f"{split}_sources.json", provenance)
            write(directory/f"{split}_rule.json", {"threshold": threshold, "scores": rule_scores.tolist(),
                  "oracle_accuracy": 1., "class_count": {"down": n//2, "up": n//2},
                  "headline_count_mean_by_class": {str(c): float(counts[labels == c].mean()) for c in [0, 1]}})
            datasets[condition][split] = SyntheticDataset(arrays)
        overlap = sampled_sources["train"] & sampled_sources["holdout"]
        if overlap:
            raise ValueError(f"Actual sampled source overlap in {condition}: {len(overlap)}")
        source_disjointness[condition] = {"source_identifier": source_field,
            "sampled_reference_count": sampled_counts,
            "sampled_unique_source_count": {s: len(v) for s, v in sampled_sources.items()},
            "actual_sampled_source_overlap_count": len(overlap), "verified_disjoint": True}
        write(out/"datasets"/condition/"source_disjointness.json", source_disjointness[condition])
    negative = {}; directory = out/"datasets"/"text_shuffled_labels"; directory.mkdir(parents=True)
    for k, (split, ds) in enumerate(datasets.get("text", {}).items()):
        arrays = {f: v.copy() for f, v in ds.arrays.items()}
        original = arrays["target"].copy()
        arrays["target"] = np.random.default_rng(CONFIG["negative_label_seed"]+k).permutation(original)
        np.savez_compressed(directory/f"{split}.npz", **arrays)
        write(directory/f"{split}_labels.json", {"original_signal_class": original.tolist(), "shuffled_target": arrays["target"].tolist(),
              "fraction_matching_signal": float((original == arrays["target"]).mean())})
        negative[split] = SyntheticDataset(arrays)
    if negative:
        datasets["text_shuffled_labels"] = negative
        source_disjointness["text_shuffled_labels"] = source_disjointness["text"] | {"same_sources_as": "text"}
        write(directory/"source_disjointness.json", source_disjointness["text_shuffled_labels"])
    else:
        write(directory/"failure.json", {"condition": "text_shuffled_labels", "status": "construction_failed", "error": "Text source control construction failed"})
    np.savez_compressed(out/"training_statistics.npz", text_mean=mu, text_pc1=direction, text_pc1_endpoints=endpoints,
                        covariance_eigenvalues=eigenvalues, text_min=text.min(0), text_max=text.max(0),
                        residual_coordinate_budget=noise_budget, sentiment_mean=sentiment.mean(0), sentiment_cuts=sentiment_cuts,
                        price_mean=prices.mean(0), price_cuts=price_cuts, training_cache_indices=indices)
    write(out/"dataset_manifest.json", {"source_scope": "Original training inputs form calibration/reference library; no original targets used",
          "synthetic_holdout_scope": "New generated examples with disjoint sampled source/nuisance pools, not an untouched financial period; statistics defined before synthetic generation",
          "matched_text_variants": "full and mean_pool use the exact same constructed train/holdout objects",
          "source_training_examples": len(prices), "source_unique_headlines": len(text),
          "text_construction": "train mean + label-tail PC1 coefficient + independently sampled orthogonal empirical residual; identical two-class coordinate budget",
          "construction_failures": construction_failures,
          "source_disjointness": source_disjointness,
          "fixed_channels": {"price": ["headline_vectors", "sentiment"], "text": ["market_features", "sentiment"], "sentiment": ["market_features", "headline_vectors"]},
          "files_sha256": {str(p.relative_to(out)): sha(p) for p in (out/"datasets").rglob("*") if p.is_file()}})
    return datasets


def _fit(out, condition, variant, seed, datasets):
    name = f"{condition}_{variant}_{seed}"; folder = out/"runs"/name
    folder.mkdir(parents=True)
    write(folder/"configuration.json", {"condition": condition, "variant": variant, "seed": seed, "configuration": CONFIG})
    start = time.monotonic(); history = []; order_digest = hashlib.sha256()
    try:
        set_seed(seed); model = ReducedFININ(variant, width=64, dropout=.1)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        torch.save(model.state_dict(), folder/"initial.pt")
        target = {s: d.arrays["target"] for s, d in datasets.items()}
        initial = {s: _diagnostic_infer(model, d) for s, d in datasets.items()}
        initial_metrics = {s: metrics(target[s], l) for s, l in initial.items()}
        write(folder/"initial_metrics.json", initial_metrics)
        loader = DataLoader(datasets["train"], batch_size=16, shuffle=True,
                            generator=torch.Generator().manual_seed(seed), num_workers=0)
        with (folder/"predictions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["stage", "epoch", "split", "row_index", "target", "logit", "probability_up"])
            writer.writeheader()
            def record(stage, epoch, split, logits):
                for i, (y, l) in enumerate(zip(target[split], logits)):
                    writer.writerow({"stage": stage, "epoch": epoch, "split": split, "row_index": i,
                                     "target": int(y), "logit": float(l), "probability_up": float(1/(1+np.exp(-np.clip(l, -700, 700))))})
                handle.flush()
            for split, logits in initial.items():
                record("initial", 0, split, logits)
            for epoch in range(1, 31):
                model.train(); total = 0.; n = 0
                for batch in loader:
                    order_digest.update(batch["row_index"].numpy().astype(np.int64).tobytes())
                    optimizer.zero_grad(set_to_none=True); logits, _ = model(batch)
                    loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["target"])
                    if not torch.isfinite(loss):
                        raise ValueError("Nonfinite positive-control training loss")
                    loss.backward(); norm = nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    if not torch.isfinite(norm):
                        raise ValueError("Nonfinite positive-control gradient")
                    optimizer.step(); total += float(loss.detach())*len(logits); n += len(logits)
                train_logits = _diagnostic_infer(model, datasets["train"])
                history.append({"epoch": epoch, "online_training_bce": total/n, "train": metrics(target["train"], train_logits)})
                record("epoch", epoch, "train", train_logits)
                torch.save(model.state_dict(), folder/f"epoch_{epoch:02d}.pt")
                write(folder/"history.json", history)
                if epoch % 10 == 0:
                    print(f"positive control {name} epoch {epoch}: train BCE {history[-1]['train']['log_loss']:.5f}", flush=True)
            torch.save(model.state_dict(), folder/"final.pt")
            torch.save(optimizer.state_dict(), folder/"final_optimizer.pt")
            final_logits = {s: _diagnostic_infer(model, d) for s, d in datasets.items()}
            final_metrics = {s: metrics(target[s], l) for s, l in final_logits.items()}
            for split, logits in final_logits.items():
                record("final", 30, split, logits)
        restored = ReducedFININ(variant).eval()
        restored.load_state_dict(torch.load(folder/"final.pt", weights_only=True, map_location="cpu"))
        replay = {s: _diagnostic_infer(restored, d) for s, d in datasets.items()}
        replay_errors = {}
        for split in datasets:
            np.testing.assert_allclose(replay[split], final_logits[split], atol=1e-6, rtol=1e-6)
            if not np.array_equal(replay[split] >= 0, final_logits[split] >= 0):
                raise ValueError("Final checkpoint replay changed decisions")
            replay_errors[split] = float(np.abs(replay[split]-final_logits[split]).max())
        positive = condition != "text_shuffled_labels"
        signal_removed = None
        if positive:
            removed, reference_mean, channel = _removed_channel(condition, datasets)
            np.savez_compressed(folder/"signal_removal_training_reference.npz", mean=reference_mean)
            removed_logits = _diagnostic_infer(model, removed)
            original_logits = final_logits["holdout"]
            probabilities = 1/(1+np.exp(-np.clip(original_logits, -700, 700)))
            removed_probabilities = 1/(1+np.exp(-np.clip(removed_logits, -700, 700)))
            with (folder/"signal_removed_holdout_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["epoch", "row_index", "target", "original_logit", "removed_logit", "original_probability_up", "removed_probability_up"])
                writer.writeheader()
                for i, y in enumerate(target["holdout"]):
                    writer.writerow({"epoch": 30, "row_index": i, "target": int(y), "original_logit": float(original_logits[i]),
                                     "removed_logit": float(removed_logits[i]), "original_probability_up": float(probabilities[i]),
                                     "removed_probability_up": float(removed_probabilities[i])})
            signal_removed = {"channel": channel, "epoch": 30, "metrics": metrics(target["holdout"], removed_logits),
                "reference": "synthetic-training mean, per-slot mask weighted for text/sentiment; price feature0 only",
                "decisions_changed": int(((original_logits >= 0) != (removed_logits >= 0)).sum()),
                "mean_abs_probability_change": float(np.abs(removed_probabilities-probabilities).mean()),
                "max_abs_probability_change": float(np.abs(removed_probabilities-probabilities).max()),
                "chance_reference_balanced_accuracy": .5,
                "interpretation": "Frozen final-model channel-removal diagnostic; no threshold tuning or forced chance criterion; finite nuisance draws may differ from exact chance"}
            write(folder/"signal_removal.json", signal_removed)
        success = final_metrics["holdout"]["balanced_accuracy"] >= .9 and final_metrics["holdout"]["log_loss"] <= .35 if positive else None
        summary = {"run": name, "condition": condition, "variant": variant, "seed": seed, "status": "complete",
                   "initial": initial_metrics, "final": final_metrics, "positive_success": success,
                   "checkpoint_rule": "epoch 30; no holdout selection", "epochs_completed": 30,
                   "checkpoint_replay_max_logit_error": replay_errors, "minibatch_order_sha256": order_digest.hexdigest(),
                   "parameter_counts": count_params(model), "elapsed_seconds": time.monotonic()-start,
                   "interpretation": "Synthetic rule learnability only; no financial forecasting evidence"}
        if signal_removed is not None:
            summary["signal_removed_holdout"] = signal_removed
        if not positive:
            count = final_metrics["holdout"]["count"]
            predicted_up = count-final_metrics["holdout"]["predicted_down"]
            summary["negative_chance_reference"] = {"expected_balanced_accuracy": .5, "count": 256,
                "accuracy_standard_error_under_conditional_balanced_label_permutation": float(np.sqrt(predicted_up*(count-predicted_up)/(count*count*(count-1)))),
                "criterion": "descriptive, no forced pass/fail threshold; a chance departure needs investigation"}
        if positive and not success:
            summary["next_diagnostic"] = "Preserve this attempt. Inspect train optimization and input sensitivity; any longer budget or signal change requires a separately declared namespace."
        summary["artifact_sha256"] = {p.name: sha(p) for p in folder.iterdir() if p.is_file() and p.name != "summary.json"}
        write(folder/"summary.json", summary)
        return summary
    except Exception as error:
        failure = {"run": name, "status": "failed", "error": repr(error), "traceback": traceback.format_exc(),
                   "epochs_completed": len(history), "elapsed_seconds": time.monotonic()-start}
        write(folder/"failure.json", failure)
        return failure


def run(root: Path, out: Path) -> dict:
    """Execute only after parent protocol/preflight review, into a fresh namespace."""
    root = Path(root).resolve(); out = Path(out).resolve()
    expected = root/"artifacts"/"diagnostics"/"pipeline_diagnosis_2026-10-13"/"positive_controls"
    if out != expected:
        raise ValueError(f"Authorized positive-control output is {expected}")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Refusing to overwrite or reuse positive-control artifacts; preserve existing attempt")
    out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(4)
    identity = _identity(root)
    write(out/"execution_identity.json", identity)
    write(out/"protocol.json", CONFIG | {"scope": "Synthetic finite unseen-example channel diagnostics, not forecasting",
          "source_scope": "Original training inputs form calibration/reference library; statistics defined before synthetic generation",
          "holdout_scope": "New generated synthetic examples with disjoint sampled source/nuisance pools; not an untouched financial period",
          "holdout_use": "initial and fixed final evaluation only; no checkpoint/hyperparameter selection",
          "failure_rule": "preserve per-condition failure and continue declared independent conditions; no silent tuning"})
    try:
        datasets = _construct(root, out)
        results = []
        specifications = [(c, "full", s) for c in ["price", "text", "sentiment"] for s in SEEDS]
        specifications += [("text", "mean_pool", s) for s in SEEDS] + [("text_shuffled_labels", "full", 17)]
        for condition, variant, seed in specifications:
            if condition in datasets:
                results.append(_fit(out, condition, variant, seed, datasets[condition]))
            else:
                results.append({"run": f"{condition}_{variant}_{seed}", "condition": condition, "variant": variant,
                                "seed": seed, "status": "construction_failed", "failure_artifact": str(out/"datasets"/condition/"failure.json")})
            write(out/"partial_results.json", results)
        after = _identity(root)
        if after != identity:
            raise ValueError("Source/cache identity changed during positive controls")
        positive = [r for r in results if r.get("condition") != "text_shuffled_labels" and r["status"] == "complete"]
        result = {"scope": "Constructed-label channel learnability only", "runs": results,
                  "all_positive_conditions_passed": len(positive) == 12 and all(r["positive_success"] for r in positive),
                  "identity_preserved": True, "attempted_runs": len(results),
                  "files_sha256": {str(p.relative_to(out)): sha(p) for p in out.rglob("*") if p.is_file()}}
        write(out/"results.json", result)
        return result
    except Exception as error:
        write(out/"failure.json", {"error": repr(error), "traceback": traceback.format_exc(),
                                  "identity_preserved": _identity(root) == identity})
        raise

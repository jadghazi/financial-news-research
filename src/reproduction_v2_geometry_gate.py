"""Artificial PC1 learning on actual TRAIN-only RoBERTa feature geometry.

No financial labels are read into a model. Separate unique-message pools provide
training and unseen examples. Repeated identical news isolates encoding geometry
from relevance/distractor selection. This is a diagnostic, not a market claim.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import time

import numpy as np
import torch

from src.encoder_comparison import metrics
from src.models.paper_finin import PaperFININ
from src.reproduction_v2_data import sha256_file
from src.reproduction_v2_features import ENCODERS, FrozenFeatures, apply_pca, fit_pca
from src.reproduction_v2_gates import SyntheticNews
from src.reproduction_v2_probe import partition, read_rows, write
from src.research_training import fit_train_only, predict_logits


def _arrays(vectors, descriptions, targets):
    n = len(vectors)
    return {"headline_vectors": np.repeat(np.asarray(vectors, np.float32)[:, None], 4, axis=1),
            "market_description": np.repeat(np.asarray(descriptions, np.float32)[None], n, axis=0),
            "sentiment": np.zeros((n, 4, 3), np.float32), "market_features": np.zeros((n, 6), np.float32),
            "mask": np.ones((n, 4), bool), "target": np.asarray(targets, np.float32)}


def run(out: Path) -> dict:
    out = Path(out)
    directory = out / "geometry_gate"
    if directory.exists(): raise FileExistsError("Preserve previous geometry diagnostic")
    config = {"status": "post-result diagnostic; cannot choose financial settings from this result",
              "seeds": [17, 42, 73], "variants": ["full", "mean_pool"], "representations": ["raw", "pca32_whitened"],
              "pool_cap": 2048, "pool_split_seed": 991301, "sample_train": 512, "sample_holdout": 256,
              "rule": "Synthetic label equals PC1 > training-pool median; PCA and median fitted exclusively on disjoint training-message pool",
              "input_rule": "Four repetitions of one actual message vector; all masks true; zero prices/sentiment; constant train-pool mean description",
              "epochs": 30, "batch_size": 16, "learning_rate": .001, "optimizer": "adam", "weight_decay": 0.,
              "width": 32, "attention_width": 32, "layers": 2, "dropout": .1, "threads": 2,
              "initialization": "ordinary PyTorch, no prior initialization", "selection": "fixed final epoch; holdout never selects epochs/settings",
              "prospective_positive_threshold": .8, "negative_control": "full pca32 seed17 with independently shuffled training labels; outcomes reported",
              "source_sha256": sha256_file(Path(__file__)),
              "dependency_sha256": {name: sha256_file(Path(name)) for name in ("src/models/paper_finin.py", "src/reproduction_v2_features.py", "src/reproduction_v2_gates.py", "src/research_training.py", "src/reproduction_v2_probe.py")},
              "cache_manifest_sha256": sha256_file(out / "roberta_cache/manifest.json"),
              "train_source_sha256": {s: sha256_file(out / f"stocknet_dev/devsamples_{s}.jsonl") for s in ("post_close", "pre_open")},
              "scope": "Computational geometry learning on real frozen TRAIN-message vectors, not actual news relevance or financial prediction"}
    directory.mkdir()
    write(directory / "configuration_before_fits.json", config)
    torch.set_num_threads(2)
    with FrozenFeatures(out / "roberta_cache") as features:
        if features.manifest["identity"]["encoder"] != ENCODERS["roberta"] or features.text.shape[1] != 768:
            raise ValueError("Geometry diagnostic requires actual pinned RoBERTa768")
        eligible = set()
        for schedule in ("post_close", "pre_open"):
            rows = read_rows(out, schedule)
            train, _, _, _ = partition(rows)
            eligible.update(features.index[h["id"]] for i in train for h in rows[i]["headlines"])
        pool = np.asarray(sorted(eligible), dtype=np.int64)
        if len(pool) < 1536: raise ValueError("Too few unique eligible training-message vectors")
        pool = np.random.default_rng(config["pool_split_seed"]).permutation(pool)[:config["pool_cap"]]
        cut = len(pool) // 2
        train_pool, holdout_pool = pool[:cut], pool[cut:]
        train_ids, holdout_ids = train_pool[:512], holdout_pool[:256]
        if len(set(train_ids).intersection(holdout_ids)): raise ValueError("Message pool overlap")
        transform = fit_pca(features.text, train_pool.tolist(), 32)
        projected = apply_pca(features.text[pool], transform)
        train_projected = projected[:cut]
        scale = train_projected.std(0).clip(min=1e-8)
        median = float(np.median(train_projected[:, 0]))
        train_y, holdout_y = (train_projected[:512, 0] > median), (projected[cut:cut+256, 0] > median)
        raw_train, raw_holdout = np.asarray(features.text[train_ids]), np.asarray(features.text[holdout_ids])
        mean = np.asarray(features.text[train_pool], np.float64).mean(0)
        pca_description = apply_pca(mean[None], transform)[0] / scale
        raw_library = np.asarray(features.text[train_pool], np.float64)
        norms = np.linalg.norm(raw_library, axis=1)
        cosine_to_mean = raw_library @ mean / (norms * np.linalg.norm(mean))
        geometry = {"fit_pool_size": len(train_pool), "holdout_pool_size": len(holdout_pool),
                    "train_unique_ids": [features.ids[i] for i in train_ids], "holdout_unique_ids": [features.ids[i] for i in holdout_ids],
                    "pca_fit_ids": [features.ids[i] for i in train_pool], "pca_fit_rows_sha256": transform["fit_rows_sha256"],
                    "pc1_training_median": median, "train_up": int(train_y.sum()), "holdout_up": int(holdout_y.sum()),
                    "train_pool_norm_mean": float(norms.mean()), "train_pool_norm_std": float(norms.std()),
                    "mean_vector_norm": float(np.linalg.norm(mean)), "mean_dimension_std": float(raw_library.std(0).mean()),
                    "cosine_to_mean_mean": float(cosine_to_mean.mean()), "cosine_to_mean_min": float(cosine_to_mean.min()),
                    "centered_energy_fraction": float(np.square(raw_library - mean).sum() / np.square(raw_library).sum())}
        write(directory / "geometry.json", geometry)
        np.savez_compressed(directory / "transform.npz", mean=transform["mean"], basis=transform["basis"], scale=scale, fit_indices=train_pool, threshold=median)
        arrays = {
            "raw": {"train": _arrays(raw_train, mean, train_y), "holdout": _arrays(raw_holdout, mean, holdout_y)},
            "pca32_whitened": {"train": _arrays(train_projected[:512] / scale, pca_description, train_y),
                               "holdout": _arrays(projected[cut:cut+256] / scale, pca_description, holdout_y)},
        }
    for representation, groups in arrays.items():
        for group, values in groups.items(): np.savez_compressed(directory / f"{representation}_{group}.npz", **values)
    conditions = [(r, v, s, False) for r in config["representations"] for v in config["variants"] for s in config["seeds"]]
    conditions += [("pca32_whitened", "full", 17, True)]
    results = []; started = time.monotonic()
    for representation, variant, seed, shuffled in conditions:
        name = f"{representation}_{variant}_{seed}" + ("_shuffled_labels" if shuffled else "")
        folder = directory / name; folder.mkdir()
        train = {key: value.copy() for key, value in arrays[representation]["train"].items()}
        if shuffled:
            train["target"] = np.random.default_rng(881301).permutation(train["target"])
            np.savez_compressed(folder / "training_inputs.npz", **train)
        holdout = arrays[representation]["holdout"]
        width = train["headline_vectors"].shape[-1]
        fit = fit_train_only(lambda: PaperFININ(variant=variant, text_width=width, width=32, attention_width=32, mlp_layers=2, dropout=.1),
                             SyntheticNews(train), seed, epochs=30, batch_size=16, learning_rate=.001, optimizer_name="adam", weight_decay=0.)
        logits = predict_logits(fit["model"], SyntheticNews(holdout))
        result = {"name": name, "representation": representation, "variant": variant, "seed": seed, "shuffled_labels": shuffled,
                  "final_train": fit["history"][-1]["train"], "holdout": metrics(holdout["target"], logits)}
        result["prospective_gate_pass"] = None if shuffled else result["holdout"]["balanced_accuracy"] >= .8
        torch.save({"initial": fit["initial_state"], "final": fit["final_state"]}, folder / "checkpoints.pt")
        write(folder / "history.json", fit["history"]); write(folder / "summary.json", result)
        with (folder / "holdout_predictions.csv").open("w", newline="") as stream:
            writer = csv.writer(stream); writer.writerow(["message_id", "synthetic_target", "logit"])
            writer.writerows(zip(geometry["holdout_unique_ids"], holdout["target"].tolist(), logits.tolist()))
        results.append(result); write(directory / "partial_results.json", {"runs": results})
        print(f"geometry {name}: trainBCE {result['final_train']['log_loss']:.5f}, unseenBA {result['holdout']['balanced_accuracy']:.3f}, unseenBCE {result['holdout']['log_loss']:.5f}", flush=True)
    summary = {"runs": results, "seconds": time.monotonic() - started,
               "files_sha256": {str(path.relative_to(directory)): sha256_file(path) for path in directory.rglob("*") if path.is_file()}}
    write(directory / "results.json", summary)
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)

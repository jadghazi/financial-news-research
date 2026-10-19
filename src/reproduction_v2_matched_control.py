"""Post-result matched no-news ablation of the complete PaperFININ architecture.

Stock descriptions, market numerical/fusion encoders and predictor remain. News
is made entirely unavailable by zeroing its mask, retaining parameter geometry,
tensor shapes and dropout calls. This is not the separate PriceIdentity model.
Financial fitting runs only when ``run`` is explicitly called by the main agent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.models.paper_finin import PaperFININ
from src.reproduction_v2_data import sha256_file
from src.reproduction_v2_features import FrozenFeatures
from src.reproduction_v2_neural import Samples
from src.reproduction_v2_probe import (bootstrap_difference, evaluated, partition,
                                       prediction_file, read_rows, write)
from src.research_training import fixed_budget_fit
from src.training import set_seed


class MatchedNoNews(PaperFININ):
    """Full architecture with no available news; description/market remain."""

    @staticmethod
    def _no_news(batch):
        return {**batch, "mask": torch.zeros_like(batch["mask"], dtype=torch.bool)}

    def trace(self, batch):
        return super().trace(self._no_news(batch))

    def forward(self, batch):
        traced = self.trace(batch)
        return traced["logits"], traced["weights"]


def _factory(text_width):
    return MatchedNoNews(variant="full", text_width=text_width, width=32,
                         attention_width=32, attention_heads=1, history_days=1,
                         mlp_layers=2, dropout=.1, attention_style="plain")


def self_check() -> dict:
    """Cheap synthetic forward/gradient/RNG checks; does not train or write."""
    generator = torch.Generator().manual_seed(902)
    batch = {"headline_vectors": torch.randn(8, 4, 8, generator=generator),
             "sentiment": torch.randn(8, 4, 3, generator=generator),
             "market_description": torch.randn(8, 8, generator=generator),
             "market_features": torch.randn(8, 6, generator=generator),
             "mask": torch.ones(8, 4, dtype=torch.bool)}
    set_seed(913)
    matched = _factory(8)
    set_seed(913)
    full = PaperFININ(variant="full", text_width=8, width=32, attention_width=32,
                      attention_heads=1, history_days=1, mlp_layers=2, dropout=.1,
                      attention_style="plain")
    for name, tensor in matched.state_dict().items():
        torch.testing.assert_close(tensor, full.state_dict()[name], rtol=0, atol=0)
    original_mask = batch["mask"].clone()
    unavailable = MatchedNoNews._no_news(batch)
    matched.train(); full.train()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(901)
        a = matched(batch)
        rng_matched = torch.random.get_rng_state().clone()
        torch.manual_seed(901)
        b = full(unavailable)
        rng_full = torch.random.get_rng_state().clone()
    torch.testing.assert_close(a[0], b[0], rtol=0, atol=0)
    torch.testing.assert_close(a[1], b[1], rtol=0, atol=0)
    torch.testing.assert_close(rng_matched, rng_full, rtol=0, atol=0)
    torch.testing.assert_close(batch["mask"], original_mask, rtol=0, atol=0)
    matched.eval()
    trace = matched.trace(batch)
    if torch.count_nonzero(trace["day_news_summaries"]) or torch.count_nonzero(trace["weights"]):
        raise AssertionError("No-news control emits nonzero news summaries or weights")
    modified = {**batch, "market_features": batch["market_features"] + .5}
    price_delta = float((matched(modified)[0] - matched(batch)[0]).abs().max())
    if price_delta <= 0:
        raise AssertionError("Market path unexpectedly inert")
    changed_news = {**batch, "headline_vectors": batch["headline_vectors"] + 999,
                    "sentiment": batch["sentiment"] - 999}
    torch.testing.assert_close(matched(changed_news)[0], trace["logits"], rtol=0, atol=0)
    return {"identical_initial_geometry": True, "identical_mask_zero_forward": True,
            "identical_dropout_rng_consumption": True, "original_mask_preserved": True,
            "news_summary_exact_zero": True, "market_path_max_logit_change": price_delta,
            "news_value_changes_inert": True, "scope": "Synthetic forward checks only"}


def _saved_scores(path, rows, validation):
    with path.open(newline="", encoding="utf-8") as stream:
        predictions = list(csv.DictReader(stream))
    if [row["id"] for row in predictions] != [rows[i]["id"] for i in validation]:
        raise ValueError("Full-model predictions and control dates differ")
    if [int(row["target_up"]) for row in predictions] != [rows[i]["target_up"] for i in validation]:
        raise ValueError("Full-model predictions and control labels differ")
    return np.array([float(row["logit"]) for row in predictions])


def run(out: Path) -> dict:
    """Six fresh matched financial fits after the main RoBERTa study completes."""
    out = Path(out)
    main = out / "neural_roberta"
    if not (main / "results.json").is_file():
        raise ValueError("Complete main RoBERTa study before this declared follow-up")
    directory = out / "matched_no_news"
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Preserve existing no-news follow-up; use a separate recovery namespace")
    source_root = Path(__file__).resolve().parents[1]
    dependencies = ["src/reproduction_v2_matched_control.py", "src/reproduction_v2_neural.py",
                    "src/reproduction_v2_probe.py", "src/reproduction_v2_features.py",
                    "src/reproduction_v2_data.py", "src/research_training.py",
                    "src/models/paper_finin.py", "src/training.py", "src/encoder_comparison.py", "src/evaluation.py"]
    config = {
        "status": "post-result matched architectural control; existing development validation remains exploratory",
        "seeds": [17, 42, 73], "schedules": ["post_close", "pre_open"],
        "epochs": 30, "batch_size": 64, "learning_rate": .001, "optimizer": "adamw",
        "weight_decay": .01, "gradient_clip": 1., "width": 32, "mlp_layers": 2,
        "dropout": .1, "attention_style": "plain", "attention_width": 32,
        "initialization": "ordinary PyTorch; exact initial states matched against corresponding full_raw saved states",
        "selection": "minimum validation raw-logit BCE; retain network and constant-fallback status separately",
        "intervention": "all news masks zero; retain full encoders/fusion/predictor/dropout geometry, market inputs and stock descriptions",
        "representation": "raw frozen RoBERTa CLS vectors; original same training-only market standardizer",
        "feature_allowlist": ["market_raw", "ticker description"],
        "limitations": ["News-path parameters remain instantiated but have no predictive contribution",
                        "Removing information changes active learned capacity, as in a standard ablation",
                        "Validation-selected comparison on one reused historical period does not establish generalization"],
        "dependency_sha256": {name: sha256_file(source_root / name) for name in dependencies},
        "main_results_sha256": sha256_file(main / "results.json"),
        "main_configuration_sha256": sha256_file(main / "configuration_before_fits.json"),
        "cache_manifest_sha256": sha256_file(out / "roberta_cache/manifest.json"),
        "preprocessing_sha256": {schedule: sha256_file(main / schedule / "preprocessing.json") for schedule in ("post_close", "pre_open")},
        "dev_source_sha256": {name: sha256_file(out / "stocknet_dev" / name) for name in ("devsamples_post_close.jsonl", "devsamples_pre_open.jsonl", "provenance.json")},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "threads": 4, "device": "cpu"},
    }
    directory.mkdir(exist_ok=True)
    write(directory / "configuration_before_fits.json", config)
    write(directory / "self_check.json", self_check())
    torch.set_num_threads(4)
    start = time.monotonic()
    results = {}
    with FrozenFeatures(out / "roberta_cache") as features:
        if features.manifest["identity"]["source_sha256"] != config["dev_source_sha256"]:
            raise ValueError("Cached text sources differ from current development sources")
        for schedule in config["schedules"]:
            rows = read_rows(out, schedule)
            train, validation, _, _ = partition(rows)
            preprocessing = json.loads((main / schedule / "preprocessing.json").read_text())
            if train.tolist() != preprocessing["training_indices"] or validation.tolist() != preprocessing["validation_indices"]:
                raise ValueError("Chronological fitting partition differs from original full model")
            raw_market = np.array([row["market_raw"] for row in rows])
            mean, scale = raw_market[train].mean(0), raw_market[train].std(0)
            scale = np.where(scale < 1e-8, 1., scale)
            np.testing.assert_array_equal(mean, preprocessing["market_mean"])
            np.testing.assert_array_equal(scale, preprocessing["market_scale"])
            market = (raw_market - mean) / scale
            descriptions = {ticker: features.description(ticker) for ticker in {row["ticker"] for row in rows}}
            datasets = {"train": Samples(rows, train, features, market, descriptions),
                        "valid": Samples(rows, validation, features, market, descriptions)}
            folder = directory / schedule
            folder.mkdir()
            write(folder / "preprocessing_reference.json", preprocessing)
            runs, control_scores, full_scores = [], [], []
            for seed in config["seeds"]:
                run_directory = folder / f"no_news_{seed}"
                run_directory.mkdir()
                full_directory = main / schedule / f"full_raw_{seed}"
                full_summary = json.loads((full_directory / "summary.json").read_text())
                for filename, checksum in full_summary["artifact_sha256"].items():
                    if sha256_file(full_directory / filename) != checksum:
                        raise ValueError("Original full-model comparison artifact changed")
                full_initial = torch.load(full_directory / "initial.pt", map_location="cpu", weights_only=True)
                set_seed(seed)
                check_model = _factory(features.text.shape[1])
                if set(check_model.state_dict()) != set(full_initial):
                    raise ValueError("Full/control initialization parameter geometry differs")
                for name, tensor in check_model.state_dict().items():
                    torch.testing.assert_close(tensor, full_initial[name], rtol=0, atol=0)
                del check_model
                begun = time.monotonic()
                fit = fixed_budget_fit(lambda: _factory(features.text.shape[1]), datasets, seed,
                                       epochs=30, batch_size=64, learning_rate=.001, clip=1.,
                                       optimizer_name="adamw", weight_decay=.01)
                for name, tensor in fit["initial_state"].items():
                    torch.testing.assert_close(tensor, full_initial[name], rtol=0, atol=0)
                for kind in ("initial", "selected", "final"):
                    torch.save(fit[f"{kind}_state"], run_directory / f"{kind}.pt")
                write(run_directory / "history.json", fit["history"])
                np.savez_compressed(run_directory / "epoch_predictions.npz",
                                    train=np.array([p["logits"]["train"] for p in fit["epoch_predictions"]]),
                                    valid=np.array([p["logits"]["valid"] for p in fit["epoch_predictions"]]))
                logits = fit["selected_logits"]["valid"]
                prediction_file(run_directory / "selected_predictions.csv", rows, validation, logits)
                paired_full = _saved_scores(full_directory / "selected_predictions.csv", rows, validation)
                summary = {
                    "seed": seed, "selected_epoch": fit["selected_epoch"], "initial_state_exactly_matched": True,
                    "train": evaluated(rows, train, fit["selected_logits"]["train"]), "valid": evaluated(rows, validation, logits),
                    "last_epoch_train": fit["history"][-1]["eval"]["train"], "last_epoch_valid": fit["history"][-1]["eval"]["valid"],
                    "constant_fallback_would_be_selected": fit["fallback_to_constant"],
                    "parameter_count": sum(parameter.numel() for parameter in fit["model"].parameters()),
                    "full_minus_no_news_ci": bootstrap_difference(rows, validation, paired_full, logits),
                    "original_full_run_summary_sha256": sha256_file(full_directory / "summary.json"),
                    "seconds": time.monotonic() - begun,
                }
                summary["artifact_sha256"] = {file.name: sha256_file(file) for file in run_directory.iterdir() if file.is_file()}
                write(run_directory / "summary.json", summary)
                runs.append(summary); control_scores.append(logits); full_scores.append(paired_full)
                print(f"matched no-news {schedule} seed{seed}: BCE {summary['valid']['log_loss']:.5f}, AUC {summary['valid']['auc']:.3f}, down {summary['valid']['predicted_down']}; {summary['seconds']:.1f}s", flush=True)
                write(folder / "partial_results.json", {"runs": runs})
            average_control, average_full = np.mean(control_scores, axis=0), np.mean(full_scores, axis=0)
            prediction_file(folder / "no_news_ensemble_predictions.csv", rows, validation, average_control)
            result = {"runs": runs, "no_news_ensemble": evaluated(rows, validation, average_control),
                      "full_ensemble": evaluated(rows, validation, average_full),
                      "full_minus_no_news_ci": bootstrap_difference(rows, validation, average_full, average_control)}
            write(folder / "results.json", result)
            results[schedule] = result
    result = {"configuration": config, "results": results, "seconds": time.monotonic() - start}
    write(directory / "results.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), indent=2))
    elif args.out is not None:
        run(args.out)
    else:
        parser.error("Supply --self-check or --out")

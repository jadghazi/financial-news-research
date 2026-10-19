"""Unseen synthetic news-rule gate for PaperFININ; no financial-validity claim.

One marked message carries a signed signal among independent distractors.
Training/holdout generation is independent, and holdout is inspected only at
initialization and the declared final epoch; it never selects epochs/settings.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from src.encoder_comparison import metrics
from src.models.paper_finin import PaperFININ
from src.reproduction_v2_data import sha256_file
from src.research_training import fit_train_only, predict_logits
from src.training import set_seed


CONFIG = {
    "train_count": 512, "holdout_count": 256, "train_generation_seed": 301301,
    "holdout_generation_seed": 401401, "shuffled_label_seed": 501501,
    "seeds": [17, 42, 73], "variants": ["full", "mean_pool"],
    "epochs": 30, "batch_size": 16, "learning_rate": .001,
    "optimizer": "adam", "weight_decay": 0., "gradient_clip": 1.,
    "text_width": 16, "width": 32, "attention_width": 32,
    "attention_heads": 1, "history_days": 1, "mlp_layers": 2,
    "dropout": .1, "attention_style": "plain", "initial_probability": None,
    "slots": 4, "real_slots_range": [2, 4], "threads": 4, "device": "cpu",
    "rule": "Only the message whose coordinate0=+2 is salient; label is its coordinate1>0. Distractor coordinate0=-2 and independent coordinate1. All other channels are unrelated.",
    "positive_gate": {"final_unseen_balanced_accuracy_min": .8},
    "selection": "fixed final epoch; holdout never used for selection",
    "negative_control": "full seed17; shuffled balanced training labels, original holdout rule labels; report outcome without a forced pass threshold",
    "scope": "Artificial identifiable signal with independent distractors; computational news-learning capacity, not actual financial semantics or generalization",
}


def generate(count: int, seed: int) -> dict[str, np.ndarray]:
    if count < 2 or count % 2:
        raise ValueError("Synthetic sample count must be positive and even")
    rng = np.random.default_rng(seed)
    target = np.repeat(np.array([0., 1.], dtype=np.float32), count // 2)
    rng.shuffle(target)
    text = rng.normal(0, .3, size=(count, 4, 16)).astype(np.float32)
    mask = np.zeros((count, 4), dtype=bool)
    for i in range(count):
        actual = int(rng.integers(2, 5))
        mask[i, :actual] = True
        text[i, :actual, 0] = -2
        text[i, :actual, 1] = rng.uniform(-1.8, 1.8, actual)
        salient = int(rng.integers(actual))
        text[i, salient, 0] = 2
        text[i, salient, 1] = (2 * target[i] - 1) * rng.uniform(.8, 1.8)
        text[i, ~mask[i]] = 0
        permutation = rng.permutation(4)
        text[i], mask[i] = text[i, permutation], mask[i, permutation]
    sentiment = rng.dirichlet([2, 2, 2], size=(count, 4)).astype(np.float32)
    sentiment[~mask] = 0
    description = np.broadcast_to(np.linspace(-.2, .2, 16, dtype=np.float32), (count, 16)).copy()
    return {"headline_vectors": text, "sentiment": sentiment,
            "market_features": rng.normal(0, .3, size=(count, 6)).astype(np.float32),
            "market_description": description, "mask": mask, "target": target}


class SyntheticNews(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays

    def __len__(self):
        return len(self.arrays["target"])

    def __getitem__(self, index):
        return {key: torch.as_tensor(value[index].copy()) if value[index].ndim else torch.tensor(value[index].item(), dtype=torch.float32)
                for key, value in self.arrays.items()}


def _write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _factory(variant):
    return PaperFININ(variant=variant, text_width=16, width=32, attention_width=32,
                     attention_heads=1, history_days=1, mlp_layers=2, dropout=.1,
                     attention_style="plain", initial_probability=None)


def _gradient_probe(variant, state, dataset):
    model = _factory(variant).eval()
    model.load_state_dict(state)
    batch = {key: torch.as_tensor(value[:32].copy()) for key, value in dataset.arrays.items()}
    batch["headline_vectors"].requires_grad_()
    logits, weights = model(batch)
    torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["target"]).backward()
    gradients = {name: float(parameter.grad.abs().sum()) if parameter.grad is not None else None
                 for name, parameter in model.named_parameters()}
    return {"parameter_abs_gradient_sum": gradients,
            "real_text_input_abs_gradient_sum": float(batch["headline_vectors"].grad[batch["mask"]].abs().sum()),
            "padded_text_input_max_abs_gradient": float(batch["headline_vectors"].grad[~batch["mask"]].abs().max()),
            "padded_attention_weight_max_abs": float(weights[~batch["mask"]].abs().max()),
            "head_weight_norm": float(model.predictor[-1].weight.detach().norm())}


def run(out: Path) -> dict:
    """Run exactly six positive fits and one shuffled-label control in a new dir."""
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Synthetic gates cannot overwrite existing outcomes")
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    source_root = Path(__file__).resolve().parents[1]
    files = ["src/reproduction_v2_gates.py", "src/models/paper_finin.py", "src/research_training.py",
             "src/training.py", "src/encoder_comparison.py", "src/evaluation.py"]
    provenance = {"config": CONFIG, "source_sha256": {name: sha256_file(source_root / name) for name in files},
                  "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    _write(out / "protocol.json", provenance)  # committed before generation and fitting
    train = generate(CONFIG["train_count"], CONFIG["train_generation_seed"])
    holdout = generate(CONFIG["holdout_count"], CONFIG["holdout_generation_seed"])
    shuffled = {key: value.copy() for key, value in train.items()}
    shuffled["target"] = np.random.default_rng(CONFIG["shuffled_label_seed"]).permutation(train["target"])
    for name, arrays in (("train", train), ("holdout", holdout), ("train_shuffled_labels", shuffled)):
        np.savez_compressed(out / f"{name}.npz", **arrays)
    holdout_dataset = SyntheticNews(holdout)
    signal_removed = {key: value.copy() for key, value in holdout.items()}
    for i in range(len(signal_removed["target"])):
        salient = signal_removed["headline_vectors"][i, :, 0] == 2
        signal_removed["headline_vectors"][i, salient, 1] = 0
    np.savez_compressed(out / "holdout_signal_removed.npz", **signal_removed)
    conditions = [(variant, seed, False) for variant in CONFIG["variants"] for seed in CONFIG["seeds"]]
    conditions.append(("full", 17, True))
    summaries = []
    for variant, seed, negative in conditions:
        name = f"{variant}_{seed}" + ("_shuffled_labels" if negative else "")
        folder = out / name
        folder.mkdir()
        dataset = SyntheticNews(shuffled if negative else train)
        started = time.monotonic()
        set_seed(seed)
        initial_model = _factory(variant).eval()
        initial_holdout = predict_logits(initial_model, holdout_dataset)
        fit = fit_train_only(lambda: _factory(variant), dataset, seed, epochs=CONFIG["epochs"],
                             batch_size=CONFIG["batch_size"], learning_rate=CONFIG["learning_rate"],
                             clip=CONFIG["gradient_clip"], optimizer_name="adam", weight_decay=0.)
        del initial_model
        logits = predict_logits(fit["model"], holdout_dataset)
        removed_logits = predict_logits(fit["model"], SyntheticNews(signal_removed))
        probe = _gradient_probe(variant, fit["initial_state"], dataset)
        updates = {key: float((fit["final_state"][key] - value).abs().sum()) for key, value in fit["initial_state"].items()}
        positive_metrics = metrics(holdout["target"], logits)
        result = {"run": name, "variant": variant, "seed": seed, "shuffled_training_labels": negative,
                  "initial_holdout": metrics(holdout["target"], initial_holdout),
                  "final_train": fit["history"][-1]["train"], "final_holdout": positive_metrics,
                  "signal_removed_holdout": metrics(holdout["target"], removed_logits),
                  "positive_gate_pass": None if negative else positive_metrics["balanced_accuracy"] >= .8,
                  "initial_gradient_probe": probe, "parameter_abs_update_sum": updates,
                  "seconds": time.monotonic() - started}
        torch.save({"initial": fit["initial_state"], "final": fit["final_state"]}, folder / "checkpoints.pt")
        _write(folder / "history.json", fit["history"])
        _write(folder / "summary.json", result)
        with (folder / "holdout_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["row", "target", "initial_logit", "final_logit", "signal_removed_logit"])
            writer.writerows(zip(range(len(logits)), holdout["target"].tolist(), initial_holdout.tolist(), logits.tolist(), removed_logits.tolist()))
        summaries.append(result)
        _write(out / "progress.json", {"completed": len(summaries), "planned": len(conditions), "runs": summaries})
        print(f"{name}: unseen BA {positive_metrics['balanced_accuracy']:.3f}, BCE {positive_metrics['log_loss']:.4f}", flush=True)
    summary = {"all_positive_gates_pass": all(row["positive_gate_pass"] for row in summaries if not row["shuffled_training_labels"]),
               "positive_fits": 6, "shuffled_control_fits": 1, "runs": summaries,
               "scope": CONFIG["scope"],
               "files_sha256": {str(file.relative_to(out)): sha256_file(file) for file in sorted(out.rglob("*")) if file.is_file()}}
    _write(out / "summary.json", summary)
    return summary

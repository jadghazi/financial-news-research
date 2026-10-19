"""Transparent binary classification metrics and dated prediction records."""

from __future__ import annotations


import numpy as np


def classification_metrics(targets: list[int], probabilities: list[float]) -> dict:
    actual = np.asarray(targets, dtype=np.int64)
    score = np.asarray(probabilities, dtype=np.float64)
    if actual.shape != score.shape or len(actual) == 0:
        raise ValueError("Targets and probabilities must have matching, nonempty shapes")
    if not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
        raise ValueError("Probabilities must be finite and lie in [0,1]")
    predicted = (score >= 0.5).astype(np.int64)
    tp = int(((actual == 1) & (predicted == 1)).sum())
    tn = int(((actual == 0) & (predicted == 0)).sum())
    fp = int(((actual == 0) & (predicted == 1)).sum())
    fn = int(((actual == 1) & (predicted == 0)).sum())
    recall_up = tp / (tp + fn) if tp + fn else 0.0
    recall_down = tn / (tn + fp) if tn + fp else 0.0
    clipped = np.clip(score, 1e-7, 1 - 1e-7)
    log_loss = float(-(actual * np.log(clipped) + (1 - actual) * np.log(1 - clipped)).mean())
    return {
        "count": len(actual),
        "accuracy": float((predicted == actual).mean()),
        "balanced_accuracy": (recall_up + recall_down) / 2,
        "log_loss": log_loss,
        "up_recall": recall_up,
        "down_recall": recall_down,
        "confusion": {"true_up": tp, "true_down": tn, "false_up": fp, "false_down": fn},
    }

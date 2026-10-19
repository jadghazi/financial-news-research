"""Train on chronological splits, select by validation loss, evaluate once on test."""

from __future__ import annotations

import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.models.baselines import NumericBaseline
from src.models.finin import ReducedFININ


VARIANTS = (
    "price_only",
    "price_sentiment",
    "mean_pool",
    "no_self_attention",
    "no_sentiment",
    "full",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_model(variant: str) -> nn.Module:
    if variant == "price_only":
        return NumericBaseline(False)
    if variant == "price_sentiment":
        return NumericBaseline(True)
    return ReducedFININ(variant)


def move_batch(batch: dict, device: str) -> dict:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@torch.inference_mode()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        logits, _ = model(batch)
        total += nn.functional.binary_cross_entropy_with_logits(logits, batch["target"], reduction="sum").item()
        count += len(logits)
    return total / count


@torch.inference_mode()
def predict(model: nn.Module, loader: DataLoader, device: str) -> tuple[list[float], list[list[float]]]:
    model.eval()
    scores: list[float] = []
    attention: list[list[float]] = []
    for batch in loader:
        batch = move_batch(batch, device)
        logits, weights = model(batch)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
        if weights is not None:
            attention.extend(weights.cpu().tolist())
        else:
            attention.extend([[] for _ in range(len(logits))])
    return scores, attention


def train_one(
    variant: str,
    seed: int,
    datasets: dict,
    device: str,
    checkpoint_path,
    max_epochs: int = 30,
    patience: int = 5,
    batch_size: int = 16,
) -> tuple[nn.Module, dict]:
    set_seed(seed)
    model = make_model(variant).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(datasets["train"], batch_size=batch_size, shuffle=True, generator=generator, num_workers=0)
    valid_loader = DataLoader(datasets["valid"], batch_size=batch_size, shuffle=False, num_workers=0)
    best_loss = float("inf")
    best_epoch = 0
    history = []
    start = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        count = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(batch)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, batch["target"])
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite training loss for {variant}, seed {seed}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(logits)
            count += len(logits)
        valid_loss = evaluate_loss(model, valid_loader, device)
        record = {"epoch": epoch, "train_loss": total_loss / count, "valid_loss": valid_loss}
        history.append(record)
        print(f"{variant} seed {seed} epoch {epoch}: train {record['train_loss']:.4f}, valid {valid_loss:.4f}", flush=True)
        if valid_loss < best_loss - 1e-5:
            best_loss = valid_loss
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
        elif epoch - best_epoch >= patience:
            break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    return model, {
        "variant": variant,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "epochs_completed": len(history),
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "history": history,
    }

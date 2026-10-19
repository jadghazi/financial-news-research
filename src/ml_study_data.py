"""Frozen one-day features with split-local numerical scaling and dated fit plans.

Only train/valid examples enter the study. PreparedDataset's legacy cache check
necessarily hashes historical test inputs too; those examples never enter a fit
or evaluation here. This module does not certify publication-time availability.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.examples import market_features, select_headlines
from src.data.nifty import sha256_file
from src.data.prices import load_prices
from src.training_data import PreparedDataset


def _indices(indices, size):
    values = list(indices)
    if not values or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in values):
        raise ValueError("Indices must be a nonempty integer sequence")
    if len(set(values)) != len(values) or any(i < 0 or i >= size for i in values):
        raise ValueError("Indices must be unique and within the cohort")
    return [int(i) for i in values]


def fit_scaler(raw, indices):
    """Population moments of explicitly supplied fitting rows only."""
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Raw market features must be a finite matrix")
    selected = values[_indices(indices, len(values))]
    mean = selected.mean(axis=0)
    scale = selected.std(axis=0, ddof=0)
    scale[scale == 0] = 1.0
    return mean, scale


class StudyCohort:
    """Load and independently align train/valid prices, labels and cached news."""

    def __init__(self, root: Path):
        root = Path(root)
        cache = PreparedDataset(root, "train")
        sessions, prices = load_prices(root)
        positions = {date: index for index, date in enumerate(sessions)}
        manifest = json.loads((root / "data/nifty_manifest.json").read_text(encoding="utf-8"))
        rows = []
        for split in ("train", "valid"):
            source_path = root / f"data/raw/nifty/{split}.jsonl"
            if sha256_file(source_path) != manifest["files"][source_path.name]["sha256"]:
                raise ValueError(f"Source checksum changed: {split}")
            source_rows = [json.loads(line) for line in source_path.read_text(encoding="utf-8").splitlines()]
            source_dates = [r["date"] for r in source_rows]
            if source_dates != sorted(set(source_dates)):
                raise ValueError(f"Unordered/duplicate source dates: {split}")
            sources = {r["date"]: r for r in source_rows}
            prepared = cache.rows if split == "train" else [
                json.loads(line) for line in (root / "data/processed/valid.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            for original in prepared:
                row = copy.deepcopy(original)
                n, d, q = row["news_date"], row["forecast_date"], row["target_date"]
                p = positions.get(d)
                if p is None or not 0 < p < len(sessions) - 1 or (n, q) != (sessions[p - 1], sessions[p + 1]):
                    raise ValueError(f"Wrong adjacent-session timeline: {row['id']}")
                if row["split"] != split or row["id"] != f"{split}_{n}":
                    raise ValueError("Wrong source split/row identity")
                raw = market_features(prices[n], prices[d])
                if raw != row["market_features_raw"]:
                    raise ValueError(f"Raw numerical features differ: {row['id']}")
                if row["target_up"] != int(prices[q]["close"] > prices[d]["close"]):
                    raise ValueError(f"Label differs from price snapshot: {row['id']}")
                if row["close_at_forecast"] != prices[d]["close"] or row["close_at_target"] != prices[q]["close"]:
                    raise ValueError(f"Saved closes differ: {row['id']}")
                if n not in sources or row["headlines"] != select_headlines(sources[n]["news"]):
                    raise ValueError(f"Selected news differs from source: {row['id']}")
                for headline in row["headlines"]:
                    if hashlib.sha256(headline["text"].encode("utf-8")).hexdigest() != headline["id"]:
                        raise ValueError("Headline content/id differs")
                    if headline["id"] not in cache.index:
                        raise ValueError("Headline absent from frozen cache")
                # Never carry the historical globally standardized column into fits.
                row.pop("market_features", None)
                row["market_features_raw"] = raw
                rows.append(row)
        dates = [r["forecast_date"] for r in rows]
        if dates != sorted(set(dates)) or len({r["id"] for r in rows}) != len(rows):
            raise ValueError("Combined cohort is unordered or duplicated")
        self.rows = rows
        self.raw_market = np.asarray([r["market_features_raw"] for r in rows], dtype=np.float64)
        self.sessions = sessions
        self.description = np.array(cache.description, dtype=np.float32, copy=True)
        self._vectors = np.zeros((len(rows), 16, 384), dtype=np.float32)
        self._scores = np.zeros((len(rows), 16, 3), dtype=np.float32)
        self._masks = np.zeros((len(rows), 16), dtype=bool)
        for i, row in enumerate(rows):
            selected = [cache.index[h["id"]] for h in row["headlines"]]
            if not 0 < len(selected) <= 16:
                raise ValueError("Headline count outside fixed one-day cap")
            self._vectors[i, :len(selected)] = cache.text[selected]
            self._scores[i, :len(selected)] = cache.sentiment[selected]
            self._masks[i, :len(selected)] = True
        if not all(np.isfinite(v).all() for v in (self.raw_market, self.description, self._vectors, self._scores)):
            raise ValueError("Nonfinite cohort features")
        for value in (self.raw_market, self.description, self._vectors, self._scores, self._masks):
            value.flags.writeable = False

    def dataset(self, indices, mean, scale):
        return StudyDataset(self, indices, mean, scale)


class StudyDataset(Dataset):
    """Pre-materialized selected examples; getters cannot mutate shared features."""

    def __init__(self, cohort: StudyCohort, indices, mean, scale):
        indices = _indices(indices, len(cohort.rows))
        mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
        if mean.shape != (6,) or scale.shape != (6,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
            raise ValueError("Scaler must contain six finite centers and positive scales")
        self.indices = indices
        self.rows = [copy.deepcopy(cohort.rows[i]) for i in indices]
        self.raw_market = np.array(cohort.raw_market[indices], copy=True)
        standardized = (self.raw_market - mean) / scale
        self._samples = []
        for j, i in enumerate(indices):
            self._samples.append({
                "market_features": torch.tensor(standardized[j], dtype=torch.float32),
                "market_description": torch.tensor(cohort.description.copy(), dtype=torch.float32),
                "headline_vectors": torch.tensor(cohort._vectors[i].copy(), dtype=torch.float32),
                "sentiment": torch.tensor(cohort._scores[i].copy(), dtype=torch.float32),
                "mask": torch.tensor(cohort._masks[i].copy(), dtype=torch.bool),
                "target": torch.tensor(cohort.rows[i]["target_up"], dtype=torch.float32),
                "row_index": i,
            })

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in self._samples[index].items()}


def chronological_plan(rows, sessions, starts=(800, 1100, 1475), ends=(1100, 1475, 1790), inner_fraction=.2):
    """Expanding outer prefixes, inner suffix selection, purge by label availability.

    Every fitting target is observable by the closing session immediately before
    the first evaluation forecast. Gaps in news coverage never stand in for gaps
    in the market calendar. All outputs are JSON-serializable index manifests.
    """
    dates = [r["forecast_date"] for r in rows]
    if dates != sorted(set(dates)) or sessions != sorted(set(sessions)):
        raise ValueError("Rows and calendar must be ordered and unique")
    if not 0 < inner_fraction < 1 or len(starts) != len(ends) or not starts:
        raise ValueError("Invalid chronological plan settings")
    positions = {date: index for index, date in enumerate(sessions)}
    for row in rows:
        if row["forecast_date"] not in positions or row["target_date"] not in positions or row["target_date"] <= row["forecast_date"]:
            raise ValueError("Forecast/target outside calendar or invalid horizon")
    plans = []
    previous_end = None
    for fold, (start, end) in enumerate(zip(starts, ends)):
        if not isinstance(start, int) or not isinstance(end, int) or not 1 <= start < end <= len(rows) or (previous_end is not None and start < previous_end):
            raise ValueError("Outer blocks must be valid and nonoverlapping")
        p = positions[rows[start]["forecast_date"]]
        if p == 0:
            raise ValueError("No prior market session for cutoff")
        cutoff = sessions[p - 1]
        prefix = list(range(start))
        outer_train = [i for i in prefix if rows[i]["target_date"] <= cutoff]
        outer_purged = [i for i in prefix if rows[i]["target_date"] > cutoff]
        inner_start = int(len(outer_train) * (1 - inner_fraction))
        if not 1 <= inner_start < len(outer_train):
            raise ValueError("Insufficient observations for inner split")
        inner_valid = outer_train[inner_start:]
        inner_p = positions[rows[inner_valid[0]]["forecast_date"]]
        if inner_p == 0:
            raise ValueError("No prior market session for inner cutoff")
        inner_cutoff = sessions[inner_p - 1]
        inner_prefix = outer_train[:inner_start]
        inner_fit = [i for i in inner_prefix if rows[i]["target_date"] <= inner_cutoff]
        inner_purged = [i for i in inner_prefix if rows[i]["target_date"] > inner_cutoff]
        if not inner_fit:
            raise ValueError("Empty inner fitting prefix after purge")
        plans.append({
            "fold": fold,
            "outer_start": start, "outer_end": end,
            "outer_train_indices": outer_train, "outer_eval_indices": list(range(start, end)),
            "outer_cutoff": cutoff, "outer_purged_indices": outer_purged,
            "inner_fit_indices": inner_fit, "inner_valid_indices": inner_valid,
            "inner_cutoff": inner_cutoff, "inner_purged_indices": inner_purged,
            "purge_rule": "target_date <= market session preceding first evaluation forecast",
            "outer_first_forecast": rows[start]["forecast_date"],
            "outer_last_forecast": rows[end - 1]["forecast_date"],
            "inner_first_forecast": rows[inner_valid[0]]["forecast_date"],
        })
        previous_end = end
    return plans

"""Lazy, timestamp-aware FININ history tensors from DEVELOPMENT records.

Histories follow exact common trading sessions for the same ticker/schedule;
missing sessions never fall back to a nearby available-news day. Historical
input rows are copied through a strict allowlist excluding all outcome fields.
Only the center record supplies the target. This module does not train or infer.
Generic cohorts must explicitly declare their exclusive development target-date
boundary. The default remains the historical StockNet 2015-10-01 boundary for
existing fixtures. Declared availability is validated, not independently proven;
missing price vintage metadata retains the legacy 22:00 UTC convention and
missing message availability metadata assumes its publication timestamp.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.reproduction_v2_data import sha256_file


INPUT_ALLOWLIST = ("id", "ticker", "forecast_schedule", "price_date", "cutoff_utc",
                   "market_raw", "market_relative", "description", "headlines", "price_available_utc")
MESSAGE_ALLOWLIST = ("id", "created_at", "available_at")
LEGACY_DEVELOPMENT_TARGET_BEFORE = "2015-10-01"


def _boundary(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("Development target boundary must be an explicit ISO date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Development target boundary must be an explicit ISO date") from exc
    if parsed.isoformat() != value:
        raise ValueError("Development target boundary must use YYYY-MM-DD")
    return value


def _time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid UTC availability timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError("Availability timestamps must be explicit UTC")
    return parsed.astimezone(timezone.utc)


def _input(row: dict) -> dict:
    """Copy feature metadata only; never copy historical labels/returns."""
    selected = {key: row[key] for key in INPUT_ALLOWLIST if key not in {"headlines", "price_available_utc"}}
    selected["market_raw"] = [float(value) for value in row["market_raw"]]
    selected["market_relative"] = [float(value) for value in row["market_relative"]]
    for key in ("market_raw", "market_relative"):
        if len(selected[key]) != 6 or not np.isfinite(selected[key]).all():
            raise ValueError("History market input must contain six finite values")
    cutoff = _time(row["cutoff_utc"])
    price_available = _time(row.get("price_available_utc", row["price_date"] + "T22:00:00Z"))
    if price_available > cutoff:
        raise ValueError("Price snapshot not available by its input cutoff")
    if "price_available_utc" in row:
        selected["price_available_utc"] = row["price_available_utc"]
    selected["headlines"] = []
    for message in row["headlines"]:
        published = _time(message["created_at"])
        if published > cutoff:
            raise ValueError("Message timestamp exceeds its source availability cutoff")
        available = _time(message.get("available_at", message["created_at"]))
        if available < published or available > cutoff:
            raise ValueError("Message availability must be at or after publication and by its source cutoff")
        if row.get("news_window_start_utc") is not None and published <= _time(row["news_window_start_utc"]):
            raise ValueError("Message timestamp precedes its source window")
        selected["headlines"].append({key: message[key] for key in MESSAGE_ALLOWLIST if key in message})
    return selected


def _indices(indices, size):
    indices = list(indices)
    if not indices or len(set(indices)) != len(indices) or any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) or i < 0 or i >= size for i in indices):
        raise ValueError("Indices must be a nonempty unique integer sequence within the cohort")
    return [int(i) for i in indices]


class HistoryCohort:
    def __init__(self, rows: list[dict], common_calendar: list[str], history_days: int = 3,
                 common_cohort_max_history: int | None = None, center_indices=None,
                 development_target_before: str = LEGACY_DEVELOPMENT_TARGET_BEFORE):
        boundary = _boundary(development_target_before)
        maximum = history_days if common_cohort_max_history is None else common_cohort_max_history
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in (history_days, maximum)) or maximum < history_days:
            raise ValueError("History and common-cohort lengths must be positive integers, maximum>=history")
        if common_calendar != sorted(set(common_calendar)):
            raise ValueError("Common trading calendar must be ordered and unique")
        if any(row["split"] not in {"train", "valid"} or _boundary(row["target_date"]) >= boundary for row in rows):
            raise ValueError("Only DEVELOPMENT source records allowed")
        positions = {date: index for index, date in enumerate(common_calendar)}
        inputs = [_input(row) for row in rows]
        keys = [(row["ticker"], row["forecast_schedule"], row["price_date"]) for row in inputs]
        if len(set(keys)) != len(keys): raise ValueError("Duplicate same-ticker/schedule session inputs")
        lookup = dict(zip(keys, inputs))
        requested = range(len(rows)) if center_indices is None else _indices(center_indices, len(rows))
        self.history_days, self.common_cohort_max_history = history_days, maximum
        self.centers, self.histories, self.center_source_indices, self.excluded = [], [], [], []
        for index in requested:
            source = rows[index]; center = inputs[index]
            position = positions.get(center["price_date"])
            if position is None:
                raise ValueError("Center price date absent from common source calendar")
            if position < maximum - 1:
                self.excluded.append({"source_index": index, "id": center["id"], "reason": "insufficient_calendar_history"}); continue
            dates = common_calendar[position - maximum + 1:position + 1]
            required = [(center["ticker"], center["forecast_schedule"], day) for day in dates]
            missing = [key[2] for key in required if key not in lookup]
            if missing:
                self.excluded.append({"source_index": index, "id": center["id"], "reason": "missing_session_inputs", "missing_price_dates": missing}); continue
            history = [lookup[key] for key in required[-history_days:]]
            cutoff = _time(center["cutoff_utc"])
            if any(_time(row["cutoff_utc"]) > cutoff or row["price_date"] > center["price_date"] for row in history):
                raise ValueError("History includes a future input")
            if any(row["description"] != center["description"] for row in history):
                raise ValueError("Changing market descriptions need an explicit time-varying descriptor implementation")
            target = source["target_up"]
            if target not in (0, 1) or not np.isfinite(target): raise ValueError("Current center target must be binary")
            target_available = _time(source["target_available_utc"])
            if target_available <= cutoff: raise ValueError("Current target is available by forecast cutoff")
            self.centers.append({"id": center["id"], "ticker": center["ticker"], "split": source["split"],
                                 "forecast_schedule": center["forecast_schedule"], "price_date": center["price_date"],
                                 "cutoff_utc": center["cutoff_utc"], "target_date": source["target_date"],
                                 "target_available_utc": source["target_available_utc"], "target_up": int(target)})
            self.histories.append(history); self.center_source_indices.append(index)
        self.provenance = {"history_days": history_days, "common_cohort_max_history": maximum,
                           "development_target_before": boundary, "include_test": False,
                           "availability_limitations": "Metadata assertions do not establish historical publication or price vintages; absent message available_at uses created_at, absent price_available_utc uses price_date 22:00 UTC",
                           "feature_allowlist": list(INPUT_ALLOWLIST), "message_feature_allowlist": list(MESSAGE_ALLOWLIST),
                           "historical_outcomes_imported": False, "calendar_sha256": hashlib.sha256(json.dumps(common_calendar).encode()).hexdigest(),
                           "input_rows_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
                           "center_ids": [row["id"] for row in self.centers], "exclusions": self.excluded}

    @classmethod
    def from_directory(cls, dev_dir: Path, schedule: str, **kwargs):
        dev_dir = Path(dev_dir)
        if schedule not in {"post_close", "pre_open"}: raise ValueError("Unknown forecast schedule")
        source = dev_dir / f"devsamples_{schedule}.jsonl"
        manifest = dev_dir / "provenance.json"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
        provenance = json.loads(manifest.read_text(encoding="utf-8"))
        if provenance.get("include_test") is not False: raise ValueError("History source must explicitly exclude test")
        if "development_target_before" in provenance:
            manifest_boundary = _boundary(provenance["development_target_before"])
            if "development_target_before" in kwargs and _boundary(kwargs["development_target_before"]) != manifest_boundary:
                raise ValueError("Caller development boundary contradicts source manifest")
            kwargs["development_target_before"] = manifest_boundary
        result = cls(rows, provenance["common_calendar"], **kwargs)
        result.provenance["source_sha256"] = {source.name: sha256_file(source), manifest.name: sha256_file(manifest),
                                              "generating_code": sha256_file(Path(__file__))}
        return result

    def __len__(self):
        return len(self.centers)


def fit_history_scaler(cohort: HistoryCohort, train_center_indices, market_input: str = "raw", fit_cutoff_utc: str | None = None) -> dict:
    """Unique market snapshots from supplied training centers' histories only."""
    if market_input not in {"raw", "relative"}: raise ValueError("market_input must be raw or relative")
    indices = _indices(train_center_indices, len(cohort))
    cutoff = _time(fit_cutoff_utc) if fit_cutoff_utc is not None else None
    unique = {}
    for index in indices:
        center = cohort.centers[index]
        if center["split"] != "train": raise ValueError("Scaler fit centers must be source training records")
        if cutoff is not None and _time(center["target_available_utc"]) >= cutoff:
            raise ValueError("Training center label not available by fitting cutoff")
        for row in cohort.histories[index]:
            if cutoff is not None and _time(row["cutoff_utc"]) >= cutoff:
                raise ValueError("History input not available before fitting cutoff")
            key = (row["ticker"], row["price_date"])
            values = row[f"market_{market_input}"]
            if key in unique and unique[key] != values:
                raise ValueError("Duplicate stock/session market snapshots disagree")
            unique[key] = values
    keys = sorted(unique)
    values = np.asarray([unique[key] for key in keys], dtype=np.float64)
    mean, scale = values.mean(0), values.std(0)
    scale = np.where(scale < 1e-8, 1., scale)
    return {"market_input": market_input, "mean": mean.tolist(), "scale": scale.tolist(),
            "fit_center_ids": [cohort.centers[i]["id"] for i in indices],
            "unique_market_ids": [f"{ticker}:{date}" for ticker, date in keys],
            "fit_values_sha256": hashlib.sha256(values.tobytes()).hexdigest(), "count": len(keys),
            "fit_cutoff_utc": fit_cutoff_utc, "definition": "unique training-history stock/session population moments"}


class HistoryDataset(Dataset):
    """Materialize only one T×K example at a time from an external mmap cache."""
    def __init__(self, cohort: HistoryCohort, features, scaler: dict, *, indices=None,
                 max_headlines: int = 64, normalize_text: bool = False):
        if not isinstance(max_headlines, int) or isinstance(max_headlines, bool) or max_headlines < 1:
            raise ValueError("Positive headline-slot count required")
        self.cohort, self.features, self.scaler = cohort, features, scaler
        self.indices = list(range(len(cohort))) if indices is None else _indices(indices, len(cohort))
        self.max_headlines, self.normalize_text = max_headlines, normalize_text
        self.rows = [cohort.centers[i] for i in self.indices]
        mean, scale = np.asarray(scaler["mean"]), np.asarray(scaler["scale"])
        if mean.shape != (6,) or scale.shape != (6,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0) or scaler["market_input"] not in {"raw", "relative"}:
            raise ValueError("Invalid training-history market transform")
        self.market_mean, self.market_scale = mean, scale

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        center_index = self.indices[index]; center = self.cohort.centers[center_index]
        history = self.cohort.histories[center_index]
        t, k, e = len(history), self.max_headlines, self.features.text.shape[1]
        text, sentiment = np.zeros((t, k, e), np.float32), np.zeros((t, k, 3), np.float32)
        mask, age = np.zeros((t, k), bool), np.zeros((t, k), np.float32)
        availability_age = np.zeros((t, k), np.float32)
        market = np.empty((t, 6), np.float32)
        cutoff = _time(center["cutoff_utc"])
        for day, row in enumerate(history):
            messages = row["headlines"]
            if len(messages) > k: raise ValueError("Existing selected messages exceed declared slots; do not silently truncate")
            ids = [message["id"] for message in messages]
            if ids:
                vectors = np.asarray(self.features.vectors(ids), np.float32)
                if self.normalize_text: vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-12)
                text[day, :len(ids)] = vectors; sentiment[day, :len(ids)] = self.features.sentiments(ids)
                mask[day, :len(ids)] = True
                elapsed = [(cutoff - _time(message["created_at"])).total_seconds() / 3600 for message in messages]
                if not np.isfinite(elapsed).all() or any(value < 0 for value in elapsed): raise ValueError("Future/invalid active news age")
                age[day, :len(ids)] = elapsed
                available_elapsed = [(cutoff - _time(message.get("available_at", message["created_at"]))).total_seconds() / 3600 for message in messages]
                if not np.isfinite(available_elapsed).all() or any(value < 0 for value in available_elapsed):
                    raise ValueError("Future/invalid active news availability age")
                availability_age[day, :len(ids)] = available_elapsed
            market[day] = (np.asarray(row[f"market_{self.scaler['market_input']}"]) - self.market_mean) / self.market_scale
        description = np.asarray(self.features.vectors([f"description:{center['ticker']}"])[0], np.float32).copy()
        if self.normalize_text: description /= max(float(np.linalg.norm(description)), 1e-12)
        if not all(np.isfinite(value).all() for value in (text, sentiment, market, description, age, availability_age)):
            raise ValueError("Nonfinite materialized history feature")
        return {"headline_vectors": torch.from_numpy(text), "sentiment": torch.from_numpy(sentiment),
                "mask": torch.from_numpy(mask), "headline_age_hours": torch.from_numpy(age),
                "headline_availability_age_hours": torch.from_numpy(availability_age),
                "market_features": torch.from_numpy(market), "market_description": torch.from_numpy(description),
                "target": torch.tensor(center["target_up"], dtype=torch.float32), "row_index": index}

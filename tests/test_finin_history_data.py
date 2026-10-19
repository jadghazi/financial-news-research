"""Calendar histories, strict feature allowlists and training-only transforms."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.finin_history_data import HistoryCohort, HistoryDataset, fit_history_scaler


class FakeFeatures:
    def __init__(self):
        self.text = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [2, 1]], np.float32)
        self.index = {"a": 0, "b": 1, "c": 2, "d": 3, "description:AAPL": 4}
        self.sentiment = np.tile(np.array([.2, .3, .5], np.float32), (5, 1))
    def vectors(self, ids): return self.text[[self.index[i] for i in ids]]
    def sentiments(self, ids): return self.sentiment[[self.index[i] for i in ids]]


class HistoryDataTests(unittest.TestCase):
    def generic_fixture(self):
        rows, calendar = self.fixture()
        calendar = [value.replace("2014", "2023") for value in calendar]
        for row in rows:
            for key in ("id", "price_date", "cutoff_utc", "target_date", "target_available_utc"):
                row[key] = row[key].replace("2014", "2023")
            for message in row["headlines"]:
                message["created_at"] = message["created_at"].replace("2014", "2023")
        return rows, calendar

    def test_generic_cohort_requires_explicit_validated_boundary(self):
        rows, calendar = self.generic_fixture()
        with self.assertRaises(ValueError):
            HistoryCohort(rows, calendar, 3)
        cohort = HistoryCohort(rows, calendar, 3, development_target_before="2024-01-01")
        self.assertEqual(len(cohort), 3)
        self.assertEqual(cohort.provenance["development_target_before"], "2024-01-01")
        for invalid in (None, "2024-1-1", "2024-02-30", "20240101"):
            with self.assertRaises(ValueError):
                HistoryCohort(rows, calendar, development_target_before=invalid)
        with self.assertRaises(ValueError):
            HistoryCohort(rows, calendar, development_target_before=rows[-1]["target_date"])

    def test_message_availability_validation_and_separate_ages(self):
        rows, calendar = self.fixture()
        rows[2]["headlines"][0]["available_at"] = "2014-01-06T21:00:00Z"
        cohort = HistoryCohort(rows, calendar, 3, center_indices=[2])
        sample = HistoryDataset(cohort, FakeFeatures(), fit_history_scaler(cohort, [0]), max_headlines=2)[0]
        self.assertEqual(sample["headline_age_hours"][2, 0].item(), 2.)
        self.assertEqual(sample["headline_availability_age_hours"][2, 0].item(), 1.)
        self.assertEqual(sample["headline_availability_age_hours"][0, 0].item(), 98.)
        self.assertEqual(sample["headline_availability_age_hours"][1].sum().item(), 0.)
        for invalid in ("2014-01-06T22:00:01Z", "2014-01-06T19:59:59Z", "2014-01-06T21:00:00"):
            changed = copy.deepcopy(rows)
            changed[2]["headlines"][0]["available_at"] = invalid
            with self.assertRaises(ValueError):
                HistoryCohort(changed, calendar, 3)

    def test_explicit_price_availability_overrides_legacy_clock(self):
        rows, calendar = self.fixture()
        rows[2]["cutoff_utc"] = "2014-01-06T21:00:00Z"
        rows[2]["price_available_utc"] = "2014-01-06T20:30:00Z"
        cohort = HistoryCohort(rows, calendar, 3, center_indices=[2])
        self.assertEqual(cohort.histories[0][-1]["price_available_utc"], "2014-01-06T20:30:00Z")
        rows[2]["price_available_utc"] = "2014-01-06T21:00:01Z"
        with self.assertRaises(ValueError):
            HistoryCohort(rows, calendar, 3)

    def test_manifest_boundary_derivation_and_contradiction_guard(self):
        rows, calendar = self.generic_fixture()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "devsamples_post_close.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            manifest = {"include_test": False, "common_calendar": calendar, "development_target_before": "2024-01-01"}
            (path / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
            cohort = HistoryCohort.from_directory(path, "post_close", history_days=3)
            self.assertEqual(cohort.provenance["development_target_before"], "2024-01-01")
            with self.assertRaises(ValueError):
                HistoryCohort.from_directory(path, "post_close", development_target_before="2025-01-01")
            del manifest["development_target_before"]
            (path / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                HistoryCohort.from_directory(path, "post_close")
            explicit = HistoryCohort.from_directory(path, "post_close", development_target_before="2024-01-01")
            self.assertEqual(explicit.provenance["development_target_before"], "2024-01-01")
            del manifest["include_test"]
            (path / "provenance.json").write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(ValueError):
                HistoryCohort.from_directory(path, "post_close")

    def fixture(self):
        calendar = ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08", "2014-01-09"]
        rows = []
        for i, date in enumerate(calendar[:-1]):
            headlines = [] if i == 1 else [{"id": "abcd"[i % 4], "created_at": date + "T20:00:00Z", "text": "ignored text"}]
            rows.append({"id": f"post_close_AAPL_{calendar[i+1]}", "ticker": "AAPL", "forecast_schedule": "post_close",
                         "price_date": date, "cutoff_utc": date + "T22:00:00Z", "target_date": calendar[i+1],
                         "target_available_utc": calendar[i+1] + "T22:00:00Z", "split": "valid" if i == 4 else "train",
                         "market_raw": [float(i)] * 6, "market_relative": [float(i)/10] * 6,
                         "description": "static Apple description", "headlines": headlines,
                         "target_up": i % 2, "target_return": 999, "open_up": 999, "band_eligible": True})
        return rows, calendar

    def test_weekend_sessions_empty_day_and_actual_news_age(self):
        rows, calendar = self.fixture()
        cohort = HistoryCohort(rows, calendar, history_days=3, center_indices=[2])
        self.assertEqual([r["price_date"] for r in cohort.histories[0]], calendar[:3])
        scaler = fit_history_scaler(cohort, [0])
        sample = HistoryDataset(cohort, FakeFeatures(), scaler, max_headlines=2)[0]
        self.assertEqual(tuple(sample["headline_vectors"].shape), (3, 2, 2))
        self.assertEqual(int(sample["mask"][1].sum()), 0)
        self.assertEqual(torch.count_nonzero(sample["headline_vectors"][1]).item(), 0)
        self.assertEqual(torch.count_nonzero(sample["headline_age_hours"][1]).item(), 0)
        self.assertEqual(sample["headline_age_hours"][0, 0].item(), 98.)  # Thu20->Mon22 includes weekend
        self.assertEqual(sample["headline_age_hours"][2, 0].item(), 2.)
        self.assertEqual(sample["target"].item(), rows[2]["target_up"])

    def test_past_labels_never_imported_and_mutation_does_not_change_features(self):
        rows, calendar = self.fixture()
        first = HistoryCohort(rows, calendar, history_days=3, center_indices=[2])
        changed = copy.deepcopy(rows)
        for row in changed[:2]:
            row["target_up"] = 1-row["target_up"]; row["target_return"] = -1000; row["open_up"] = -20
        second = HistoryCohort(changed, calendar, history_days=3, center_indices=[2])
        self.assertEqual(first.histories, second.histories)
        for history in first.histories:
            for row in history:
                self.assertNotIn("target_up", row); self.assertNotIn("target_return", row)
        scaler = fit_history_scaler(first, [0])
        a, b = HistoryDataset(first, FakeFeatures(), scaler)[0], HistoryDataset(second, FakeFeatures(), scaler)[0]
        for field in a:
            if isinstance(a[field], torch.Tensor): torch.testing.assert_close(a[field], b[field], rtol=0, atol=0)

    def test_common_cohort_and_no_missing_session_fallback(self):
        rows, calendar = self.fixture()
        short = HistoryCohort(rows, calendar, 1, common_cohort_max_history=3)
        long = HistoryCohort(rows, calendar, 3, common_cohort_max_history=3)
        self.assertEqual(short.center_source_indices, long.center_source_indices)
        self.assertEqual([r["id"] for r in short.centers], [r["id"] for r in long.centers])
        missing = HistoryCohort(rows[:1]+rows[2:], calendar, 3)
        self.assertNotIn(rows[2]["id"], [r["id"] for r in missing.centers])
        self.assertTrue(any(r["reason"] == "missing_session_inputs" for r in missing.excluded))

    def test_scaler_unique_training_only_snapshots_and_availability(self):
        rows, calendar = self.fixture()
        cohort = HistoryCohort(rows, calendar, 3)
        scaler = fit_history_scaler(cohort, [0, 1])
        self.assertEqual(scaler["count"], 4)  # overlap sessions counted once, not six times
        np.testing.assert_allclose(scaler["mean"], np.full(6, 1.5))
        changed = copy.deepcopy(rows); changed[-1]["market_raw"] = [9999.] * 6
        after = HistoryCohort(changed, calendar, 3)
        self.assertEqual(scaler, fit_history_scaler(after, [0, 1]))
        with self.assertRaises(ValueError): fit_history_scaler(cohort, [2])  # validation center
        with self.assertRaises(ValueError): fit_history_scaler(cohort, [0, 0])
        with self.assertRaises(ValueError): fit_history_scaler(cohort, [0], fit_cutoff_utc=cohort.centers[0]["target_available_utc"])

    def test_future_timestamp_and_shape_guard(self):
        rows, calendar = self.fixture()
        rows[2]["headlines"][0]["created_at"] = "2014-01-06T22:00:01Z"
        with self.assertRaises(ValueError): HistoryCohort(rows, calendar, 3)
        rows, calendar = self.fixture()
        rows[2]["cutoff_utc"] = "2014-01-06T12:00:00Z"
        with self.assertRaises(ValueError): HistoryCohort(rows, calendar, 3)


if __name__ == "__main__": unittest.main()

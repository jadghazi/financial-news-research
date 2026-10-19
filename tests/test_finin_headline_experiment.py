"""Availability/matched-date and frozen-transform tests without training."""
import copy
from datetime import date, timedelta
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.finin_headline_experiment import TransformFeatures, _masked_draw, _masked_interval, fit_control_arm, fit_text_transform, fold_training_cohort, history_collection_metadata, model_factory, rolling_plans, validate_frozen_features
from src.finin_history_data import HistoryCohort
from src.reproduction_v2_features import ENCODERS, SENTIMENT


class TinyFeatures:
    def __init__(self):
        self.ids = {"a": 0, "b": 1, "later": 2, "description:SPY": 3}
        self.text = np.array([[1., 3.], [3., 7.], [1000., 3000.], [5., 2.]], np.float32)
    def vectors(self, ids): return self.text[[self.ids[i] for i in ids]]
    def sentiments(self, ids): return np.zeros((len(ids), 3), np.float32)


class HeadlineExperimentTests(unittest.TestCase):
    def test_control_dispatch_activates_news_only_for_declared_secondary_arm(self):
        calls = []
        sentinel = object()
        def record_fit(dataset, *, coverage=False, news=False):
            self.assertIs(dataset, sentinel)
            calls.append((coverage, news))
            return {"coverage": coverage, "news": news}
        fit_control_arm(record_fit, sentinel, "linear_prices")
        fit_control_arm(record_fit, sentinel, "linear_prices_coverage")
        fitted = fit_control_arm(record_fit, sentinel, "linear_prices_coverage_news")
        self.assertEqual(calls, [(False, False), (True, False), (True, True)])
        self.assertTrue(fitted["news"])
        with self.assertRaises(ValueError):
            fit_control_arm(record_fit, sentinel, "undeclared_arm")

    def test_sensitivity_samples_full_clock_before_mask_and_counts_empty_draws(self):
        class FixedStarts:
            def __init__(self, start): self.start = start
            def integers(self, low, high, size): return np.full(size, self.start, dtype=int)
        delta = np.array([1., 90., 90., 9.])
        mask = np.array([True, False, False, True])
        total, count = _masked_draw(delta, [4], mask, FixedStarts(0), block=2)
        # Full blocks [0,1],[0,1] retain [0,0], rather than jumping over the
        # excluded dates to compressed adjacent eligible observations [0,3].
        self.assertEqual((total, count), (2., 2))
        self.assertEqual(_masked_draw(delta, [4], mask, FixedStarts(1), block=2), (0., 0))
        sparse = np.zeros(30, dtype=bool); sparse[0] = True
        values = np.arange(30, dtype=float)
        first = _masked_interval(values, [15, 15], sparse)
        second = _masked_interval(values, [15, 15], sparse)
        self.assertEqual(first, second)
        self.assertEqual(first["replicates"], 2000)
        self.assertGreater(first["empty_attempts"], 0)
        self.assertEqual(first["attempts"], 2000 + first["empty_attempts"])
        self.assertEqual(first["original_timeline_counts_by_fold"], [15, 15])
        self.assertEqual(first["original_eligible_count"], 1)
        self.assertEqual(first["ci95"], [0., 0.])
        self.assertGreater(first["sampled_eligible_count"]["max"], 1)

    def test_complete_history_uses_all_three_explicit_source_flags(self):
        source = [{"ticker": "SP500", "forecast_schedule": "post_close", "price_date": f"2023-01-{day:02d}",
                   "collection_missing": False, "missing_collection_dates": [], "target_up": 1}
                  for day in (3, 4, 5)]
        inputs = [{key: row[key] for key in ("ticker", "forecast_schedule", "price_date")} for row in source]
        cohort = SimpleNamespace(histories=[inputs])
        complete = history_collection_metadata(cohort, source)[0]
        self.assertTrue(complete["history_collection_flags_available"])
        self.assertFalse(complete["history_collection_missing"])
        changed = copy.deepcopy(source)
        changed[0]["collection_missing"] = True
        changed[0]["missing_collection_dates"] = ["2023-01-01"]
        changed[0]["target_up"] = 0
        missing = history_collection_metadata(cohort, changed)[0]
        self.assertTrue(missing["history_collection_missing"])
        self.assertEqual(missing["history_missing_collection_dates"], ["2023-01-01"])
        # Center day alone is complete; older-day missing flags still exclude it.
        self.assertFalse(changed[-1]["collection_missing"])
        del changed[1]["collection_missing"]
        unknown = history_collection_metadata(cohort, changed)[0]
        self.assertFalse(unknown["history_collection_flags_available"])
        self.assertIsNone(unknown["history_collection_missing"])

    def test_frozen_encoder_guard_rejects_same_width_wrong_models_and_revisions(self):
        manifest = {"identity": {"encoder": copy.deepcopy(ENCODERS["roberta"]), "sentiment": copy.deepcopy(SENTIMENT),
                                 "pooling": "last_hidden_state[:,0,:]", "l2_normalized": False},
                    "states": {kind: {"model_provenance": {"model": configuration["name"], "revision": configuration["revision"]}}
                               for kind, configuration in (("text", ENCODERS["roberta"]), ("sentiment", SENTIMENT))}}
        fixture = SimpleNamespace(text=np.zeros((2, 768)), manifest=manifest)
        self.assertTrue(validate_frozen_features(fixture))
        for field in ("name", "revision"):
            changed = copy.deepcopy(fixture)
            changed.manifest["identity"]["encoder"][field] = "wrong-but-still-768"
            with self.assertRaises(ValueError): validate_frozen_features(changed)
        for kind in ("text", "sentiment"):
            for field in ("model", "revision"):
                changed = copy.deepcopy(fixture)
                changed.manifest["states"][kind]["model_provenance"][field] = "wrong"
                with self.assertRaises(ValueError): validate_frozen_features(changed)
        changed = copy.deepcopy(fixture)
        changed.manifest["identity"]["sentiment"]["revision"] = "wrong"
        with self.assertRaises(ValueError): validate_frozen_features(changed)

    def test_history_lengths_share_five_session_centers_and_no_news_initialization(self):
        calendar = [f"2023-01-{i:02d}" for i in range(2, 12)]
        records = []
        for i, day in enumerate(calendar[:-1]):
            records.append({"id": f"SPY:{day}", "ticker": "SPY", "forecast_schedule": "post_close",
                            "price_date": day, "cutoff_utc": day+"T22:00:00Z", "target_date": calendar[i+1],
                            "target_available_utc": calendar[i+1]+"T22:00:00Z", "split": "train",
                            "target_up": i % 2, "market_raw": [float(i)]*6, "market_relative": [float(i)]*6,
                            "description": "fixed SPY", "headlines": []})
        cohorts = [HistoryCohort(records, calendar, history_days=t, common_cohort_max_history=5,
                                 development_target_before="2024-01-01") for t in (1, 3, 5)]
        self.assertEqual(cohorts[0].center_source_indices, cohorts[1].center_source_indices)
        self.assertEqual(cohorts[1].center_source_indices, cohorts[2].center_source_indices)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            full = model_factory("full_concat")
            torch.manual_seed(17)
            nonews = model_factory("matched_no_news")
            self.assertEqual(full.state_dict().keys(), nonews.state_dict().keys())
            for name, value in full.state_dict().items():
                torch.testing.assert_close(value, nonews.state_dict()[name], atol=0, rtol=0)

    def test_fold_ownership_is_explicit_and_does_not_mutate_source(self):
        source = SimpleNamespace(centers=[{"id": "early", "split": "train"},
                                          {"id": "later_past", "split": "valid"},
                                          {"id": "future", "split": "valid"}], histories=[[1], [2], [3]])
        original = copy.deepcopy(source.centers)
        folded = fold_training_cohort(source, [0, 1])
        self.assertEqual([r["split"] for r in folded.centers], ["train", "train", "valid"])
        self.assertEqual([r["source_split"] for r in folded.centers], ["train", "valid", "valid"])
        self.assertEqual(source.centers, original)
        folded.centers[1]["split"] = "valid"
        self.assertEqual(source.centers, original)

    def test_fixed_fold_dates_and_strict_label_availability_purge(self):
        centers = []
        current = date(2022, 9, 20)
        while current < date(2024, 1, 1):
            previous = current-timedelta(days=1)
            centers.append({"id": str(current), "ticker": "SPY", "target_date": str(current), "split": "train",
                            "cutoff_utc": f"{previous}T22:00:00Z", "target_available_utc": f"{current}T22:00:00Z"})
            current += timedelta(days=1)
        plans = rolling_plans(centers)
        first = plans[0]
        initial2022 = next(i for i,r in enumerate(centers) if r["target_date"] == "2022-09-20")
        self.assertIn(initial2022, first["train_indices"])
        dec31 = next(i for i,r in enumerate(centers) if r["target_date"] == "2022-12-31")
        jan31 = next(i for i,r in enumerate(centers) if r["target_date"] == "2023-01-31")
        self.assertIn(dec31, first["training_purged_indices"])
        self.assertNotIn(dec31, first["train_indices"])
        self.assertIn(jan31, first["selection_purged_indices"])
        self.assertEqual(first["evaluation_target_dates"][0], "2023-02-01")
        self.assertEqual(first["evaluation_target_dates"][-1], "2023-03-31")
        self.assertEqual(plans[1]["evaluation_target_dates"][0], "2023-06-01")
        self.assertEqual(plans[1]["evaluation_target_dates"][-1], "2023-07-31")
        self.assertEqual(plans[2]["evaluation_target_dates"][0], "2023-08-01")
        self.assertEqual(plans[2]["evaluation_target_dates"][-1], "2023-09-30")
        may_past = next(i for i,r in enumerate(centers) if r["target_date"] == "2023-05-15")
        self.assertNotIn(may_past, plans[1]["train_indices"])
        self.assertIn(may_past, plans[1]["selection_indices"])
        self.assertIn(may_past, plans[2]["train_indices"])
        evaluation = [d for p in plans for d in p["evaluation_target_dates"]]
        self.assertEqual(len(evaluation), len(set(evaluation)))
        for plan in plans:
            for role in ("train", "selection", "evaluation"):
                self.assertEqual(plan[f"{role}_ids"], [centers[i]["id"] for i in plan[f"{role}_indices"]])

    def test_training_transform_excludes_later_features_and_preserves_cache(self):
        cohort = SimpleNamespace(
            centers=[{"id": "train", "target_available_utc": "2023-01-10T22:00:00Z"},
                     {"id": "later", "target_available_utc": "2023-03-10T22:00:00Z"}],
            histories=[[{"cutoff_utc": "2023-01-09T22:00:00Z", "headlines": [{"id": "a"}, {"id": "b"}]}],
                       [{"cutoff_utc": "2023-03-09T22:00:00Z", "headlines": [{"id": "later"}]}]],
        )
        # HistoryCohort supplies __len__; a minimal wrapper keeps this a pure
        # statistic test without constructing encoder/model/training machinery.
        class Cohort:
            centers, histories = cohort.centers, cohort.histories
            def __len__(self): return len(self.centers)
        cohort = Cohort()
        features = TinyFeatures()
        before = features.text.copy()
        fitted = fit_text_transform(cohort, features, [0], fit_cutoff_utc="2023-02-01T22:00:00Z", mode="center_rms")
        np.testing.assert_array_equal(fitted["mean"], [2., 5.])
        self.assertAlmostEqual(fitted["rms"], np.sqrt(2.5))
        features.text[2] *= 100
        after = fit_text_transform(cohort, features, [0], fit_cutoff_utc="2023-02-01T22:00:00Z", mode="center_rms")
        self.assertEqual(fitted, after)
        adapter = TransformFeatures(features, fitted)
        returned = adapter.vectors(["a"])
        returned[:] = 99
        np.testing.assert_array_equal(features.text[:2], before[:2])
        with self.assertRaises(ValueError):
            fit_text_transform(cohort, features, [0, 1], fit_cutoff_utc="2023-02-01T22:00:00Z", mode="center_rms")
        with self.assertRaises(ValueError):
            fit_text_transform(cohort, features, [0], fit_cutoff_utc=None, mode="center_rms")


if __name__ == "__main__":
    unittest.main()

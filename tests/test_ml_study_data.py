"""Boundary, preprocessing and immutable input tests without model training."""
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from src.ml_study_data import StudyDataset, chronological_plan, fit_scaler


class StudyDataTests(unittest.TestCase):
    def test_scaler_uses_only_declared_fit_rows_and_population_std(self):
        raw = np.array([[1., 4.], [3., 4.], [10000., -10000.]])
        mean, scale = fit_scaler(raw, [0, 1])
        np.testing.assert_array_equal(mean, [2., 4.])
        np.testing.assert_array_equal(scale, [1., 1.])
        changed = raw.copy()
        changed[2] *= 100
        altered = fit_scaler(changed, [0, 1])
        np.testing.assert_array_equal(altered[0], mean)
        np.testing.assert_array_equal(altered[1], scale)

    def test_scaler_rejects_duplicate_indices(self):
        with self.assertRaises(ValueError):
            fit_scaler(np.ones((3, 6)), [0, 0])

    def test_purge_uses_actual_sessions_when_news_dates_have_gaps(self):
        sessions = [f"2020-01-{i:02d}" for i in range(1, 17)]
        forecasts = [1, 2, 4, 5, 6, 8, 10, 11, 12]
        rows = [{"forecast_date": sessions[i], "target_date": sessions[i + 1]} for i in forecasts]
        plan = chronological_plan(rows, sessions, starts=(5,), ends=(9,), inner_fraction=.4)[0]
        self.assertEqual(plan["outer_cutoff"], sessions[7])
        self.assertEqual(plan["outer_train_indices"], list(range(5)))
        self.assertEqual(plan["outer_purged_indices"], [])
        self.assertEqual(plan["inner_valid_indices"], [3, 4])
        self.assertEqual(plan["inner_cutoff"], sessions[4])
        # Row 2 forecasts on the immediately preceding market session and has
        # a target unavailable at that closing cutoff, despite the news gap.
        self.assertEqual(plan["inner_purged_indices"], [2])
        self.assertEqual(plan["inner_fit_indices"], [0, 1])

    def test_future_targets_are_purged_even_with_longer_horizon(self):
        sessions = [f"2020-01-{i:02d}" for i in range(1, 21)]
        rows = [{"forecast_date": sessions[i], "target_date": sessions[i + 2]} for i in range(1, 17)]
        plans = chronological_plan(rows, sessions, starts=(9, 12), ends=(12, 16), inner_fraction=.4)
        self.assertEqual(plans[0]["outer_purged_indices"], [7, 8])
        self.assertEqual(plans[0]["outer_train_indices"], list(range(7)))
        for plan in plans:
            for i in plan["outer_train_indices"]:
                self.assertLessEqual(rows[i]["target_date"], plan["outer_cutoff"])
            for i in plan["inner_fit_indices"]:
                self.assertLessEqual(rows[i]["target_date"], plan["inner_cutoff"])
        self.assertFalse(set(plans[0]["outer_eval_indices"]) & set(plans[1]["outer_eval_indices"]))

    def test_rejects_overlapping_outer_blocks(self):
        sessions = [f"2020-01-{i:02d}" for i in range(1, 16)]
        rows = [{"forecast_date": sessions[i], "target_date": sessions[i + 1]} for i in range(1, 12)]
        with self.assertRaises(ValueError):
            chronological_plan(rows, sessions, starts=(6, 7), ends=(8, 10), inner_fraction=.4)

    def test_tensor_selection_scaling_and_getter_are_consistent(self):
        cohort = SimpleNamespace(
            rows=[{"id": "first", "target_up": 0}, {"id": "second", "target_up": 1}],
            raw_market=np.array([[0.] * 6, [3.] * 6]),
            description=np.arange(384, dtype=np.float32),
            _vectors=np.stack([np.ones((16, 384)), np.full((16, 384), 2)]).astype(np.float32),
            _scores=np.ones((2, 16, 3), dtype=np.float32),
            _masks=np.ones((2, 16), dtype=bool),
        )
        data = StudyDataset(cohort, [1, 0], np.ones(6), np.full(6, 2.))
        sample = data[0]
        self.assertEqual(data.rows[0]["id"], "second")
        self.assertEqual(sample["row_index"], 1)
        self.assertEqual(sample["target"].item(), 1)
        torch.testing.assert_close(sample["market_features"], torch.ones(6))
        torch.testing.assert_close(sample["headline_vectors"], torch.full((16, 384), 2.))
        self.assertEqual(sample["mask"].dtype, torch.bool)
        sample["headline_vectors"].zero_()
        sample["market_description"].zero_()
        torch.testing.assert_close(data[0]["headline_vectors"], torch.full((16, 384), 2.))
        torch.testing.assert_close(data[0]["market_description"], torch.arange(384, dtype=torch.float32))
        np.testing.assert_array_equal(data.raw_market[0], np.full(6, 3.))


if __name__ == "__main__":
    unittest.main()

"""Meaningful chronological/intercept tests; no financial data or NN fits."""

from datetime import date, timedelta
import unittest

import numpy as np

from src.encoder_comparison import metrics
from src.finin_calibration_study import apply_offsets, chronological_blocks, fit_offsets, TICKERS


class CalibrationTests(unittest.TestCase):
    def test_zero_logit_global_intercept_recovers_prevalence(self):
        labels = np.array([0.] * 30 + [1.] * 70)
        fitted = fit_offsets(np.zeros(100), labels, ["AAPL"] * 100, "global")
        self.assertAlmostEqual(fitted["alpha"], np.log(.7 / .3), places=6)
        self.assertLess(fitted["gradient_max"], 1e-5)
        np.testing.assert_array_equal(list(fitted["stock_delta"].values()), np.zeros(8))

    def test_stock_intercept_preserves_within_stock_ranking_and_zero_sum(self):
        logits = np.tile(np.linspace(-2, 2, 12), 8)
        tickers = np.repeat(TICKERS, 12).tolist()
        labels = np.concatenate([np.array([0.] * (2 + i) + [1.] * (10 - i)) for i in range(8)])
        fitted = fit_offsets(logits, labels, tickers, "stock")
        shifted = apply_offsets(logits, tickers, fitted)
        self.assertAlmostEqual(sum(fitted["stock_delta"].values()), 0., places=12)
        self.assertLess(fitted["gradient_max"], 1e-5)
        for ticker in TICKERS:
            chosen = np.array([t == ticker for t in tickers])
            self.assertEqual(metrics(labels[chosen], logits[chosen])["auc"], metrics(labels[chosen], shifted[chosen])["auc"])
            np.testing.assert_allclose(np.diff(shifted[chosen]), np.diff(logits[chosen]))

    def test_chronological_boundaries_purge_unavailable_labels(self):
        days = [date(2015, 7, 1) + timedelta(days=i) for i in range(64)]
        rows = []
        for day in days:
            for ticker in TICKERS:
                rows.append({"split": "valid", "target_date": day.isoformat(), "ticker": ticker,
                             "cutoff_utc": f"{(day - timedelta(days=1)).isoformat()}T22:00:00Z",
                             "target_available_utc": f"{day.isoformat()}T22:00:00Z"})
        blocks = chronological_blocks(rows)
        self.assertEqual(len(blocks["selection"]), 23 * 8)
        self.assertEqual(len(blocks["calibration"]), 15 * 8)
        self.assertEqual(len(blocks["evaluation"]), 24 * 8)
        self.assertEqual(len(blocks["selection_purged"]), 8)
        self.assertEqual(len(blocks["calibration_purged"]), 8)
        for row in rows:
            row["cutoff_utc"] = f"{row['target_date']}T12:00:00Z"
        before_open = chronological_blocks(rows)
        self.assertEqual(len(before_open["selection"]), 24 * 8)
        self.assertEqual(len(before_open["calibration"]), 16 * 8)
        rows[-1]["split"] = "test"
        with self.assertRaises(ValueError): chronological_blocks(rows)


if __name__ == "__main__":
    unittest.main()

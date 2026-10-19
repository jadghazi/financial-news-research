"""Final linear-news design must match the original probe without fitting."""

import unittest

import numpy as np

from src.reproduction_v2_final import linear_news_design
from src.reproduction_v2_probe import controls


class FakeFeatures:
    def __init__(self):
        self.text = np.array([[1, 3], [5, 7], [-2, 8]], dtype=np.float32)
        self.sentiment = np.array([[.1, .2, .7], [.3, .3, .4], [.8, .1, .1]], dtype=np.float32)
        self.index = {"a": 0, "b": 1, "c": 2}

    def vectors(self, ids):
        return self.text[[self.index[i] for i in ids]]

    def sentiments(self, ids):
        return self.sentiment[[self.index[i] for i in ids]]


class FinalLinearDesignTests(unittest.TestCase):
    def setUp(self):
        self.features = FakeFeatures()
        self.rows = [
            {"ticker": "AAPL", "raw_news_count": 3, "deduplicated_news_count": 2, "retained_news_count": 2,
             "headlines": [{"id": "a"}, {"id": "b"}], "market_relative": [1, 2, 3, 4, 5, 6]},
            {"ticker": "AAPL", "raw_news_count": 4, "deduplicated_news_count": 2, "retained_news_count": 2,
             "headlines": [{"id": "b"}, {"id": "c"}], "market_relative": [6, 5, 4, 3, 2, 1]},
            {"ticker": "AMZN", "raw_news_count": 0, "deduplicated_news_count": 0, "retained_news_count": 0,
             "headlines": [], "market_relative": [0, 0, 0, 0, 0, 0]},
        ]

    def test_raw_means_and_empty_zero_design_match_probe(self):
        identity, coverage, prices = controls(self.rows)
        base = np.column_stack([identity, coverage])
        text = np.array([self.features.text[:2].mean(0), self.features.text[1:].mean(0), np.zeros(2, np.float32)])
        sentiment = np.array([self.features.sentiment[:2].mean(0), self.features.sentiment[1:].mean(0), np.zeros(3, np.float32)])
        news = np.column_stack([text, sentiment])
        np.testing.assert_array_equal(linear_news_design(self.rows, self.features, "news"), np.column_stack([base, news]))
        np.testing.assert_array_equal(linear_news_design(self.rows, self.features, "prices_news"), np.column_stack([base, prices, news]))

    def test_joint_news_disruption_keeps_recipient_control_channels(self):
        original = linear_news_design(self.rows, self.features, "prices_news")
        donors = np.array([1, 0, 2])
        altered = linear_news_design(self.rows, self.features, "prices_news", donors)
        np.testing.assert_array_equal(altered[:, :-5], original[:, :-5])
        np.testing.assert_array_equal(altered[:, -5:], original[donors, -5:])
        with self.assertRaises(ValueError):
            linear_news_design(self.rows, self.features, "news", np.array([2, 0, 1]))
        with self.assertRaises(ValueError):
            linear_news_design(self.rows, self.features, "news", np.array([1, 0]))


if __name__ == "__main__":
    unittest.main()

"""Synthetic generation controls; no model fits in unit tests."""

import unittest

import numpy as np

from src.reproduction_v2_gates import generate


class ReproductionV2GateTests(unittest.TestCase):
    def test_one_salient_news_rule_and_label_independent_mask(self):
        data = generate(512, 301301)
        salient = data["headline_vectors"][:, :, 0] == 2
        np.testing.assert_array_equal(salient.sum(1), np.ones(512))
        self.assertTrue(np.all(data["mask"][salient]))
        known_rule = (data["headline_vectors"][:, :, 1][salient] > 0).astype(float)
        np.testing.assert_array_equal(known_rule, data["target"])
        self.assertEqual(int(data["target"].sum()), 256)
        self.assertTrue(np.all((data["mask"].sum(1) >= 2) & (data["mask"].sum(1) <= 4)))
        self.assertEqual(np.count_nonzero(data["headline_vectors"][~data["mask"]]), 0)
        self.assertEqual(np.count_nonzero(data["sentiment"][~data["mask"]]), 0)
        self.assertGreater(len(set(np.argmax(salient, 1))), 1)

    def test_independent_generation_and_exact_replay(self):
        a, b = generate(256, 301301), generate(256, 401401)
        replay = generate(256, 301301)
        for key in a:
            np.testing.assert_array_equal(a[key], replay[key])
        self.assertFalse(np.array_equal(a["headline_vectors"], b["headline_vectors"]))
        self.assertFalse(np.array_equal(a["market_features"], b["market_features"]))
        with self.assertRaises(ValueError):
            generate(3, 1)


if __name__ == "__main__":
    unittest.main()

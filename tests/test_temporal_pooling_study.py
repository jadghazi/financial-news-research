"""Meaningful temporal aggregation checks; no fitting or model inference."""
import unittest
import copy
import json

import numpy as np

from src import encoder_comparison as ec
from src.temporal_pooling_study import design, metrics_match, pool_features, temporal_weights


class TemporalPoolingTests(unittest.TestCase):
    def test_actual_metric_dictionary_replay_checks_nested_confusion(self):
        measured = ec.metrics(np.array([0., 1., 0., 1.]), np.array([-1., 2., .3, -.2]))
        saved = json.loads(json.dumps(measured))
        self.assertTrue(metrics_match(measured, saved))
        wrong_confusion = copy.deepcopy(saved)
        wrong_confusion["confusion"]["true_up"] += 1
        self.assertFalse(metrics_match(measured, wrong_confusion))
        wrong_loss = copy.deepcopy(saved)
        wrong_loss["log_loss"] += .001
        self.assertFalse(metrics_match(measured, wrong_loss))
        missing = copy.deepcopy(saved)
        del missing["auc"]
        self.assertFalse(metrics_match(measured, missing))

    def test_fully_observed_weights_match_session_half_life(self):
        observed = np.ones((1, 3), dtype=bool)
        np.testing.assert_allclose(temporal_weights(observed, "decay"), [[1/7, 2/7, 4/7]])
        np.testing.assert_allclose(temporal_weights(observed, "uniform"), [[1/3] * 3])
        np.testing.assert_array_equal(temporal_weights(observed, "latest"), [[0., 0., 1.]])

    def test_missing_slots_renormalize_and_empty_returns_zero(self):
        observed = np.array([[True, False, True], [False, False, False]])
        np.testing.assert_allclose(temporal_weights(observed, "decay"), [[.2, 0., .8], [0., 0., 0.]])
        np.testing.assert_allclose(temporal_weights(observed, "uniform"), [[.5, 0., .5], [0., 0., 0.]])
        data = np.array([[[2.], [999.], [4.]], [[999.], [999.], [999.]]])
        np.testing.assert_allclose(pool_features(data, observed, "decay"), [[3.6], [0.]])

    def test_latest_never_falls_back_to_an_older_available_day(self):
        observed = np.array([[True, True, False]])
        np.testing.assert_array_equal(temporal_weights(observed, "latest"), [[0., 0., 0.]])
        np.testing.assert_array_equal(pool_features(np.array([[[2.], [4.], [0.]]]), observed, "latest"), [[0.]])

    def test_pooling_changes_content_before_linear_model_with_common_controls(self):
        arrays = {"prices": np.ones((1, 6)), "coverage": np.arange(9)[None, :],
                  "available": np.ones((1, 3), dtype=bool),
                  "daily_text": np.array([[[0.], [3.], [9.]]]),
                  "daily_sentiment": np.array([[[3.], [6.], [0.]]])}
        latest = design(arrays, "latest")
        uniform = design(arrays, "uniform")
        decay = design(arrays, "decay")
        np.testing.assert_array_equal(latest[:, :15], uniform[:, :15])
        np.testing.assert_array_equal(latest[:, :15], decay[:, :15])
        np.testing.assert_allclose(latest[:, 15:], [[9., 0.]])
        np.testing.assert_allclose(uniform[:, 15:], [[4., 3.]])
        np.testing.assert_allclose(decay[:, 15:], [[6., 15/7]])
        self.assertEqual(latest.shape, uniform.shape)
        self.assertEqual(latest.shape, decay.shape)


if __name__ == "__main__":
    unittest.main()

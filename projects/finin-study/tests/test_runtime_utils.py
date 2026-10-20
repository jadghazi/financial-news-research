"""Contracts for extracted helpers; see docs/code_guide.md#shared-runtime-helpers."""
import hashlib
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch

from src.runtime_utils import metrics, set_seed, sha256_file


class RuntimeUtilsTests(unittest.TestCase):
    def test_streaming_hash_matches_content_digest(self):
        content = bytes(range(256)) * 5000
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.bin"
            path.write_bytes(content)
            self.assertEqual(sha256_file(path), hashlib.sha256(content).hexdigest())

    def test_seeding_reproduces_all_cpu_random_streams(self):
        def draw():
            return random.random(), np.random.random(4), torch.rand(4)
        set_seed(17)
        first = draw()
        set_seed(17)
        second = draw()
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        torch.testing.assert_close(first[2], second[2], atol=0, rtol=0)

    def test_logit_metrics_preserve_ties_and_extreme_error_size(self):
        tied = metrics(np.array([0, 1]), np.zeros(2))
        self.assertEqual(tied["auc"], 0.5)
        self.assertEqual(tied["brier"], 0.25)
        self.assertAlmostEqual(tied["log_loss"], np.log(2))
        extreme = metrics(np.array([0, 1]), np.array([1000., -1000.]))
        self.assertEqual(extreme["log_loss"], 1000.)
        self.assertEqual(extreme["accuracy"], 0.)


if __name__ == "__main__":
    unittest.main()

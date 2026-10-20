"""Cache contracts without model downloads; see docs/code_guide.md#frozen-text-and-sentiment-features."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.frozen_features import _cache_kind, _validate_values, _verify_models
from src.runtime_utils import sha256_file


class FakeEncoder:
    def __init__(self, kind):
        self.kind = kind
        self.provenance = {"model": "test-only", "files_sha256": {}}

    def encode(self, texts):
        if self.kind == "text":
            return np.array([[len(text)] * 768 for text in texts], dtype=np.float32), 0
        return np.array([[.1, .1, .8] if "severe losses" in text else [.8, .1, .1]
                         for text in texts], dtype=np.float32), 0


class FrozenFeaturesTests(unittest.TestCase):
    items = [("a", "profit improves"), ("b", "losses increase")]

    def cache(self, folder, kind="text", identity=None):
        return _cache_kind(kind, self.items, Path(folder), identity or {"schema": 1},
                           "roberta", 2, 64, False, 2)

    def test_completed_cache_resumes_without_loading_model(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.frozen_features.load_frozen", return_value=FakeEncoder("text")):
                state = self.cache(folder)
            with patch("src.frozen_features.load_frozen", side_effect=AssertionError("Must not encode again")):
                self.assertEqual(self.cache(folder), state)
            values = np.load(Path(folder) / "text.npy")
            self.assertEqual(values.shape, (2, 768))
            np.testing.assert_array_equal(values[:, 0], [15., 15.])

    def test_identity_change_rejects_completed_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.frozen_features.load_frozen", return_value=FakeEncoder("text")):
                self.cache(folder)
            with self.assertRaisesRegex(ValueError, "identity changed"):
                self.cache(folder, identity={"schema": 2})

    def test_committed_array_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.frozen_features.load_frozen", return_value=FakeEncoder("text")):
                self.cache(folder)
            array = np.load(Path(folder) / "text.npy", mmap_mode="r+")
            array[0, 0] += 1
            array.flush()
            del array
            with self.assertRaisesRegex(ValueError, "prefix corrupted"):
                self.cache(folder)

    def test_sentiment_controls_and_probabilities_are_retained(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("src.frozen_features.load_frozen", return_value=FakeEncoder("sentiment")):
                state = self.cache(folder, kind="sentiment")
            controls = np.asarray(state["benchmark"]["sentiment_controls_positive_negative"])
            np.testing.assert_array_equal(controls.argmax(1), [0, 2])
            values = np.load(Path(folder) / "sentiment.npy")
            _validate_values(values, 2, 3, "sentiment")

    def test_invalid_sentiment_and_changed_model_file_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "sentiment probabilities"):
            _validate_values(np.zeros((2, 3), dtype=np.float32), 2, 3, "sentiment")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fake-model.bin"
            path.write_bytes(b"original")
            provenance = {"files_sha256": {str(path): sha256_file(path)}}
            _verify_models(provenance)
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "checksum changed"):
                _verify_models(provenance)


if __name__ == "__main__":
    unittest.main()

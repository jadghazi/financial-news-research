"""Fake-encoder cache integrity/resume and training-only transformation tests."""

from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.reproduction_v2_features import FrozenFeatures, apply_pca, build_cache, collect_items, fit_pca


class FakeFrozen:
    def __init__(self, kind, fail_at=None):
        self.kind, self.calls, self.fail_at = kind, 0, fail_at
        self.provenance = {"files_sha256": {}, "model": f"fake-{kind}", "revision": "unit-test"}

    def encode(self, texts):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("Simulated interruption after committed prefix")
        if self.kind == "sentiment":
            rows = [[.1, .1, .8] if "severe losses" in text else [.8, .1, .1] for text in texts]
        else:
            rows = [[len(text) * .001 + i * .0001 for i in range(384)] for text in texts]
        return np.asarray(rows, dtype=np.float32), 0


class ReproductionV2FeatureTests(unittest.TestCase):
    def source(self, root, count=3):
        directory = root / "dev"
        directory.mkdir()
        headlines = []
        for i in range(count):
            text = f"actual message {i}"
            headlines.append({"id": hashlib.sha256(text.encode()).hexdigest(), "text": text})
        row = {"split": "train", "ticker": "AAPL", "description": "Apple technology company",
               "target_date": "2014-01-06", "headlines": headlines}
        for schedule in ("post_close", "pre_open"):
            (directory / f"devsamples_{schedule}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        return directory, row

    def factory(self, kind, encoder, max_tokens, normalize):
        return FakeFrozen(kind)

    def test_cache_lookup_dedup_empty_means_and_completed_reuse(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            dev, row = self.source(root)
            self.assertEqual(len(collect_items(dev)), 4)
            with patch("src.reproduction_v2_features.load_frozen", side_effect=self.factory) as loader:
                result = build_cache(dev, root / "features", benchmark_size=2)
                self.assertEqual(loader.call_count, 2)
            self.assertEqual(result["message_items"], 3)
            self.assertEqual(result["description_items"], 1)
            with patch("src.reproduction_v2_features.load_frozen", side_effect=AssertionError("No reload expected")):
                self.assertEqual(result, build_cache(dev, root / "features", benchmark_size=2))
            features = FrozenFeatures(root / "features")
            text, sentiment = features.daily_mean(row)
            identifiers = [r["id"] for r in row["headlines"]]
            np.testing.assert_allclose(text, features.vectors(identifiers).mean(0))
            np.testing.assert_allclose(sentiment, [.8, .1, .1])
            self.assertEqual(features.description("AAPL").shape, (384,))
            empty = features.daily_mean({"headlines": []})
            self.assertEqual(np.count_nonzero(empty[0]), 0)
            self.assertEqual(np.count_nonzero(empty[1]), 0)
            features.close()

    def test_partial_resume_only_encodes_uncommitted_rows(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            dev, _ = self.source(root, count=600)
            failing = FakeFrozen("text", fail_at=34)  # benchmark one batch, then 32 committed batches
            with patch("src.reproduction_v2_features.load_frozen", return_value=failing):
                with self.assertRaises(RuntimeError):
                    build_cache(dev, root / "features", benchmark_size=2)
            state = json.loads((root / "features/text_state.json").read_text())
            self.assertEqual(state["next"], 512)
            committed = np.load(root / "features/text.npy")[:512].copy()
            resumed = FakeFrozen("text")
            with patch("src.reproduction_v2_features.load_frozen", side_effect=lambda kind, *args: resumed if kind == "text" else FakeFrozen(kind)):
                manifest = build_cache(dev, root / "features", benchmark_size=2)
            self.assertEqual(resumed.calls, 6)  # 601 items minus 512; no repeat benchmark
            self.assertEqual(manifest["states"]["text"]["next"], 601)
            np.testing.assert_array_equal(np.load(root / "features/text.npy")[:512], committed)

    def test_changed_input_and_corrupted_cache_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root = Path(temp)
            dev, row = self.source(root)
            with patch("src.reproduction_v2_features.load_frozen", side_effect=self.factory):
                build_cache(dev, root / "features", benchmark_size=2)
            array = np.lib.format.open_memmap(root / "features/text.npy", mode="r+")
            array[0, 0] += 1
            array.flush()
            array._mmap.close()
            with self.assertRaises(ValueError):
                FrozenFeatures(root / "features")
            with self.assertRaises(ValueError):
                build_cache(dev, root / "features", benchmark_size=2)
            row["description"] = "different description"
            for schedule in ("post_close", "pre_open"):
                (dev / f"devsamples_{schedule}.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                build_cache(dev, root / "features", benchmark_size=2)

    def test_test_rows_and_changed_message_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dev, row = self.source(root)
            row["split"] = "test"
            (dev / "devsamples_post_close.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                collect_items(dev)
            row["split"] = "train"
            row["headlines"][0]["text"] = "content changed without ID"
            (dev / "devsamples_post_close.jsonl").write_text(json.dumps(row) + "\n")
            with self.assertRaises(ValueError):
                collect_items(dev)

    def test_pca_fitting_is_unchanged_by_validation_and_application_centered(self):
        raw = np.array([[1, 0, 2], [2, 1, 1], [4, 0, 2], [100, -20, 30], [-5, 50, 1]], dtype=float)
        a = fit_pca(raw, [0, 1, 2], 2)
        changed = raw.copy()
        changed[3:] *= -100
        b = fit_pca(changed, [0, 1, 2], 2)
        np.testing.assert_array_equal(a["mean"], b["mean"])
        np.testing.assert_array_equal(a["basis"], b["basis"])
        self.assertEqual(a["fit_rows_sha256"], b["fit_rows_sha256"])
        transformed = apply_pca(raw, a)
        np.testing.assert_allclose(transformed[:3].mean(0), [0, 0], atol=1e-7)
        np.testing.assert_allclose(transformed, (raw - a["mean"]) @ a["basis"].T, rtol=1e-6)
        self.assertEqual(transformed.shape, (5, 2))
        for invalid in ([0, 0], [-1], [1.2], []):
            with self.assertRaises(ValueError):
                fit_pca(raw, invalid, 2)


if __name__ == "__main__":
    unittest.main()

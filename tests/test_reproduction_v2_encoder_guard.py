"""Reject mislabeled encoder caches before any financial fit or test unsealing."""

from contextlib import ExitStack
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.reproduction_v2_features import ENCODERS
from src.reproduction_v2_neural import run as neural_run
from src.reproduction_v2_final import run as final_run


class EncoderGuardTests(unittest.TestCase):
    def fixture(self, root, failure):
        (root / "gates").mkdir()
        (root / "gates/summary.json").write_text(json.dumps({"all_positive_gates_pass": True}))
        cache = root / "roberta_cache"
        cache.mkdir()
        encoder = copy.deepcopy(ENCODERS["roberta"])
        generator = {"model": encoder["name"], "revision": encoder["revision"], "files_sha256": {}}
        width = 768
        if failure == "identity":
            encoder = copy.deepcopy(ENCODERS["bge"])
        elif failure == "generator":
            generator["model"], generator["revision"] = ENCODERS["bge"]["name"], ENCODERS["bge"]["revision"]
        elif failure == "width":
            width = 384
        else:
            raise ValueError("Unknown fixture failure")
        manifest = {"identity": {"encoder": encoder}, "states": {"text": {"model_provenance": generator}}}
        (cache / "manifest.json").write_text(json.dumps(manifest))
        np.save(cache / "text.npy", np.zeros((2, width), dtype=np.float32))
        return manifest

    def test_neural_guard_rejects_identity_generator_and_width_before_fitting(self):
        expected = {"identity": "Named experiment encoder", "generator": "Cache generator", "width": "Actual cache width"}
        for failure, message in expected.items():
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                self.fixture(root, failure)
                mocks = [stack.enter_context(patch(f"src.reproduction_v2_neural.{name}", side_effect=AssertionError(f"{name} must not execute")))
                         for name in ("fixed_budget_fit", "read_rows", "fit_pca", "FrozenFeatures")]
                # Stub loading an array to avoid leaving an mmap handle open
                # when the width-error branch raises on Windows.
                if failure == "width":
                    stack.enter_context(patch("src.reproduction_v2_neural.np.load", return_value=np.zeros((2, 384), np.float32)))
                with self.assertRaisesRegex(ValueError, message):
                    neural_run(root, encoder="roberta")
                for mocked in mocks:
                    mocked.assert_not_called()
                self.assertFalse((root / "neural_roberta/configuration_before_fits.json").exists())

    def test_final_guard_rejects_before_prepare_or_test_feature_encoding(self):
        expected = {"identity": "not the cached development encoder", "generator": "Development generator", "width": "vector width"}
        for failure, message in expected.items():
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp, ExitStack() as stack:
                root = Path(temp)
                self.fixture(root, failure)
                final = root / "final_evaluation"
                final.mkdir()
                (final / "selection_before_unseal.json").write_text(json.dumps({
                    "pairs": {"pre_open": ["news_pca32"], "post_close": ["full_raw"]}, "matched_no_news": False}))
                # Dependency digests precede the cache guard but don't need any
                # real repository artifacts to establish this regression.
                stack.enter_context(patch("src.reproduction_v2_final.sha256_file", return_value="fixture-checksum"))
                mocks = [stack.enter_context(patch(f"src.reproduction_v2_final.{name}", side_effect=AssertionError(f"{name} must not execute")))
                         for name in ("prepare", "load_frozen", "predict_logits", "_cache")]
                if failure == "width":
                    stack.enter_context(patch("src.reproduction_v2_final.np.load", return_value=np.zeros((2, 384), np.float32)))
                with self.assertRaisesRegex(ValueError, message):
                    final_run(root)
                for mocked in mocks:
                    mocked.assert_not_called()
                self.assertFalse((final / "execution_before_unseal.json").exists())
                self.assertFalse((final / "stocknet_unsealed").exists())


if __name__ == "__main__":
    unittest.main()

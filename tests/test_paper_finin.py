"""Cheap equation/masking/gradient checks; no market-training claims."""

import math
import unittest

import torch

from src.models.paper_finin import PaperFININ


class PaperFININTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(912)
        self.batch = {
            "headline_vectors": torch.randn(3, 2, 4, 8),
            "sentiment": torch.randn(3, 2, 4, 3),
            "market_features": torch.randn(3, 2, 6),
            "market_description": torch.randn(8),
            "mask": torch.tensor([[[1, 1, 0, 0], [0, 0, 0, 0]],
                                  [[1, 1, 1, 0], [1, 0, 0, 0]],
                                  [[1, 1, 1, 1], [1, 1, 0, 0]]], dtype=torch.bool),
        }

    def model(self, **kwargs):
        return PaperFININ(text_width=8, width=6, attention_width=4,
                          history_days=2, dropout=0, **kwargs).eval()

    def test_padding_and_empty_days_are_inert_for_all_variants(self):
        for variant in ("full", "no_self_attention", "mean_pool", "news_only"):
            model = self.model(variant=variant)
            changed = {k: v.clone() for k, v in self.batch.items()}
            for key in ("headline_vectors", "sentiment"):
                changed[key][~changed["mask"]] = float("nan")
            original = model.trace(self.batch)
            altered = model.trace(changed)
            torch.testing.assert_close(original["logits"], altered["logits"], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(altered["logits"]).all())
            torch.testing.assert_close(altered["day_news_summaries"][0, 1], torch.zeros(6))
            self.assertEqual(torch.count_nonzero(altered["weights"][~self.batch["mask"]]).item(), 0)
            expected = self.batch["mask"].any(-1).float()
            torch.testing.assert_close(altered["weights"].sum(-1), expected)

    def test_news_permutation_equivariance_and_prediction_invariance(self):
        permutation = torch.tensor([2, 0, 3, 1])
        changed = dict(self.batch)
        for key in ("headline_vectors", "sentiment", "mask"):
            changed[key] = self.batch[key][:, :, permutation]
        for style in ("plain", "residual_norm"):
            model = self.model(attention_style=style)
            first, second = model.trace(self.batch), model.trace(changed)
            torch.testing.assert_close(first["logits"], second["logits"])
            torch.testing.assert_close(first["weights"][:, :, permutation], second["weights"])

    def test_equation6_manual_scores_and_news_only_normalization(self):
        model = self.model()
        traced = model.trace(self.batch)
        scores = torch.einsum("bta,btka->btk", traced["market_query"], traced["news_keys"]) / math.sqrt(4)
        torch.testing.assert_close(scores, traced["query_scores"])
        valid_scores = scores[1, 0, :3]
        torch.testing.assert_close(valid_scores.softmax(0), traced["weights"][1, 0, :3])
        manual_summary = (traced["news_refined"] * traced["weights"][..., None]).sum(2)
        torch.testing.assert_close(manual_summary, traced["day_news_summaries"])

    def test_trace_forward_and_eq7_history_geometry(self):
        model = self.model()
        traced = model.trace(self.batch)
        logits, weights = model(self.batch)
        torch.testing.assert_close(logits, traced["logits"])
        torch.testing.assert_close(weights, traced["weights"])
        self.assertEqual(tuple(traced["mlp_input"].shape), (3, 24))
        torch.testing.assert_close(traced["mlp_input"][:, :12], traced["market_encoded"].flatten(1))
        torch.testing.assert_close(traced["mlp_input"][:, 12:], traced["day_news_summaries"].flatten(1))

    def test_legacy_one_day_geometry_matches_explicit_history(self):
        model = PaperFININ(text_width=8, width=6, attention_width=4, dropout=0).eval()
        one_day = {k: (v[:, :1] if k != "market_description" else v) for k, v in self.batch.items()}
        legacy = {k: (v[:, 0] if k != "market_description" else v) for k, v in one_day.items()}
        a, b = model(one_day), model(legacy)
        torch.testing.assert_close(a[0], b[0])
        torch.testing.assert_close(a[1][:, 0], b[1])

    def test_shared_text_projection_and_gradients_in_active_blocks(self):
        model = self.model()
        traced = model.trace(self.batch)
        torch.testing.assert_close(traced["market_projected_text"], model.text_projection(self.batch["market_description"]).expand(3, 2, 6))
        torch.testing.assert_close(traced["news_projected_text"][self.batch["mask"]], model.text_projection(self.batch["headline_vectors"])[self.batch["mask"]])
        torch.nn.functional.binary_cross_entropy_with_logits(traced["logits"], torch.tensor([1., 0., 1.])).backward()
        modules = (model.text_projection, model.market_numbers, model.news_numbers,
                   model.market_fusion, model.news_fusion, model.self_query,
                   model.self_key, model.self_value, model.self_output,
                   model.market_query, model.news_key, model.predictor)
        for module in modules:
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    def test_news_only_does_not_observe_actual_price_history(self):
        model = self.model(variant="news_only")
        changed = dict(self.batch)
        changed["market_features"] = self.batch["market_features"] * 100 + 20
        traced = model.trace(self.batch)
        torch.testing.assert_close(traced["logits"], model(changed)[0], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(traced["predictor_market"]).item(), 0)
        gradient_batch = dict(self.batch)
        gradient_batch["market_features"] = self.batch["market_features"].clone().requires_grad_()
        model(gradient_batch)[0].sum().backward()
        self.assertIsNone(gradient_batch["market_features"].grad)

    def test_prior_initialization_is_explicit_and_exact(self):
        model = self.model(initial_probability=.61)
        torch.testing.assert_close(model(self.batch)[0].sigmoid(), torch.full((3,), .61))

    def test_invalid_shapes_masks_and_active_nan_fail(self):
        model = self.model()
        for key, value in (("mask", self.batch["mask"].float()),
                           ("market_features", torch.randn(3, 2, 5)),
                           ("market_description", torch.randn(7))):
            with self.assertRaises(ValueError):
                model({**self.batch, key: value})
        corrupted = self.batch["headline_vectors"].clone()
        corrupted[0, 0, 0, 0] = float("nan")
        with self.assertRaises(ValueError):
            model({**self.batch, "headline_vectors": corrupted})


if __name__ == "__main__":
    unittest.main()

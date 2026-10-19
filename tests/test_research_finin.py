"""Controlled attention/actual-age checks, independent of forecasting results."""
import unittest
import torch
from src.models.paper_finin import PaperFININ
from src.models.research_finin import ResearchFININ


class ResearchFININTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(90)
        self.options = dict(text_width=8, width=6, attention_width=4, history_days=2, dropout=0)
        self.batch = dict(headline_vectors=torch.randn(2, 2, 3, 8),
                          sentiment=torch.randn(2, 2, 3, 3),
                          market_features=torch.randn(2, 2, 6), market_description=torch.randn(8),
                          mask=torch.tensor([[[1,1,0],[1,0,0]],[[0,0,0],[0,0,0]]],dtype=torch.bool),
                          headline_age_hours=torch.tensor([[[48.,24.,0.],[0.,0.,0.]],[[0.,0.,0.],[0.,0.,0.]]]))

    def model(self, **extra):
        return ResearchFININ(**self.options, **extra).eval()

    def test_default_matches_paper_including_training_dropout_and_rng(self):
        for style in ('plain','residual_norm'):
            options = {**self.options, 'dropout': .2, 'attention_style':style}
            paper, research = PaperFININ(**options), ResearchFININ(**options)
            research.load_state_dict(paper.state_dict())
            for training in (False,True):
                paper.train(training); research.train(training)
                torch.manual_seed(188); a = paper.trace(self.batch); rng = torch.get_rng_state()
                torch.manual_seed(188); b = research.trace(self.batch)
                for key in a: torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))

    def test_query_switch_retains_identical_refined_news_and_price_history(self):
        full, equal = self.model(), self.model(query_weighting=False)
        equal.load_state_dict(full.state_dict())
        a,b = full.trace(self.batch),equal.trace(self.batch)
        torch.testing.assert_close(a['news_refined'],b['news_refined'],rtol=0,atol=0)
        torch.testing.assert_close(a['predictor_market'],b['predictor_market'],rtol=0,atol=0)
        torch.testing.assert_close(b['weights'][0],torch.tensor([[.5,.5,0.],[1.,0.,0.]]))
        self.assertGreater((a['weights']-b['weights']).abs().max().item(),0)
        independent=self.model(interaction=False)
        independent.load_state_dict(full.state_dict())
        c=independent.trace(self.batch)
        torch.testing.assert_close(c['news_refined'],c['news_encoded'])
        self.assertEqual(c['self_attention_weights'].count_nonzero().item(),0)
        self.assertAlmostEqual(c['weights'][0,0].sum().item(),1,places=6)

    def test_decay_uses_actual_age_ratios_and_same_price_input(self):
        uniform=self.model(query_weighting=False,temporal_aggregation='pooled')
        decay=self.model(query_weighting=False,temporal_aggregation='pooled',half_life_hours=24)
        decay.load_state_dict(uniform.state_dict())
        a,b=uniform.trace(self.batch),decay.trace(self.batch)
        torch.testing.assert_close(a['temporal_news_weights'][0],torch.tensor([[.25,.25,0.],[.5,0.,0.]]))
        # Unnormalized masses: .5*2^-2, .5*2^-1, 1*2^0.
        torch.testing.assert_close(b['temporal_news_weights'][0],torch.tensor([[1/11,2/11,0.],[8/11,0.,0.]]))
        torch.testing.assert_close(a['mlp_input'][:,:12],b['mlp_input'][:,:12],rtol=0,atol=0)
        torch.testing.assert_close(b['temporal_news_weights'].sum((1,2)),torch.tensor([1.,0.]))
        self.assertTrue(torch.isfinite(b['logits']).all())
        torch.testing.assert_close(b['mlp_input'][1,12:],torch.zeros(6))

    def test_large_half_life_recovers_uniform_and_padding_is_inert(self):
        uniform=self.model(temporal_aggregation='pooled')
        decay=self.model(temporal_aggregation='pooled',half_life_hours=1e12)
        decay.load_state_dict(uniform.state_dict())
        torch.testing.assert_close(uniform.trace(self.batch)['temporal_news_weights'],decay.trace(self.batch)['temporal_news_weights'])
        changed={k:v.clone() for k,v in self.batch.items()}
        for key in ('headline_vectors','sentiment','headline_age_hours'): changed[key][~changed['mask']]=float('nan')
        torch.testing.assert_close(decay(self.batch)[0],decay(changed)[0],rtol=0,atol=0)

    def test_bad_ages_and_absorbable_concat_decay_are_rejected(self):
        decay=self.model(temporal_aggregation='pooled',half_life_hours=24)
        for age in (-1.,float('nan'),float('inf')):
            ages=self.batch['headline_age_hours'].clone();ages[0,0,0]=age
            with self.assertRaises(ValueError): decay({**self.batch,'headline_age_hours':ages})
        with self.assertRaises(ValueError): decay({k:v for k,v in self.batch.items() if k!='headline_age_hours'})
        with self.assertRaises(ValueError): self.model(half_life_hours=24)

    def test_channel_ablations_are_invariant_and_pooled_prior_is_honored(self):
        for switch,key in (('sentiment_available','sentiment'),('prices_available','market_features'),('news_available','headline_vectors')):
            model=self.model(**{switch:False})
            changed={**self.batch,key:self.batch[key]*100+5}
            torch.testing.assert_close(model(self.batch)[0],model(changed)[0],rtol=0,atol=0)
        prior=self.model(temporal_aggregation='pooled',initial_probability=.61)
        torch.testing.assert_close(prior(self.batch)[0].sigmoid(),torch.full((2,),.61))

    def test_pooled_decay_gradient_flows_through_both_attention_mechanisms(self):
        model=self.model(temporal_aggregation='pooled',half_life_hours=24)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(self.batch)[0],torch.tensor([1.,0.]))
        loss.backward()
        for module in (model.text_projection,model.self_value,model.market_query,model.news_key,model.predictor):
            gradients=[p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients),0)


if __name__=='__main__': unittest.main()

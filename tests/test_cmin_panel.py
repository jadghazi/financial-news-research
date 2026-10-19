import copy
from datetime import datetime, timezone
import unittest

import numpy as np
import torch

from src.cmin_panel_data import availability, price_channels, rolling_plans
from src.cmin_panel_experiment import PanelDataset, fit_scaler, make_model, score


class CMINPanelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.arrays = {"prices":np.arange(36,dtype=np.float32).reshape(1,6,6),
                       "news_indices":np.array([[[0,1],[0,-1],[1,-1],[2,-1],[3,-1],[3,-1]]]),
                       "descriptions":np.array([4]),"publication_proxy":np.ones((1,6,2))*100,
                       "available_times":np.ones((1,6,2))*200,"close_seconds":np.arange(6)*1000+1000}
        self.text = np.arange(5*768,dtype=np.float32).reshape(5,768)/100
        self.sentiment = np.tile([.2,.5,.3],(5,1)).astype(np.float32)
        self.rows = [{"stock_index":0,"session_index":j,"target_date":f"2020-01-0{j+1}","target":j%2}
                     for j in (2,3,4)]

    def test_availability_keeps_unknown_clock_out(self):
        self.assertEqual(availability("2020-07-03"),datetime(2020,7,5,tzinfo=timezone.utc))
        with self.assertRaises(ValueError): availability("2020-07-03",1)

    def test_price_features_are_independent_of_future_target(self):
        current={"Open":99,"High":102,"Low":98,"Close":100,"Adj Close":50,"Volume":1000}
        previous={"Adj Close":49}
        values=price_channels(current,previous)
        self.assertAlmostEqual(values[0],50/49-1,places=6)
        self.assertAlmostEqual(values[1],-.01,places=6)
        # Uniform price level rescaling leaves all five ratio channels invariant.
        scaled={k:(v*4 if k!="Volume" else v) for k,v in current.items()}
        np.testing.assert_allclose(values,price_channels(scaled,{"Adj Close":196}),atol=1e-6)

    def test_training_scaler_does_not_use_future_rows_or_text(self):
        a=fit_scaler(self.rows,[0],self.arrays,self.text,2)
        changed=copy.deepcopy(self.arrays);changed["prices"][0,3:]+=10000
        text=self.text.copy();text[2:]+=10000
        b=fit_scaler(self.rows,[0],changed,text,2)
        self.assertEqual(a,b)
        self.assertEqual(a["training_text_indices"],[0,1])

    def test_age_ablation_retains_prices_and_removes_only_news(self):
        scaler=fit_scaler(self.rows,[0],self.arrays,self.text,2)
        full=PanelDataset(self.rows,[1],self.arrays,self.text,self.sentiment,scaler,2)[0]
        recent=PanelDataset(self.rows,[1],self.arrays,self.text,self.sentiment,scaler,2,news_age_sessions=(0,1))[0]
        torch.testing.assert_close(full["market_features"],recent["market_features"])
        self.assertFalse(recent["mask"][0].any())
        self.assertTrue(recent["mask"][1].any())

    def test_unavailable_story_is_rejected(self):
        scaler=fit_scaler(self.rows,[0],self.arrays,self.text,2)
        changed=copy.deepcopy(self.arrays);changed["available_times"].fill(1e10)
        with self.assertRaises(ValueError):
            PanelDataset(self.rows,[0],changed,self.text,self.sentiment,scaler,2)[0]

    def test_boundary_labels_are_known_before_fit_cutoff(self):
        dates=["2019-12-27","2019-12-30","2019-12-31","2020-01-02","2020-01-03","2020-05-01","2020-05-04"]
        rows=[{"target_date":d,"cutoff_utc":dates[i-1]+"T21:00:00+00:00" if i else "2019-12-26T21:00:00+00:00",
               "target_available_utc":d+"T21:00:00+00:00"} for i,d in enumerate(dates)]
        p=rolling_plans(rows,[["2020-01-01","2020-05-01","2020-09-01"]])[0]
        self.assertIn(2,p["purged_train"])
        self.assertIn(4,p["purged_valid"])
        self.assertTrue(set(p["train"]).isdisjoint(p["valid"]))

    def batch(self):
        scaler=fit_scaler(self.rows,[0],self.arrays,self.text,2)
        ds=PanelDataset(self.rows,[0,1],self.arrays,self.text,self.sentiment,scaler,2)
        return next(iter(torch.utils.data.DataLoader(ds,batch_size=2)))

    def model(self,arm):
        return make_model({"width":8,"history_sessions":2,"dropout":0.,"attention_style":"plain",
                           "temporal_aggregation":"concat"},arm,None).eval()

    def test_price_only_does_not_depend_on_news_or_sentiment(self):
        batch=self.batch();model=self.model("price_only");original=model(batch)[0]
        batch["headline_vectors"]+=100;batch["sentiment"]*=10
        torch.testing.assert_close(original,model(batch)[0])

    def test_news_only_removes_prices_from_query_and_predictor(self):
        batch=self.batch();model=self.model("news_only");original=model(batch)[0]
        batch["market_features"]+=100
        torch.testing.assert_close(original,model(batch)[0])

    def test_direct_market_ablation_keeps_market_conditioning(self):
        batch=self.batch();model=self.model("finin");original=model.trace(batch)
        restricted=self.model("no_direct_market");restricted.load_state_dict(model.state_dict())
        changed=restricted.trace(batch)
        torch.testing.assert_close(original["query_scores"],changed["query_scores"])
        torch.testing.assert_close(original["day_news_summaries"],changed["day_news_summaries"])
        self.assertEqual(changed["mlp_input"][:,:16].abs().sum(),0)
        torch.testing.assert_close(original["mlp_input"][:,16:],changed["mlp_input"][:,16:])

    def test_separate_attention_switches_and_mlp_trace(self):
        batch=self.batch();full=self.model("finin");trace=full.trace(batch)
        self.assertEqual(trace["mlp_input"].shape,(2,32))
        noquery=self.model("no_query").trace(batch)
        expected=batch["mask"].float()/batch["mask"].sum(-1,keepdim=True).clamp_min(1)
        torch.testing.assert_close(noquery["weights"],expected)
        nointeraction=self.model("no_interaction").trace(batch)
        self.assertEqual(nointeraction["self_attention_weights"].abs().sum(),0)

    def test_decay_requires_pooling_and_finite_age(self):
        from src.models.research_finin import ResearchFININ
        with self.assertRaises(ValueError): ResearchFININ(half_life_hours=24,temporal_aggregation="concat")
        model=ResearchFININ(text_width=768,width=8,attention_width=8,history_days=2,
                            temporal_aggregation="pooled",half_life_hours=24,dropout=0).eval()
        batch=self.batch();trace=model.trace(batch)
        self.assertTrue(torch.isfinite(trace["logits"]).all())
        weights=trace["temporal_news_weights"]
        torch.testing.assert_close(weights.sum((1,2)),torch.ones(2))
        batch["headline_age_hours"][batch["mask"]]=-1
        with self.assertRaises(ValueError):model(batch)

    def test_mcc_and_auc_ties(self):
        result=score([0,1,0,1],[0,0,0,0])
        self.assertEqual(result["mcc"],0)
        self.assertEqual(result["auc"],.5)
        self.assertAlmostEqual(result["log_loss"],np.log(2))

    def test_progress_evidence_preserves_training_and_engine_state(self):
        from src.research_training import fixed_budget_fit
        from torch import nn
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__();self.layer=nn.Sequential(nn.Linear(1,4),nn.Dropout(.2),nn.Linear(4,1))
            def forward(self,batch):
                return self.layer(batch["x"]).squeeze(-1),None
        data=[{"x":torch.tensor([float(i)]),"target":torch.tensor(float(i%2))} for i in range(8)]
        reference=fixed_budget_fit(Tiny,{"train":data,"valid":data},17,epochs=2)
        evidence=[]
        def record(value):
            evidence.append(value["history"]["epoch"])
            torch.rand(5)
            value["history"]["eval"]["valid"]["log_loss"]=-100
            for state in value["selected_state"].values():state.fill_(100)
        observed=fixed_budget_fit(Tiny,{"train":data,"valid":data},17,epochs=2,progress_callback=record)
        self.assertEqual(evidence,[1,2])
        self.assertEqual(reference["history"],observed["history"])
        for name, value in reference["final_state"].items():
            torch.testing.assert_close(value,observed["final_state"][name],atol=0,rtol=0)


if __name__=="__main__": unittest.main()

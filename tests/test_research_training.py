"""Regression coverage for diagnostics perturbing optimization and loss saturation."""
import random,unittest
import numpy as np
import torch
from torch.utils.data import Dataset
from src.models.baselines import NumericBaseline
from src.research_training import isolated_diagnostics,predict_logits,fixed_budget_fit,predict_fitted,fit_train_only
from src.encoder_comparison import metrics

class Rows(Dataset):
    def __init__(self,seed):
        self.x=np.random.default_rng(seed).normal(size=(32,6)).astype(np.float32)
        self.y=(self.x[:,0]>0).astype(np.float32)
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return {'market_features':torch.tensor(self.x[i]),'target':torch.tensor(self.y[i]),'row_index':i}

class ResearchTrainingTest(unittest.TestCase):
    def test_prediction_preserves_training_randomness_and_modes(self):
        model=NumericBaseline(False,width=8,dropout=.3).train();dataset=Rows(21)
        before=torch.get_rng_state().clone();predict_logits(model,dataset,batch_size=7)
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertTrue(model.training)
        self.assertTrue(model.predictor[2].training)

    def test_extra_evaluation_cannot_change_trained_weights(self):
        torch.set_num_threads(4);ds={'train':Rows(21),'valid':Rows(22)}
        factory=lambda:NumericBaseline(False,width=8,dropout=.3)
        first=fixed_budget_fit(factory,ds,17,epochs=3,batch_size=8)
        def extra(model,data):
            predict_logits(model,data['train'],5);predict_logits(model,data['valid'],11)
            torch.rand(27);np.random.rand(7);random.random();model.eval()
        second=fixed_budget_fit(factory,ds,17,epochs=3,batch_size=8,extra_diagnostic=extra)
        for name,value in first['final_state'].items():self.assertTrue(torch.equal(value,second['final_state'][name]),name)
        self.assertEqual(first['history'],second['history']);self.assertEqual(first['selected_epoch'],second['selected_epoch'])

    def test_diagnostic_exception_restores_all_random_streams(self):
        model=NumericBaseline(False,width=8).train();before=torch.get_rng_state().clone()
        py=random.getstate();ns=np.random.get_state()
        with self.assertRaises(RuntimeError):
            with isolated_diagnostics(model):
                torch.rand(10);random.random();np.random.rand(10);model.eval();raise RuntimeError('deliberate failure')
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(py,random.getstate())
        self.assertTrue(np.array_equal(ns[1],np.random.get_state()[1]));self.assertTrue(model.training)

    def test_stable_loss_retains_extreme_probability_errors(self):
        result=metrics(np.array([0,1]),np.array([1000.,-1000.]))
        self.assertEqual(result['log_loss'],1000.);self.assertLess(result['clipped_probability_log_loss'],17.)

    def test_test_split_cannot_be_used_for_checkpoint_selection(self):
        with self.assertRaises(ValueError):fixed_budget_fit(lambda:NumericBaseline(False),{'train':Rows(1),'test':Rows(2)},17,epochs=1)

    def test_returned_model_uses_selected_weights_and_exposes_fallback(self):
        class Signal(torch.nn.Module):
            def __init__(self):super().__init__();self.weight=torch.nn.Parameter(torch.tensor(0.))
            def forward(self,batch):return self.weight*batch['market_features'][:,0],None
        train=Rows(21);valid=Rows(22);valid.y=1-valid.y
        fitted=fixed_budget_fit(Signal,{'train':train,'valid':valid},17,epochs=3,batch_size=8,learning_rate=.1)
        self.assertEqual(fitted['selected_epoch'],1)
        self.assertTrue(torch.equal(fitted['model'].weight,fitted['selected_state']['weight']))
        self.assertFalse(torch.equal(fitted['model'].weight,fitted['final_state']['weight']))
        outputs=predict_fitted(fitted,valid);self.assertTrue(outputs['fallback'])
        self.assertTrue(np.array_equal(outputs['network_logits'],fitted['selected_logits']['valid']))
        self.assertTrue(np.all(outputs['system_logits']==outputs['system_logits'][0]))

    def test_mutating_diagnostic_is_detected_and_weights_restored(self):
        model=NumericBaseline(False,width=8);before={}
        def mutation(model,data):
            before.update({n:v.clone() for n,v in model.state_dict().items()})
            model.predictor[0].weight.add_(1)
        with self.assertRaisesRegex(ValueError,'state restored'):
            fixed_budget_fit(lambda:model,{'train':Rows(21),'valid':Rows(22)},17,epochs=1,extra_diagnostic=mutation)
        for n,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[n]),n)

    def test_fractional_labels_cannot_be_silently_reported_as_directions(self):
        train=Rows(21);train.y[0]=.5
        with self.assertRaisesRegex(ValueError,'binary targets'):
            fixed_budget_fit(lambda:NumericBaseline(False),{'train':train,'valid':Rows(22)},17,epochs=1)

    def test_fixed_epoch_refit_matches_identical_inner_training(self):
        torch.set_num_threads(4);factory=lambda:NumericBaseline(False,width=8,dropout=.3)
        datasets={'train':Rows(21),'valid':Rows(22)}
        inner=fixed_budget_fit(factory,datasets,17,epochs=3,batch_size=8,optimizer_name='adamw',weight_decay=.01)
        refit=fit_train_only(factory,datasets['train'],17,epochs=3,batch_size=8,optimizer_name='adamw',weight_decay=.01)
        for name,value in inner['final_state'].items():
            self.assertTrue(torch.equal(value,refit['final_state'][name]),name)
        self.assertEqual(len(refit['history']),3)

    def test_zero_decay_adamw_matches_adam_and_decay_changes_weights(self):
        factory=lambda:NumericBaseline(False,width=8,dropout=.3);train=Rows(21)
        adam=fit_train_only(factory,train,17,epochs=2,batch_size=8)
        zero=fit_train_only(factory,train,17,epochs=2,batch_size=8,optimizer_name='adamw')
        decay=fit_train_only(factory,train,17,epochs=2,batch_size=8,optimizer_name='adamw',weight_decay=.1)
        for name,value in adam['final_state'].items():
            torch.testing.assert_close(value,zero['final_state'][name],atol=1e-7,rtol=1e-7)
        self.assertFalse(torch.equal(zero['final_state']['predictor.0.weight'],decay['final_state']['predictor.0.weight']))

if __name__=='__main__':unittest.main()

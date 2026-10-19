import unittest
import torch
from src.finin_headline_controls import fit,predict


class ConvexControlsTests(unittest.TestCase):
    def test_known_price_signal_converges_and_news_content_is_not_a_feature(self):
        rows=[dict(market_features=torch.tensor([[float(i%2)*2-1,0,0,0,0,0]]),
                   mask=torch.tensor([[True,False]]),target=torch.tensor(float(i%2)),
                   headline_vectors=torch.ones(1,2,8)*i) for i in range(20)]
        fitted=fit(rows,coverage=True)
        logits=predict(fitted,rows)
        self.assertTrue(all((logits>=0)==[i%2 for i in range(20)]))
        self.assertLess(fitted['max_abs_gradient'],1e-5)
        changed=[{**r,'target':1-r['target'],'headline_vectors':r['headline_vectors']*100} for r in rows]
        self.assertTrue((predict(fitted,changed)==logits).all())
        # Test-set values never update the saved training preprocessing.
        mean=list(fitted['mean']);predict(fitted,[{**r,'market_features':r['market_features']*1000} for r in rows])
        self.assertEqual(fitted['mean'],mean)

    def test_mean_news_control_can_learn_text_with_constant_prices_and_ignores_padding(self):
        rows=[dict(market_features=torch.zeros(3,6),mask=torch.tensor([[True,False]]*3),
                   target=torch.tensor(float(i%2)),sentiment=torch.zeros(3,2,3),
                   headline_vectors=torch.tensor([[[float(i%2)*2-1,0],[999.,999.]]]*3)) for i in range(20)]
        fitted=fit(rows,coverage=True,news=True)
        logits=predict(fitted,rows)
        self.assertTrue(all((logits>=0)==[i%2 for i in range(20)]))
        changed=[]
        for row in rows:
            values=row['headline_vectors'].clone();values[:,1]=float('nan')
            changed.append({**row,'headline_vectors':values,'target':1-row['target']})
        self.assertTrue((predict(fitted,changed)==logits).all())


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
from src.reproduction_v2_probe import partition, controls, fit_lexical, lexical, disruption


class ProbeGuards(unittest.TestCase):
    def test_availability_equality_purged_and_dates_kept_joint(self):
        rows=[]
        for day in range(1,31):
            for ticker in ['AAPL','BA']:
                rows.append({'id':f'{ticker}{day}','ticker':ticker,'target_date':f'2015-06-{day:02}',
                             'cutoff_utc':f'2015-06-{day-1:02}T22:00:00Z',
                             'target_available_utc':f'2015-06-{day:02}T22:00:00Z',
                             'split':'train' if day<26 else 'valid'})
        tr,ev,it,iv=partition(rows)
        self.assertEqual(max(rows[i]['target_date'] for i in tr),'2015-06-24')
        self.assertTrue(all(rows[i]['target_available_utc']<min(rows[j]['cutoff_utc'] for j in iv) for i in it))
        self.assertTrue(set(rows[i]['target_date'] for i in it).isdisjoint(rows[i]['target_date'] for i in iv))

    def test_feature_allowlist_ignores_future_fields(self):
        row={'ticker':'AAPL','market_relative':[0]*6,'headlines':[],
             'raw_news_count':0,'deduplicated_news_count':0,'retained_news_count':0,
             'target_up':0,'target_return':999,'open_up':1,'band_eligible':True}
        a=controls([row]);row.update(target_up=1,target_return=-999,open_up=0,band_eligible=False);b=controls([row])
        for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_lexical_vocabulary_excludes_validation_only_word(self):
        rows=[{'id':str(i),'headlines':[{'text':'earnings grow'}]} for i in range(5)]
        rows.append({'id':'valid','headlines':[{'text':'futureonly earnings'}]})
        fitted=fit_lexical(rows,list(range(5)));self.assertNotIn('futureonly',fitted['vocabulary'])
        x=lexical(rows,fitted);self.assertEqual(x.shape,(6,2));self.assertAlmostEqual(np.linalg.norm(x[-1]),1.)

    def test_disruption_preserves_stock_and_count(self):
        rows=[{'ticker':t,'headlines':[{}]*k} for t in ['AAPL','BA'] for k in [1,2] for _ in range(12)]
        donors=disruption(rows,np.arange(len(rows)))
        for i,j in enumerate(donors):
            self.assertEqual(rows[i]['ticker'],rows[j]['ticker']);self.assertEqual(len(rows[i]['headlines']),len(rows[j]['headlines']))


if __name__=='__main__':unittest.main()

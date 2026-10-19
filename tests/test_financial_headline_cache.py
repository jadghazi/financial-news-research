import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from src.financial_headline_cache import collect


class HeadlineCacheTests(unittest.TestCase):
    def test_content_and_availability_identity_and_exclusive_boundary(self):
        text='Company announces an earnings increase.'
        row=dict(ticker='SP500',description='S&P 500 index',split='valid',target_date='2023-12-29',
                 cutoff_utc='2023-12-28T22:00:00Z',headlines=[dict(text=text,id=hashlib.sha256(text.encode()).hexdigest(),
                 created_at='2023-12-27T12:00:00Z',available_at='2023-12-28T00:00:00Z')])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'provenance.json').write_text(json.dumps(dict(include_test=False,development_target_before='2024-01-01')))
            def check(value):
                (root/'devsamples_post_close.jsonl').write_text(json.dumps(value)+'\n')
                return collect(root)
            items,_=check(row);self.assertEqual(len(items),2)
            for value in ({**row,'target_date':'2024-01-01'},
                          {**row,'headlines':[{**row['headlines'][0],'text':text+' edited'}]},
                          {**row,'headlines':[{**row['headlines'][0],'available_at':'2023-12-29T00:00:00Z'}]}):
                with self.assertRaises(ValueError):check(value)


if __name__=='__main__':unittest.main()

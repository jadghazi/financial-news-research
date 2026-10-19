"""Read existing development predictions/source; no fitting or test access."""
import csv
import json
from pathlib import Path
import re

import numpy as np

from src.reproduction_v2_probe import read_rows, partition, write, evaluated, bootstrap_difference
from src.reproduction_v2_data import sha256_file


def run(out):
    out=Path(out);destination=out/'development_audit.json'
    if destination.exists():raise FileExistsError('Retain previous audit')
    results={}
    for schedule in ['post_close','pre_open']:
        rows=read_rows(out,schedule);tr,ev,_,_=partition(rows)
        traintext={h['id'] for i in tr for h in rows[i]['headlines']}
        validtext={h['id'] for i in ev for h in rows[i]['headlines']}
        texts=[h['text'] for r in rows for h in r['headlines']]
        quote=re.compile(r'\b(price|shares?|stock|trading|market|up|down|bullish|bearish|short|long|buy|sell)\b',re.I)
        summary={'train_count':len(tr),'validation_count':len(ev),'unique_train_message_ids':len(traintext),
                 'unique_validation_message_ids':len(validtext),'repeated_exact_ids_across_train_valid':len(traintext&validtext),
                 'quotation_keyword_fraction':float(np.mean([bool(quote.search(t)) for t in texts])),
                 'keyword_scope':'Loose descriptive keywords, not a classifier or proof of price leakage; tweets may discuss already observed quotes',
                 'train_up_fraction':float(np.mean([rows[i]['target_up'] for i in tr])),
                 'validation_up_fraction':float(np.mean([rows[i]['target_up'] for i in ev])),
                 'news_count_quantiles':np.quantile([r['retained_news_count'] for r in rows],[0,.25,.5,.75,.95,1]).tolist(),
                 'models':{}}
        scores={}
        for name in ['full_raw','mean_raw','full_pca32','mean_pca32','news_pca32','price_identity']:
            path=out/'neural_roberta'/schedule/f'{name}_ensemble_predictions.csv'
            with path.open(newline='') as f:records=list(csv.DictReader(f))
            assert [r['id'] for r in records]==[rows[i]['id'] for i in ev]
            logits=np.array([float(r['logit']) for r in records]);scores[name]=logits
            details={}
            for group,mask in [('empty',np.array([not rows[i]['headlines'] for i in ev])),('with_text',np.array([bool(rows[i]['headlines']) for i in ev]))]:
                indices=ev[mask]
                # Each stock must have both labels for standard per-stock AUC; report aggregate only for subsets.
                from src.encoder_comparison import metrics
                details[group]=metrics([rows[i]['target_up'] for i in indices],logits[mask])
            summary['models'][name]=details
        summary['full_raw_minus_mean_raw']=bootstrap_difference(rows,ev,scores['full_raw'],scores['mean_raw'])
        summary['full_pca_minus_mean_pca']=bootstrap_difference(rows,ev,scores['full_pca32'],scores['mean_pca32'])
        results[schedule]=summary
    write(destination,{'scope':'Post-result development diagnostic, no heldout final access or fitting',
                       'code_sha256':sha256_file(Path(__file__)),'results':results})
    return results


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.out),indent=2))

"""Saved-run diagnosis and bounded frozen-checkpoint sensitivities; never trains."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.evaluation import classification_metrics
from src.training import make_model
from src.training_data import PreparedDataset

SEEDS = (17, 42, 73)
VARIANTS = ('price_only', 'price_sentiment', 'mean_pool', 'no_self_attention', 'no_sentiment', 'full')

def read(p):
    return json.loads((ROOT / p).read_text(encoding='utf-8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def loss(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1-1e-7)
    return -(y * np.log(p) + (1-y) * np.log(1-p))

def metrics(y, p):
    result = classification_metrics(list(y), list(p))
    p = np.asarray(p)
    result.update(probability_mean=float(p.mean()), probability_std=float(p.std()),
                  probability_min=float(p.min()), probability_max=float(p.max()),
                  predicted_down=int((p < .5).sum()), brier=float(np.mean((p-y)**2)))
    pos, neg = p[y == 1], p[y == 0]
    result['auc'] = float(np.mean((pos[:,None]>neg[None,:]) + .5*(pos[:,None]==neg[None,:])))
    return result

def interval(delta, block):
    rng = np.random.default_rng(20261011)
    n = len(delta)
    starts = rng.integers(0, n-block+1, size=(5000, math.ceil(n/block)))
    indices = (starts[:,:,None]+np.arange(block)).reshape(5000,-1)[:,:n]
    q = np.quantile(np.asarray(delta)[indices].mean(axis=1), [.025,.975])
    return dict(mean=float(np.mean(delta)), lower=float(q[0]), upper=float(q[1]))

@torch.inference_mode()
def evaluate(model, dataset, *, refs=None, mode='original', permutation=None, sentiment_mean=None):
    probabilities, markets, summaries, attention = [], [], [], []
    for batch in DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0):
        if permutation is not None:
            donors = [dataset[int(permutation[int(i)])] for i in batch['row_index']]
            for key in ('headline_vectors','sentiment','mask'):
                batch[key] = torch.stack([row[key] for row in donors])
        if mode == 'sentiment_train_mean':
            batch['sentiment'] = sentiment_mean.view(1,1,3).expand_as(batch['sentiment'])
        if not hasattr(model, 'encoders'):
            logits, _ = model(batch)
        else:
            market, news = model.encoders(batch['market_description'], batch['market_features'],
                         batch['headline_vectors'], batch['sentiment'], use_sentiment=model.variant!='no_sentiment')
            summary, weights = model.attention(market, news, batch['mask'],
                         use_self_attention=model.variant not in ('mean_pool','no_self_attention'),
                         use_market_attention=model.variant!='mean_pool' and mode!='uniform_market_weights')
            markets.append(market.numpy()); summaries.append(summary.numpy())
            attention.extend([w[m].numpy() for w,m in zip(weights,batch['mask'])])
            if mode == 'news_summary_train_mean':
                summary = refs['news'].expand_as(summary)
            if mode == 'direct_market_train_mean':
                market = refs['market'].expand_as(market)
            logits = model.predictor(torch.cat((market,summary),dim=-1)).squeeze(-1)
        probabilities.extend(torch.sigmoid(logits).tolist())
    return dict(probabilities=np.asarray(probabilities),
                markets=np.concatenate(markets) if markets else None,
                summaries=np.concatenate(summaries) if summaries else None,
                attention=attention)

def main():
    if (HERE/'diagnostics.json').exists():
        raise RuntimeError('Existing diagnosis preserved; choose a new directory to rerun.')
    torch.set_num_threads(4)
    torch.manual_seed(0)
    provenance = {}
    for folder in ('src','data/processed','artifacts/features','artifacts/runs'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                provenance[str(p.relative_to(ROOT))] = sha(p)
    for p in (HERE/'diagnose.py', HERE/'PROTOCOL.md', ROOT/'data/nifty_manifest.json', ROOT/'data/spy_manifest.json'):
        provenance[str(p.relative_to(ROOT))] = sha(p)
    datasets = {s:PreparedDataset(ROOT,s) for s in ('train','valid','test')}
    targets = {s:np.asarray([r['target_up'] for r in d.rows]) for s,d in datasets.items()}
    y = targets['test']; prior = float(targets['train'].mean())
    saved = read('artifacts/runs/all_runs.json')['runs']
    assert len(saved)==19
    keyset = {(r['variant'],r.get('seed')) for r in saved}
    assert keyset == {('always_up',None)}|{(v,s) for v in VARIANTS for s in SEEDS}
    summary = {'scope':'Retrospective exploratory diagnosis; no fitting or selection',
       'created_utc':datetime.now(timezone.utc).isoformat(),
       'environment':dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=4),
       'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
       'git_status_before_outputs':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
       'source_hashes':provenance,'split_counts':{s:len(t) for s,t in targets.items()},
       'split_up_prevalence':{s:float(t.mean()) for s,t in targets.items()},
       'constant_reference':{s:metrics(t,np.repeat(prior,len(t))) for s,t in targets.items()},
       'runs':[], 'full_checkpoints':[], 'bootstrap':{}, 'changed_dates':[]}
    by = {}
    predictions_csv = []
    for r in saved:
        stem = r['variant'] if r.get('seed') is None else f"{r['variant']}_{r['seed']}"
        assert r == read(f'artifacts/runs/{stem}.json')
        assert len(r['predictions'])==len(y)
        for p,row in zip(r['predictions'],datasets['test'].rows):
            assert all(p[k]==row[k] for k in ('news_date','forecast_date','target_date','target_up'))
        p = np.asarray([a['probability_up'] for a in r['predictions']]); by[(r['variant'],r.get('seed'))]=p
        m = metrics(y,p)
        for k in ('accuracy','balanced_accuracy','log_loss','up_recall','down_recall'):
            assert abs(m[k]-r['test_metrics'][k])<1e-12
        assert m['confusion']==r['test_metrics']['confusion']
        record = dict(variant=r['variant'], seed=r.get('seed'), metrics=m)
        if r.get('seed') is not None:
            model = make_model(r['variant'])
            model.load_state_dict(torch.load(ROOT/f'artifacts/runs/{stem}.pt',map_location='cpu',weights_only=True));model.eval()
            replay = evaluate(model,datasets['test'])
            record['replay_max_error']=float(np.max(np.abs(replay['probabilities']-p)))
            assert record['replay_max_error']<1e-6
            record.update(best_epoch=r['best_epoch'], epochs_completed=r['epochs_completed'],
                          best_validation_loss=r['best_validation_loss'], history=r['history'],
                          stored_parameter_count=sum(v.numel() for v in model.parameters()))
            best=math.inf;epoch=0
            for h in r['history']:
                if h['valid_loss']<best-1e-5:best=h['valid_loss'];epoch=h['epoch']
            assert epoch==r['best_epoch'] and best==r['best_validation_loss']
        summary['runs'].append(record)
    for seed in SEEDS:
        print(f'Frozen full checkpoint {seed}: train/validation and prespecified sensitivities',flush=True)
        model=make_model('full');model.load_state_dict(torch.load(ROOT/f'artifacts/runs/full_{seed}.pt',map_location='cpu',weights_only=True));model.eval()
        train=evaluate(model,datasets['train'])
        refs={ 'market':torch.from_numpy(train['markets'].mean(axis=0)),
               'news':torch.from_numpy(train['summaries'].mean(axis=0)) }
        # Training headline-slot mean only; real headline masks exclude padding.
        sums=np.zeros(3);count=0
        for b in DataLoader(datasets['train'],batch_size=64,shuffle=False):
            sums+=(b['sentiment']*b['mask'].unsqueeze(-1)).sum(dim=(0,1)).numpy();count+=int(b['mask'].sum())
        sentiment_mean=torch.tensor(sums/count,dtype=torch.float32)
        detail={'seed':seed,'reference_fitted_on':'training headline slots and original train representations only',
                'splits':{}}
        for split,dataset in datasets.items():
            original=train if split=='train' else evaluate(model,dataset)
            p=original['probabilities'];t=targets[split];c=metrics(t,p)
            if split=='test':assert np.max(np.abs(p-by[('full',seed)]))<1e-6
            if split=='valid':assert abs(c['log_loss']-next(r for r in saved if r['variant']=='full' and r['seed']==seed)['best_validation_loss'])<1e-6
            ws=original['attention']; uniform_gap=[float(np.max(w)-1/len(w)) for w in ws]
            entropy=[float(-np.sum(w*np.log(np.clip(w,1e-12,1)))/np.log(len(w))) for w in ws if len(w)>1]
            state={'original':c,'attention':{'median_max_weight_excess_over_uniform':float(np.median(uniform_gap)),
                 'max_weight_excess_over_uniform':float(max(uniform_gap)),
                 'median_normalized_entropy':float(np.median(entropy)), 'min_normalized_entropy':float(min(entropy))},
                 'sensitivities':{}}
            modes=('news_summary_train_mean','direct_market_train_mean','uniform_market_weights','sentiment_train_mean')
            treatments={name:(name,None) for name in modes}
            for n in (101,202,303):
                treatments[f'permuted_news_{n}']=('permuted_news',np.random.default_rng(n).permutation(len(dataset)))
            for name,(mode,perm) in treatments.items():
                v=evaluate(model,dataset,refs=refs,mode=mode,permutation=perm,sentiment_mean=sentiment_mean)['probabilities']
                delta=v-p
                state['sensitivities'][name]={'metrics':metrics(t,v),'mean_abs_probability_change':float(np.mean(np.abs(delta))),
                    'max_abs_probability_change':float(np.max(np.abs(delta))),
                    'decision_changes':int(((v>=.5)!=(p>=.5)).sum()),
                    'log_loss_change':float(loss(t,v).mean()-loss(t,p).mean())}
                for i,row in enumerate(dataset.rows):
                    predictions_csv.append(dict(split=split,seed=seed,condition=name,news_date=row['news_date'],
                       forecast_date=row['forecast_date'],target_date=row['target_date'],target_up=int(t[i]),
                       original_probability=float(p[i]),diagnostic_probability=float(v[i]),
                       donor_news_date=dataset.rows[int(perm[i])]['news_date'] if perm is not None else ''))
            detail['splits'][split]=state
        summary['full_checkpoints'].append(detail)
    full=np.stack([by[('full',s)] for s in SEEDS])
    comparators={'always_up':np.ones_like(full),'training_prior':np.full_like(full,prior)}
    for v in VARIANTS:
        if v!='full':comparators[v]=np.stack([by[(v,s)] for s in SEEDS])
    for name,p in comparators.items():
        acc=((full>=.5)==y).mean(axis=0)-((p>=.5)==y).mean(axis=0)
        losses=np.mean([loss(y,f)-loss(y,c) for f,c in zip(full,p)],axis=0)
        summary['bootstrap'][name]={str(b):{'accuracy_full_minus_reference':interval(acc,b),
                 'log_loss_full_minus_reference':interval(losses,b)} for b in (5,10,20)}
    for seed,p in zip(SEEDS,full):
        for i in np.where(p<.5)[0]:
            row=datasets['test'].rows[i]
            effect=1 if y[i]==0 else -1
            controls={x['condition']:x['diagnostic_probability'] for x in predictions_csv
                      if x['seed']==seed and x['split']=='test' and x['forecast_date']==row['forecast_date']}
            summary['changed_dates'].append(dict(seed=seed,news_date=row['news_date'],forecast_date=row['forecast_date'],
                 target_date=row['target_date'],target_up=int(y[i]),probability_up=float(p[i]),
                 net_correct_change_vs_always_up=effect,market_features_raw=row['market_features_raw'],
                 control_probabilities=controls,
                 price_only_same_seed_probability=float(by[('price_only',seed)][i])))
    summary['variant_means']={v:{k:float(np.mean([r['metrics'][k] for r in summary['runs'] if r['variant']==v]))
           for k in ('accuracy','balanced_accuracy','log_loss','brier','auc','predicted_down')}
           for v in (*VARIANTS,'always_up')}
    (HERE/'diagnostics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (HERE/'sensitivity_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(predictions_csv[0]));writer.writeheader();writer.writerows(predictions_csv)
    with (HERE/'changed_decisions.csv').open('w',newline='',encoding='utf-8') as f:
        cols=['seed','news_date','forecast_date','target_date','target_up','probability_up','net_correct_change_vs_always_up','price_only_same_seed_probability']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows({k:r[k] for k in cols} for r in summary['changed_dates'])
    print(json.dumps({'variant_means':summary['variant_means'],'changed_dates':summary['changed_dates'],
        'full_sensitivities':[{ 'seed':r['seed'],'test':r['splits']['test'],'train_loss':r['splits']['train']['original']['log_loss'],
              'valid_loss':r['splits']['valid']['original']['log_loss']} for r in summary['full_checkpoints']],
        'uncertainty':{k:v['10'] for k,v in summary['bootstrap'].items() if k in ('always_up','training_prior','mean_pool','price_only')}}))

if __name__=='__main__':main()

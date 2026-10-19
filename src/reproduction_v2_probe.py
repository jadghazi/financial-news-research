"""Versioned chronological convex probes; explicit features, no test access."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import torch

from src.encoder_comparison import metrics
from src.pipeline_diagnosis import linear_predict
from src.reproduction_v2_data import TICKER_DESCRIPTIONS, sha256_file

PENALTIES = [.1, .01, .001, .0001]
TICKERS = list(TICKER_DESCRIPTIONS)


def fit_linear(x,y,penalty):
    # Same objective as earlier probes; expanded solver budget, not new selection.
    start=time.monotonic();mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-8,1.,scale)
    z=torch.tensor((x-mean)/scale,dtype=torch.float64);labels=torch.tensor(y,dtype=torch.float64)
    model=torch.nn.Linear(x.shape[1],1,dtype=torch.float64)
    with torch.no_grad():model.weight.zero_();model.bias.fill_(np.log(y.mean()/(1-y.mean())))
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=2000,tolerance_grad=1e-9,tolerance_change=1e-14,line_search_fn='strong_wolfe');calls=0
    def closure():
        nonlocal calls
        calls+=1;optimizer.zero_grad(set_to_none=True);logit=model(z).squeeze(1)
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,labels)+penalty*model.weight.square().sum()/2
        if not torch.isfinite(loss):raise ValueError('Nonfinite convex objective')
        loss.backward();return loss
    optimizer.step(closure);objective=closure();gradient=max(float(p.grad.abs().max()) for p in model.parameters())
    result={'mean':mean.tolist(),'scale':scale.tolist(),'weight':model.weight.detach().numpy()[0].tolist(),'bias':float(model.bias.detach()[0]),
            'penalty':penalty,'objective':float(objective.detach()),'gradient_max':gradient,'iterations':int(optimizer.state[model.weight].get('n_iter',0)),
            'closure_calls':calls,'seconds':time.monotonic()-start,'converged':gradient<=1e-5,'max_iter':2000}
    if not result['converged']:raise ValueError(f'Convex fit failed: {gradient}')
    return result


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def read_rows(out, schedule):
    rows = [json.loads(line) for line in (out / 'stocknet_dev' / f'devsamples_{schedule}.jsonl').read_text(encoding='utf-8').splitlines()]
    if any(r['split'] not in {'train', 'valid'} or r['target_date'] >= '2015-10-01' for r in rows):
        raise ValueError('Development only')
    return rows


def partition(rows):
    evaluation = np.array([i for i,r in enumerate(rows) if r['split']=='valid'])
    cutoff = min(rows[i]['cutoff_utc'] for i in evaluation)
    training = np.array([i for i,r in enumerate(rows) if r['split']=='train' and r['target_available_utc'] < cutoff])
    dates = sorted({rows[i]['target_date'] for i in training})
    split_date = dates[int(.8*len(dates))]
    inner_eval = np.array([i for i in training if rows[i]['target_date']>=split_date])
    inner_cutoff = min(rows[i]['cutoff_utc'] for i in inner_eval)
    inner_train = np.array([i for i in training if rows[i]['target_date']<split_date and rows[i]['target_available_utc']<inner_cutoff])
    for fit,ev in [(training,evaluation),(inner_train,inner_eval)]:
        if len(fit)==0 or any(rows[i]['target_available_utc']>=min(rows[j]['cutoff_utc'] for j in ev) for i in fit):
            raise ValueError('Label availability violation')
    return training, evaluation, inner_train, inner_eval


def controls(rows):
    # Explicit allowlist: ticker, base price-relative features, message counts.
    identity=np.eye(len(TICKERS))[[TICKERS.index(r['ticker']) for r in rows]]
    coverage=np.array([[np.log1p(r['raw_news_count']),np.log1p(r['deduplicated_news_count']),np.log1p(r['retained_news_count']),int(not r['headlines'])] for r in rows])
    prices=np.array([r['market_relative'] for r in rows])
    return identity,coverage,prices


def tokens(row):
    return re.findall(r'[a-z][a-z0-9_]{1,}', ' '.join(h['text'] for h in row['headlines']).lower())


def fit_lexical(rows, indices, max_features=1000, min_df=5):
    df=Counter()
    for i in indices: df.update(set(tokens(rows[i])))
    vocab=sorted((t for t,n in df.items() if n>=min_df),key=lambda t:(-df[t],t))[:max_features]
    idf=np.array([np.log((1+len(indices))/(1+df[t]))+1 for t in vocab])
    return {'vocabulary':vocab,'idf':idf.tolist(),'fit_ids':[rows[i]['id'] for i in indices],
            'definition':'day-document sublinear TF, smoothed train-document IDF, L2 normalization; max1000 minDF5'}


def lexical(rows, fitted):
    mapping={t:j for j,t in enumerate(fitted['vocabulary'])};idf=np.array(fitted['idf']);x=np.zeros((len(rows),len(mapping)))
    for i,r in enumerate(rows):
        for t,n in Counter(tokens(r)).items():
            if t in mapping:x[i,mapping[t]]=(1+np.log(n))*idf[mapping[t]]
    return x/np.linalg.norm(x,axis=1,keepdims=True).clip(min=1e-12)


def prediction_file(path, rows, indices, logits):
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=['id','ticker','target_date','cutoff_utc','target_up','logit']);w.writeheader()
        for i,l in zip(indices,logits):w.writerow({k:rows[i][k] for k in w.fieldnames if k!='logit'}|{'logit':float(l)})


def evaluated(rows, indices, logits):
    y=np.array([rows[i]['target_up'] for i in indices]);m=metrics(y,logits)
    by_ticker={t:metrics(y[[rows[i]['ticker']==t for i in indices]],np.asarray(logits)[[rows[i]['ticker']==t for i in indices]]) for t in TICKERS}
    m['mean_within_ticker_auc']=float(np.mean([v['auc'] for v in by_ticker.values()]));m['by_ticker']=by_ticker
    return m


def prior_logits(rows,fit,ev,stock_specific=False):
    y=np.array([rows[i]['target_up'] for i in fit]);global_prior=(y.sum()+1)/(len(y)+2)
    p={t:(sum(rows[i]['target_up'] for i in fit if rows[i]['ticker']==t)+20*global_prior)/(sum(rows[i]['ticker']==t for i in fit)+20) for t in TICKERS}
    probabilities=np.array([p[rows[i]['ticker']] if stock_specific else global_prior for i in ev])
    return np.log(probabilities/(1-probabilities)),{'global_prior':float(global_prior),'stock_priors':p,'shrinkage_pseudocount':20}


def disruption(rows, evaluation):
    # Fixed count/ticker matched within-validation permutation, no label use.
    rng=np.random.default_rng(66101);donors=np.arange(len(rows));groups={}
    for i in evaluation:groups.setdefault((rows[i]['ticker'],len(rows[i]['headlines'])),[]).append(i)
    for group in groups.values():donors[group]=rng.permutation(group)
    return donors


def bootstrap_difference(rows,indices,first,second):
    y=np.array([rows[i]['target_up'] for i in indices]);a=np.logaddexp(0,first)-y*first;b=np.logaddexp(0,second)-y*second
    dates=sorted({rows[i]['target_date'] for i in indices})
    per_date=np.array([(a-b)[[rows[i]['target_date']==d for i in indices]].mean() for d in dates])
    # Five-session moving blocks, all tickers jointly, circular wrap declared.
    rng=np.random.default_rng(71013);vals=[]
    for _ in range(2000):
        starts=rng.integers(len(dates),size=int(np.ceil(len(dates)/5)))
        sample=((starts[:,None]+np.arange(5))%len(dates)).ravel()[:len(dates)]
        vals.append(per_date[sample].mean())
    return {'bce_difference':float(per_date.mean()),'ci95':np.quantile(vals,[.025,.975]).tolist(),'dates':len(dates),'block_length':5,'resamples':2000,'definition':'first minus second; negative favors first; circular date-block descriptive CI, not multiple-search correction'}


def run(out, representation='lexical'):
    out=Path(out);directory=out/f'probe_{representation}_solver2'
    if directory.exists():raise FileExistsError('Preserve previous probe; use new namespace')
    directory.mkdir();torch.set_num_threads(4);started=time.monotonic()
    config={'representation':representation,'penalties':PENALTIES,'solver_max_iter':2000,'solver_gradient_gate':1e-5,'models':['identity','coverage','prices','news','prices_news'],
            'objective':'mean BCE + lambda/2*sum standardized weights squared; unpenalized intercept',
            'inner_selection':'last20% train target dates; strict label availability purge; minimum BCE, strongest exact tie',
            'feature_allowlist':['ticker','market_relative','raw_news_count','deduplicated_news_count','retained_news_count','headlines.text/id'],
            'source_sha256':{s:sha256_file(out/'stocknet_dev'/f'devsamples_{s}.jsonl') for s in ['post_close','pre_open']},
            'code_sha256':sha256_file(Path(__file__)),'protocol_sha256':sha256_file(out/'PROTOCOL.md')}
    write(directory/'configuration_before_fits.json',config);all_results={}
    features=None
    if representation!='lexical':
        from src.reproduction_v2_features import FrozenFeatures
        features=FrozenFeatures(out/f'{representation}_cache')
    for schedule in ['post_close','pre_open']:
        rows=read_rows(out,schedule);tr,ev,it,iv=partition(rows);y=np.array([r['target_up'] for r in rows],float)
        identity,coverage,prices=controls(rows);base=np.column_stack([identity,coverage]);folder=directory/schedule;folder.mkdir()
        write(folder/'split.json',{k:v.tolist() for k,v in zip(['train','valid','inner_train','inner_valid'],[tr,ev,it,iv])})
        if representation=='lexical':
            inner_repr=fit_lexical(rows,it);outer_repr=fit_lexical(rows,tr)
            nx_inner=lexical(rows,inner_repr);nx_outer=lexical(rows,outer_repr)
            write(folder/'inner_representation.json',inner_repr);write(folder/'outer_representation.json',outer_repr)
        else:
            means=[features.daily_mean(r) for r in rows]
            nx_outer=nx_inner=np.column_stack([np.array([x[0] for x in means]),np.array([x[1] for x in means])])
        results={};predictions={}
        for name,specific in [('global_prior',False),('stock_prior',True)]:
            logits,state=prior_logits(rows,tr,ev,specific);prediction_file(folder/f'{name}_predictions.csv',rows,ev,logits)
            results[name]={'valid':evaluated(rows,ev,logits),'state':state};predictions[name]=logits
        for name in config['models']:
            modeldir=folder/name;modeldir.mkdir();candidates=[];best=float('inf');selected=None
            def design(news):
                return {'identity':identity,'coverage':base,'prices':np.column_stack([base,prices]),
                        'news':np.column_stack([base,news]),'prices_news':np.column_stack([base,prices,news])}[name]
            ix=design(nx_inner);ox=design(nx_outer)
            for penalty in PENALTIES:
                model=fit_linear(ix[it],y[it],penalty);l=linear_predict(model,ix[iv]);m=evaluated(rows,iv,l)
                write(modeldir/f'inner_model_{penalty}.json',model);prediction_file(modeldir/f'inner_predictions_{penalty}.csv',rows,iv,l)
                candidates.append({'penalty':penalty,'metrics':m})
                if m['log_loss']<best:best=m['log_loss'];selected=penalty
            model=fit_linear(ox[tr],y[tr],selected);l=linear_predict(model,ox[ev]);prediction_file(modeldir/'valid_predictions.csv',rows,ev,l)
            write(modeldir/'model.json',model);summary={'selected_penalty':selected,'inner_candidates':candidates,'train':evaluated(rows,tr,linear_predict(model,ox[tr])),'valid':evaluated(rows,ev,l),'feature_count':ox.shape[1]}
            if name in ['news','prices_news']:
                donor=disruption(rows,ev);altered=design(nx_outer[donor]);changed=linear_predict(model,altered[ev]);prediction_file(modeldir/'text_permuted_predictions.csv',rows,ev,changed)
                write(modeldir/'disruption.json',{'donors':donor[ev].tolist(),'changed_donors':int((donor[ev]!=ev).sum())})
                summary['text_permuted']=evaluated(rows,ev,changed);summary['decisions_changed']=int(((l>=0)!=(changed>=0)).sum());summary['permutation_loss_change_ci']=bootstrap_difference(rows,ev,changed,l)
            write(modeldir/'summary.json',summary);results[name]=summary;predictions[name]=l
            print(f'{representation} {schedule} {name}: BCE {summary["valid"]["log_loss"]:.6f} AUC {summary["valid"]["auc"]:.3f} down {summary["valid"]["predicted_down"]}/{len(ev)}',flush=True)
        results['news_minus_coverage']=bootstrap_difference(rows,ev,predictions['news'],predictions['coverage'])
        results['prices_news_minus_prices']=bootstrap_difference(rows,ev,predictions['prices_news'],predictions['prices'])
        write(folder/'results.json',results);all_results[schedule]=results
    if features:features.close()
    write(directory/'results.json',{'configuration':config,'results':all_results,'seconds':time.monotonic()-started})
    return all_results


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--representation',choices=['lexical','bge','roberta'],default='lexical');args=parser.parse_args();run(args.out,args.representation)

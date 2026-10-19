"""Bounded development fits for the new inspectable FININ adaptation."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.models.paper_finin import PaperFININ
from src.research_training import fixed_budget_fit, predict_logits
from src.reproduction_v2_features import ENCODERS, FrozenFeatures, fit_pca, apply_pca
from src.reproduction_v2_probe import (write, read_rows, partition, controls, prediction_file,
                                       evaluated, disruption, bootstrap_difference, prior_logits)
from src.reproduction_v2_data import sha256_file

SEEDS=[17,42,73]
# Frozen before first neural financial fit. No adaptive architecture search.
CANDIDATES=[
    {'name':'full_raw','variant':'full','representation':'raw','attention_style':'plain','market_input':'raw'},
    {'name':'mean_raw','variant':'mean_pool','representation':'raw','attention_style':'plain','market_input':'raw'},
    {'name':'full_pca32','variant':'full','representation':'pca32','attention_style':'plain','market_input':'raw'},
    {'name':'mean_pca32','variant':'mean_pool','representation':'pca32','attention_style':'plain','market_input':'raw'},
    {'name':'news_pca32','variant':'news_only','representation':'pca32','attention_style':'plain','market_input':'raw'},
    {'name':'price_identity','variant':'price_only','representation':'raw','attention_style':'plain','market_input':'raw'},
]


class Samples(Dataset):
    def __init__(self,rows,indices,features,market,description):
        self.rows=[rows[i] for i in indices]
        self.market=torch.tensor(market[indices],dtype=torch.float32)
        width=features.text.shape[1];n=len(indices)
        text=np.zeros((n,64,width),np.float32);sent=np.zeros((n,64,3),np.float32);mask=np.zeros((n,64),bool)
        descriptions=[]
        for j,r in enumerate(self.rows):
            ids=[h['id'] for h in r['headlines']];k=len(ids)
            if k:text[j,:k]=features.vectors(ids);sent[j,:k]=features.sentiments(ids);mask[j,:k]=True
            descriptions.append(description[r['ticker']])
        self.text=torch.from_numpy(text);self.sent=torch.from_numpy(sent);self.mask=torch.from_numpy(mask)
        self.descriptions=torch.tensor(np.array(descriptions),dtype=torch.float32)
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        return {'headline_vectors':self.text[i],'sentiment':self.sent[i],'mask':self.mask[i],
                'market_features':self.market[i], 'market_description':self.descriptions[i],
                'target':torch.tensor(self.rows[i]['target_up'],dtype=torch.float32),'row_index':i}


class ProjectedFeatures:
    def __init__(self,source,values):self.source=source;self.text=values
    def vectors(self,ids):return self.text[[self.source.index[i] for i in ids]]
    def sentiments(self,ids):return self.source.sentiments(ids)


class PriceIdentity(nn.Module):
    """Same stock-description projection and numeric encoder; no news input."""
    def __init__(self,text_width,width=32,dropout=.1):
        super().__init__();self.text=nn.Linear(text_width,width)
        self.numeric=nn.Sequential(nn.Linear(6,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,width))
        self.predictor=nn.Sequential(nn.Linear(2*width,width),nn.ReLU(),nn.Dropout(dropout),nn.Linear(width,1))
    def forward(self,batch):
        values=torch.cat([self.text(batch['market_description']),self.numeric(batch['market_features'])],-1)
        return self.predictor(values).squeeze(-1),None


def run(out,encoder='bge'):
    out=Path(out);directory=out/f'neural_{encoder}'
    if not json.loads((out/'gates'/'summary.json').read_text())['all_positive_gates_pass']:
        raise ValueError('Known-signal gate failed; diagnose before financial neural fits')
    directory.mkdir(exist_ok=True);torch.set_num_threads(4)
    cache_manifest=json.loads((out/f'{encoder}_cache'/'manifest.json').read_text())
    if cache_manifest['identity']['encoder']!=ENCODERS[encoder]:
        raise ValueError('Named experiment encoder differs from actual cached model')
    actual_generator=cache_manifest['states']['text']['model_provenance']
    if (actual_generator['model'],actual_generator['revision'])!=(ENCODERS[encoder]['name'],ENCODERS[encoder]['revision']):
        raise ValueError('Cache generator does not match requested pinned encoder')
    if np.load(out/f'{encoder}_cache'/'text.npy',mmap_mode='r').shape[1]!=ENCODERS[encoder]['width']:
        raise ValueError('Actual cache width differs from requested encoder')
    config={'candidates':CANDIDATES,'seeds':SEEDS,'epochs':30,'batch_size':64,'learning_rate':.001,
            'optimizer':'adamw','weight_decay':.01,'clip':1.,'width':32,'dropout':.1,
            'initialization':'ordinary default PyTorch; no prior-head or weighted loss',
            'market':'train-only standardization of raw open/high/low/close/adjusted-close/volume; matched across models, relative-channel comparison stays in convex probes',
            'text':'raw CLS OR train-unique-message PCA32, centered projected coordinates scaled by train std; descriptions use same transform',
            'selection':'minimum validation raw-logit BCE; report all seeds/candidates; no test data',
            'feature_allowlist':['ticker description','market_raw','headlines.id','headline masks'],
            'protocol_sha256':sha256_file(out/'PROTOCOL.md'),'source_sha256':sha256_file(Path(__file__)),
            'cache_manifest_sha256':sha256_file(out/f'{encoder}_cache'/'manifest.json'),
            'dependency_sha256':{str(p):sha256_file(p) for p in map(Path,['src/models/paper_finin.py','src/research_training.py','src/reproduction_v2_probe.py','src/reproduction_v2_data.py','src/reproduction_v2_features.py','src/encoder_comparison.py','src/training.py','src/evaluation.py'])},
            'dev_source_sha256':{name:sha256_file(out/'stocknet_dev'/name) for name in ['devsamples_post_close.jsonl','devsamples_pre_open.jsonl','provenance.json']},
            'environment':{'python':__import__('platform').python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':4,'device':'cpu'}}
    identity_path=directory/'configuration_before_fits.json'
    if identity_path.exists():
        if json.loads(identity_path.read_text())!=config:raise ValueError('Immutable neural execution identity changed')
    else:write(identity_path,config)
    if (directory/'results.json').exists():return json.loads((directory/'results.json').read_text())
    features=FrozenFeatures(out/f'{encoder}_cache')
    if features.text.shape[1]!=ENCODERS[encoder]['width']:
        raise ValueError('Cached vector width disagrees with requested encoder')
    if features.manifest['identity']['source_sha256']!=config['dev_source_sha256']:
        raise ValueError('Cached text provenance differs from current samples')
    results={};start=time.monotonic()
    for schedule in ['post_close','pre_open']:
        rows=read_rows(out,schedule);tr,ev,_,_=partition(rows);folder=directory/schedule;folder.mkdir(exist_ok=True)
        market=np.array([r['market_raw'] for r in rows]);mean=market[tr].mean(0);scale=market[tr].std(0);scale=np.where(scale<1e-8,1.,scale);market=(market-mean)/scale
        fitids=sorted({features.index[h['id']] for i in tr for h in rows[i]['headlines']})
        transform=fit_pca(features.text,fitids,32);projected=apply_pca(features.text,transform)
        pca_scale=projected[fitids].std(0).clip(min=1e-8);projected=projected/pca_scale
        pca_path=folder/'pca_transform.npz'
        if pca_path.exists():
            old=np.load(pca_path)
            for key,value in [('mean',transform['mean']),('basis',transform['basis']),('scale',pca_scale),('fit_indices',fitids)]:
                np.testing.assert_array_equal(old[key],value)
        else:np.savez(pca_path,mean=transform['mean'],basis=transform['basis'],scale=pca_scale,fit_indices=fitids)
        preprocessing={'market_mean':mean.tolist(),'market_scale':scale.tolist(),
              'pca_fit_ids':[features.ids[i] for i in fitids],'fit_rows_sha256':transform['fit_rows_sha256'],
              'training_indices':tr.tolist(),'validation_indices':ev.tolist(),'pca_transform_sha256':sha256_file(pca_path)}
        if (folder/'preprocessing.json').exists():
            if json.loads((folder/'preprocessing.json').read_text())!=preprocessing:raise ValueError('Resume preprocessing changed')
        else:write(folder/'preprocessing.json',preprocessing)
        datasets={}
        for representation,source in [('raw',features),('pca32',ProjectedFeatures(features,projected))]:
            desc={t:source.text[features.index[f'description:{t}']] for t in {r['ticker'] for r in rows}}
            datasets[representation]={'train':Samples(rows,tr,source,market,desc),'valid':Samples(rows,ev,source,market,desc)}
        schedule_results={};seed_scores={}
        for spec in CANDIDATES:
            group=[];ds=datasets[spec['representation']];width=ds['train'].text.shape[-1]
            for seed in SEEDS:
                name=f'{spec["name"]}_{seed}';rundir=folder/name;begin=time.monotonic()
                if (rundir/'summary.json').exists():
                    import csv
                    with (rundir/'selected_predictions.csv').open(newline='') as handle: saved=list(csv.DictReader(handle))
                    if [r['id'] for r in saved]!=[rows[i]['id'] for i in ev]:raise ValueError('Resume prediction IDs differ')
                    summary=json.loads((rundir/'summary.json').read_text())
                    for p,h in summary['artifact_sha256'].items():
                        if sha256_file(rundir/p)!=h:raise ValueError('Completed run artifact changed')
                    group.append(summary)
                    seed_scores[name]=np.array([float(r['logit']) for r in saved]);continue
                if rundir.exists():raise FileExistsError('Partial run retained; require separate recovery namespace')
                rundir.mkdir()
                def factory():
                    if spec['variant']=='price_only':return PriceIdentity(width)
                    return PaperFININ(variant=spec['variant'],text_width=width,width=32,attention_width=32,attention_heads=1,mlp_layers=2,dropout=.1,attention_style=spec['attention_style'])
                fit=fixed_budget_fit(factory,ds,seed,epochs=30,batch_size=64,learning_rate=.001,clip=1.,optimizer_name='adamw',weight_decay=.01)
                for kind in ['initial','selected','final']:torch.save(fit[f'{kind}_state'],rundir/f'{kind}.pt')
                write(rundir/'history.json',fit['history'])
                np.savez_compressed(rundir/'epoch_predictions.npz',train=np.array([p['logits']['train'] for p in fit['epoch_predictions']]),valid=np.array([p['logits']['valid'] for p in fit['epoch_predictions']]))
                logits=fit['selected_logits']['valid'];prediction_file(rundir/'selected_predictions.csv',rows,ev,logits)
                summary={'candidate':spec,'seed':seed,'selected_epoch':fit['selected_epoch'],
                         'train':evaluated(rows,tr,fit['selected_logits']['train']),'valid':evaluated(rows,ev,logits),
                         'last_epoch_train':fit['history'][-1]['eval']['train'],'last_epoch_valid':fit['history'][-1]['eval']['valid'],
                         'constant_fallback_would_be_selected':fit['fallback_to_constant'],
                         'parameter_count':sum(p.numel() for p in fit['model'].parameters()),'seconds':time.monotonic()-begin}
                if spec['variant']!='price_only':
                    donor=disruption(ds['valid'].rows,np.arange(len(ev)))
                    altered=Samples(rows,ev,features if spec['representation']=='raw' else ProjectedFeatures(features,projected),market,
                        {t:(features.text if spec['representation']=='raw' else projected)[features.index[f'description:{t}']] for t in {r['ticker'] for r in rows}})
                    # Same masks/counts/ticker/prices; jointly disrupt text and its matched sentiment.
                    altered.text=ds['valid'].text[donor].clone();altered.sent=ds['valid'].sent[donor].clone()
                    changed=predict_logits(fit['model'],altered);prediction_file(rundir/'permuted_predictions.csv',rows,ev,changed)
                    summary['text_permuted']=evaluated(rows,ev,changed);summary['decisions_changed']=int(((logits>=0)!=(changed>=0)).sum())
                    summary['permutation_loss_change_ci']=bootstrap_difference(rows,ev,changed,logits)
                    write(rundir/'permutation.json',{'donors':donor.tolist(),'changed_donors':int((donor!=np.arange(len(ev))).sum())})
                summary['artifact_sha256']={p.name:sha256_file(p) for p in rundir.iterdir() if p.is_file()}
                write(rundir/'summary.json',summary);group.append(summary);seed_scores[name]=logits
                print(f'{encoder} {schedule} {name}: epoch {fit["selected_epoch"]}, trainBCE {summary["train"]["log_loss"]:.5f}, validBCE {summary["valid"]["log_loss"]:.5f}, AUC {summary["valid"]["auc"]:.3f}, down {summary["valid"]["predicted_down"]}/512; {summary["seconds"]:.1f}s',flush=True)
                write(folder/'partial_results.json',schedule_results|{spec['name']:group})
            schedule_results[spec['name']]=group
        comparison={}
        for spec in CANDIDATES:
            average=np.mean([seed_scores[f'{spec["name"]}_{s}'] for s in SEEDS],axis=0)
            prediction_file(folder/f'{spec["name"]}_ensemble_predictions.csv',rows,ev,average)
            comparison[spec['name']]={'ensemble_valid':evaluated(rows,ev,average)}
            if spec['name']!='price_identity':
                price=np.mean([seed_scores[f'price_identity_{s}'] for s in SEEDS],axis=0)
                comparison[spec['name']]['minus_price_ci']=bootstrap_difference(rows,ev,average,price)
        write(folder/'results.json',{'runs':schedule_results,'comparisons':comparison});results[schedule]={'runs':schedule_results,'comparisons':comparison}
    features.close();write(directory/'results.json',{'configuration':config,'results':results,'seconds':time.monotonic()-start})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--encoder',default='bge',choices=['bge','roberta']);a=p.parse_args();run(a.out,a.encoder)

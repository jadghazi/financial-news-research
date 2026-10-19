"""Isolated fixed 1/3-session experiment; original pipelines remain unchanged."""
import csv,hashlib,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torch.utils.data._utils.collate import default_collate
from src import encoder_comparison as ec
from src.data.examples import market_features,select_headlines
from src.data.prices import load_prices
from src.training import make_model,set_seed
from src.training_data import PreparedDataset

SEEDS=[17,42,73]
ARMS=['full_p1n1','full_p3n1','full_p3n3','mean_p3n3','price_p1','price_p3']
FIELDS=['news_date','forecast_date','target_date','target_up']

def cohort(root,split):
    original=ec.rows(root,split);sessions,prices=load_prices(root);positions={d:i for i,d in enumerate(sessions)}
    raw={r['date']:r for r in map(json.loads,(root/f'data/raw/nifty/{split}.jsonl').read_text(encoding='utf-8').splitlines())}
    start=min(raw);manifest=ec.read(root/'data/processed/manifest.json');mean=manifest['market_train_mean'];scale=manifest['market_train_scale']
    rows=[];excluded=[]
    for r in original:
        p=positions[r['forecast_date']];news_dates=sessions[p-3:p];market_dates=sessions[p-2:p+1]
        if len(news_dates)!=3 or news_dates[0]<start:
            excluded.append({'news_date':r['news_date'],'reason':'three-session same-split warm-up'});continue
        if news_dates[-1]!=r['news_date'] or sessions[p+1]!=r['target_date']:raise ValueError('Forecast lag/target mismatch')
        if int(prices[r['target_date']]['close']>prices[r['forecast_date']]['close'])!=r['target_up']:raise ValueError('Label mismatch')
        feature_days=[];news=[]
        for n,d in zip(news_dates,market_dates):
            values=market_features(prices[n],prices[d]);feature_days.append([(v-m)/s for v,m,s in zip(values,mean,scale)])
            news.append(select_headlines(raw[n]['news']) if n in raw else [])
        if feature_days[-1]!=r['market_features'] or news[-1]!=r['headlines']:raise ValueError('Current input parity failure')
        rows.append(r|{'market_dates':market_dates,'source_news_dates':news_dates,'market_history':feature_days,'news_history':news,'news_available':[bool(h) for h in news]})
    return rows,excluded

class HistoryDataset(Dataset):
    def __init__(self,root,split):
        base=PreparedDataset(root,split);self.rows,self.excluded=cohort(root,split)
        self.index=base.index;self.text=base.text;self.sentiment=base.sentiment;self.description=base.description
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];text=np.zeros((3,16,384),np.float32);sentiment=np.zeros((3,16,3),np.float32);mask=np.zeros((3,16),bool)
        for j,heads in enumerate(r['news_history']):
            ids=[self.index[h['id']] for h in heads];n=len(ids)
            text[j,:n]=self.text[ids];sentiment[j,:n]=self.sentiment[ids];mask[j,:n]=True
        return {'market_features':torch.tensor(r['market_history'],dtype=torch.float32),'market_description':torch.from_numpy(self.description.copy()),'headline_vectors':torch.from_numpy(text),'sentiment':torch.from_numpy(sentiment),'mask':torch.from_numpy(mask),'target':torch.tensor(r['target_up'],dtype=torch.float32),'row_index':i}

class HistoryModel(nn.Module):
    def __init__(self,arm):
        super().__init__();self.arm=arm
        if arm.startswith('price'):
            self.predictor=nn.Sequential(nn.Linear(18,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1));return
        base=make_model('full');self.encoders=base.encoders;self.attention=base.attention
        self.predictor=nn.Sequential(nn.Linear(384,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,1))
    def representations(self,batch):
        b=len(batch['target']);numbers=batch['market_features'].reshape(b*3,6)
        desc=batch['market_description'][:,None,:].expand(-1,3,-1).reshape(b*3,384)
        market,news=self.encoders(desc,numbers,batch['headline_vectors'].reshape(b*3,16,384),batch['sentiment'].reshape(b*3,16,3))
        mask=batch['mask'].reshape(b*3,16);available=mask.any(1).reshape(b,3)
        if self.arm in ['full_p1n1','full_p3n1']:available=available&torch.tensor([False,False,True],device=available.device)
        active=available.flatten();summary=torch.zeros_like(market);weights=market.new_zeros((b*3,16))
        if active.any():
            pooled,w=self.attention(market[active],news[active],mask[active],use_self_attention=self.arm!='mean_p3n3',use_market_attention=self.arm!='mean_p3n3')
            summary[active]=pooled;weights[active]=w
        market=market.reshape(b,3,64);summary=summary.reshape(b,3,64)
        if self.arm=='full_p1n1':market=market*market.new_tensor([0,0,1])[None,:,None]
        return market,summary,weights.reshape(b,3,16)
    def forward(self,batch,override=None):
        if self.arm.startswith('price'):
            x=batch['market_features']
            if self.arm=='price_p1':x=x*x.new_tensor([0,0,1])[None,:,None]
            return self.predictor(x.reshape(len(x),18)).squeeze(-1),None
        market,summary,weights=self.representations(batch)
        if override is not None:
            summary=summary.clone();available=batch['mask'].any(-1)
            summary[:,:2]=override[None,:2,:]*available[:,:2,None]
        daily=torch.cat([market,summary],dim=-1).reshape(len(market),384)
        return self.predictor(daily).squeeze(-1),weights

def counts(model):
    total=sum(p.numel() for p in model.parameters());unused=0
    if model.arm=='full_p1n1':unused=256*64
    elif model.arm=='full_p3n1':unused=128*64
    elif model.arm=='mean_p3n3':unused=sum(p.numel() for p in model.attention.parameters())
    elif model.arm=='price_p1':unused=12*64
    return {'total':total,'structurally_unused':unused,'active':total-unused}

@torch.inference_mode()
def infer(model,ds,permutation=None,override=None):
    model.eval();values=[]
    for batch in DataLoader(ds,batch_size=64,shuffle=False,num_workers=0):
        if permutation is not None:
            for lag in [0,1]:
                alt=default_collate([ds[int(permutation[lag][i])] for i in batch['row_index'].tolist()])
                for f in ['headline_vectors','sentiment']:batch[f][:,lag]=alt[f][:,lag]
        l,_=model(batch,override);values.extend(l.double().tolist())
    return np.array(values)

def identity(root,out):
    files=['src/history_comparison.py','scripts/12_compare_histories.py','src/encoder_comparison.py','src/models/encoders.py','src/models/attention.py','src/data/examples.py','src/data/prices.py','src/training_data.py','src/training.py','src/models/finin.py','src/models/baselines.py','src/evaluation.py','src/features/cache.py','data/processed/manifest.json','data/spy_manifest.json','data/raw/spy/daily.json','artifacts/features/manifest.json']
    files += [f'data/{folder}/{s}.jsonl' for folder in ['processed','raw/nifty'] for s in ['train','valid']]
    return {'files_sha256':{n:ec.sha(root/n) for n in files},'protocol_sha256':ec.sha(out/'PROTOCOL.md'),'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':4,'device':'cpu'},'config':{'arms':ARMS,'seeds':SEEDS,'epochs':30,'lr':.001,'batch':16,'clip':1,'cap':16,'lookback':3,'dropout':.1,'width':64}}

def prepare(root,out,ds):
    for s,d in ds.items():
        path=out/f'{s}_cohort.jsonl';text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in d.rows)
        if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('Existing cohort changed')
        if not path.exists():path.write_text(text,encoding='utf-8')
    summary={s:{'count':len(d),'up':sum(r['target_up'] for r in d.rows),'available_news_days':{str(n):sum(sum(r['news_available'])==n for r in d.rows) for n in [1,2,3]},'excluded':d.excluded,'cohort_sha256':ec.sha(out/f'{s}_cohort.jsonl')} for s,d in ds.items()}
    if (out/'cohort.json').exists() and ec.read(out/'cohort.json')!=summary:raise ValueError('Cohort summary differs')
    ec.write(out/'cohort.json',summary);print(json.dumps(summary),flush=True)

def gates(root,out,ds):
    path=out/'gates.json'
    if path.exists():
        if not ec.read(path)['passed']:raise ValueError('Previous failed gate retained')
        return
    result={'passed':False,'rows_verified':0,'tiny':[]};all_ids=set()
    for split,d in ds.items():
        old=PreparedDataset(root,split);old_index={r['news_date']:i for i,r in enumerate(old.rows)}
        for i,r in enumerate(d.rows):
            b=d[i];a=old[old_index[r['news_date']]]
            for f in ['market_features','headline_vectors','sentiment','mask']:torch.testing.assert_close(b[f][-1],a[f],atol=0,rtol=0)
            for f in ['market_description','target']:torch.testing.assert_close(b[f],a[f],atol=0,rtol=0)
            for f in ['market_features','headline_vectors','sentiment','market_description']:
                if not torch.isfinite(b[f]).all():raise ValueError('Nonfinite history features')
            for lag,heads in enumerate(r['news_history']):
                if int(b['mask'][lag].sum())!=len(heads):raise ValueError('News mask/count mismatch')
                for j,h in enumerate(heads):
                    all_ids.add(h['id'])
                    if not np.array_equal(b['headline_vectors'][lag,j].numpy(),d.text[d.index[h['id']]]) or not np.array_equal(b['sentiment'][lag,j].numpy(),d.sentiment[d.index[h['id']]]):raise ValueError('History ID mapping mismatch')
            result['rows_verified']+=1
    result['unique_cached_ids_verified']=len(all_ids)
    d=ds['train'];indices=sorted(sum(([i for i,r in enumerate(d.rows) if r['target_up']==label][:16] for label in [0,1]),[]))
    ec.write(out/'tiny_subset.json',{'indices':indices,'news_dates':[d.rows[i]['news_date'] for i in indices]})
    base=default_collate([d[i] for i in indices]);base['mask'][0,0]=False;base['mask'][0,2,-1]=False
    initial={};result['arms']={}
    for arm in ARMS:
        set_seed(42);model=HistoryModel(arm);initial[arm]={n:p.detach().clone() for n,p in model.named_parameters()}
        optimizer=torch.optim.Adam(model.parameters(),lr=.001);batch={k:v.clone() for k,v in base.items()}
        for f in ['market_features','headline_vectors','sentiment']:batch[f].requires_grad_(True)
        l,w=model(batch);loss=nn.functional.binary_cross_entropy_with_logits(l,batch['target']);loss.backward()
        blocks=['predictor'] if arm.startswith('price') else ['encoders.text','encoders.market_numbers','encoders.news_numbers','encoders.market_fusion','encoders.news_fusion']+([] if arm.startswith('mean') else ['attention.self_attention','attention.normalization','attention.query','attention.key'])+['predictor']
        grad={b:float(sum(p.grad.square().sum() for n,p in model.named_parameters() if n.startswith(b) and p.grad is not None).sqrt()) for b in blocks}
        if not all(np.isfinite(v) and v>0 for v in grad.values()):raise ValueError('Dead history block gradient')
        if not arm.startswith('price'):
            for f in ['headline_vectors','sentiment']:
                if not torch.all(batch[f].grad[~batch['mask']]==0):raise ValueError('Masked/empty input gradient')
                if arm in ['full_p1n1','full_p3n1'] and not torch.all(batch[f].grad[:,:2]==0):raise ValueError('Suppressed old news gradient')
                if not torch.isfinite(batch[f].grad).all() or batch[f].grad.norm()<=0:raise ValueError('Dead news input gradient')
        if arm in ['full_p1n1','price_p1'] and not torch.all(batch['market_features'].grad[:,:2]==0):raise ValueError('Suppressed old market gradient')
        if not torch.isfinite(batch['market_features'].grad).all() or batch['market_features'].grad.norm()<=0:raise ValueError('Dead price input gradient')
        nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        updates={b:float(sum((p.detach()-initial[arm][n]).square().sum() for n,p in model.named_parameters() if n.startswith(b)).sqrt()) for b in blocks}
        if not all(np.isfinite(v) and v>0 for v in updates.values()):raise ValueError('Dead block update')
        model.eval()
        with torch.inference_mode():
            l,w=model(batch);alt={k:v.detach().clone() for k,v in batch.items()}
            for f in ['headline_vectors','sentiment']:alt[f][~alt['mask']]=500
            pad,_=model(alt);torch.testing.assert_close(l,pad,atol=1e-6,rtol=1e-6)
            if not arm.startswith('price'):
                for f in ['headline_vectors','sentiment','mask']:alt[f]=batch[f].flip(2)
                perm,_=model(alt);torch.testing.assert_close(l,perm,atol=1e-6,rtol=1e-6)
                available=batch['mask'].any(-1)
                if arm in ['full_p1n1','full_p3n1']:available=available&torch.tensor([False,False,True])
                torch.testing.assert_close(w.sum(-1),available.float(),atol=1e-6,rtol=0)
                if not torch.all(w[~batch['mask']]==0):raise ValueError('Masked attention weight')
                market,summary,_=model.representations(batch)
                if not torch.all(summary[~available]==0):raise ValueError('Biased empty/suppressed news summary')
            alt={k:v.detach().clone() for k,v in batch.items()}
            if arm in ['full_p1n1','full_p3n1']:
                for f in ['headline_vectors','sentiment']:alt[f][:,:2]=700
            if arm in ['full_p1n1','price_p1']:alt['market_features'][:,:2]=700
            suppressed,_=model(alt);torch.testing.assert_close(l,suppressed,atol=1e-6,rtol=1e-6)
        result['arms'][arm]={'parameters':counts(model),'block_gradients':grad,'block_updates':updates,'padding_error':float((l-pad).abs().max()),'suppression_error':float((l-suppressed).abs().max())}
    for arms in [ARMS[:4],ARMS[4:]]:
        for arm in arms[1:]:
            for n,p in initial[arms[0]].items():torch.testing.assert_close(p,initial[arm][n],atol=0,rtol=0)
    result['matched_initialization_verified']=True
    # Original one-day model remains functionally representable in the final slot.
    set_seed(17);one=make_model('full').eval();history=HistoryModel('full_p1n1').eval()
    history.encoders.load_state_dict(one.encoders.state_dict());history.attention.load_state_dict(one.attention.state_dict())
    with torch.no_grad():
        history.predictor[0].weight.zero_();history.predictor[0].weight[:,256:]=one.predictor[0].weight
        history.predictor[0].bias.copy_(one.predictor[0].bias);history.predictor[3].load_state_dict(one.predictor[3].state_dict())
    current={f:(base[f][:,-1] if f in ['market_features','headline_vectors','sentiment','mask'] else base[f]) for f in base}
    with torch.inference_mode():a,_=one(current);b,_=history(base)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6);result['one_day_functional_mapping_error']=float((a-b).abs().max())
    for condition in ['full','prices_fixed']:
        set_seed(42);model=HistoryModel('full_p3n3');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        batch={k:v.clone() for k,v in base.items()}
        if condition=='prices_fixed':batch['market_features']=torch.tensor(np.mean([r['market_history'] for r in d.rows],axis=0),dtype=torch.float32)[None].expand_as(batch['market_features'])
        hist=[];passed=False
        for step in range(601):
            if step%25==0:
                model.eval()
                with torch.inference_mode(),torch.random.fork_rng(devices=[]):l,_=model(batch)
                m=ec.metrics(batch['target'].numpy(),l.numpy());hist.append({'updates':step,'metrics':m})
                if m['accuracy']>=.95 and m['log_loss']<=.1:passed=True;break
                if step==600:break
            model.train();optimizer.zero_grad(set_to_none=True);l,_=model(batch);loss=nn.functional.binary_cross_entropy_with_logits(l,batch['target'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite tiny history loss')
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        torch.save(model.state_dict(),out/f'tiny_{condition}.pt');ec.write(out/f'tiny_{condition}_predictions.json',[{k:d.rows[i][k] for k in FIELDS}|{'training_index':i,'logit':float(x)} for i,x in zip(indices,l)])
        result['tiny'].append({'condition':condition,'passed':passed,'updates':step,'history':hist});ec.write(path,result)
        if not passed:raise ValueError('Finite-budget history memorization gate failed')
    model.eval();restored=HistoryModel('full_p3n3').eval();restored.load_state_dict(torch.load(out/'tiny_prices_fixed.pt',weights_only=True))
    for n,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[n],atol=0,rtol=0)
    with torch.inference_mode():a,_=model(batch);b,_=restored(batch)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    if not torch.equal(a>=0,b>=0):raise ValueError('Replay changed decisions')
    result.update(passed=True,checkpoint_error=float((a-b).abs().max()),checkpoint_weights_exact=True,checkpoint_decisions_exact=True)
    ec.write(path,result);print('History timeline/learning/masking gates passed',flush=True)

def append_predictions(writer,name,epoch,split,ds,l):
    for r,x in zip(ds.rows,l):writer.writerow({k:r[k] for k in FIELDS}|{'run':name,'epoch':epoch,'split':split,'logit':float(x)})

def controls(model,ds,folder,original):
    reference=[];effective={}
    with torch.inference_mode():
        summaries=[];masks=[]
        for batch in DataLoader(ds['train'],batch_size=64,shuffle=False):
            _,summary,_=model.representations(batch);summaries.append(summary);masks.append(batch['mask'].any(-1))
        summary=torch.cat(summaries);available=torch.cat(masks)
        reference=torch.stack([summary[:,j][available[:,j]].mean(0) for j in range(3)])
    ec.write(folder/'older_news_reference.json',reference.tolist());result={};valid=ds['valid'];y=np.array([r['target_up'] for r in valid.rows])
    with (folder/'sensitivity.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['condition']+FIELDS+['original_logit','altered_logit'])
        writer.writeheader()
        for condition in ['older_news_mean','older_permute_101','older_permute_202','older_permute_303']:
            permutation=None;changed={}
            if condition!='older_news_mean':
                rng=np.random.default_rng(int(condition.split('_')[-1]));permutation={}
                for lag in [0,1]:
                    perm=np.arange(len(valid));groups={}
                    for i,r in enumerate(valid.rows):groups.setdefault(len(r['news_history'][lag]),[]).append(i)
                    for n,indices in groups.items():
                        if n>0:perm[indices]=rng.permutation(indices)
                    permutation[lag]=perm;available=np.array([bool(r['news_history'][lag]) for r in valid.rows]);changed[str(lag)]={'available':int(available.sum()),'changed_bundle_index':int(((perm!=np.arange(len(valid)))&available).sum())}
                    if any(len(valid.rows[i]['news_history'][lag])!=len(valid.rows[int(perm[i])]['news_history'][lag]) for i in range(len(valid))):raise ValueError('Permutation changed count/missingness')
            logits=infer(model,valid,permutation=permutation,override=reference if condition=='older_news_mean' else None)
            p=1/(1+np.exp(-np.clip(original,-700,700)));q=1/(1+np.exp(-np.clip(logits,-700,700)))
            result[condition]={'metrics':ec.metrics(y,logits),'decisions_changed':int(((original>=0)!=(logits>=0)).sum()),'mean_abs_probability_change':float(np.abs(p-q).mean()),'bundle_changes_by_lag':changed}
            for r,a,b in zip(valid.rows,original,logits):writer.writerow({k:r[k] for k in FIELDS}|{'condition':condition,'original_logit':a,'altered_logit':b})
    return result

def fit(out,arm,seed,ds,prior):
    name=f'{arm}_{seed}';folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        saved=ec.read(folder/'summary.json')
        for n,h in saved['artifact_sha256'].items():
            if ec.sha(folder/n)!=h:raise ValueError('Existing run changed')
        return saved
    if (folder/'predictions.csv').exists():raise ValueError('Partial fit retained; new attempt required')
    set_seed(seed);model=HistoryModel(arm);torch.save(model.state_dict(),folder/'initial.pt');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loader=DataLoader(ds['train'],batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    target={s:np.array([r['target_up'] for r in d.rows]) for s,d in ds.items()};hist=[];best=float('inf');best_epoch=None;best_scores=None;orders=hashlib.sha256();start=time.monotonic()
    with (folder/'predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run','epoch','split']+FIELDS+['logit']);writer.writeheader()
        for epoch in range(1,31):
            model.train();total=0.;n=0
            for batch in loader:
                orders.update(batch['row_index'].numpy().astype(np.int64).tobytes());optimizer.zero_grad(set_to_none=True)
                l,_=model(batch);loss=nn.functional.binary_cross_entropy_with_logits(l,batch['target'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite comparison fit')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();total+=float(loss.detach())*len(l);n+=len(l)
            with torch.random.fork_rng(devices=[]):scores={s:infer(model,d) for s,d in ds.items()}
            evaluated={s:ec.metrics(target[s],l) for s,l in scores.items()};hist.append({'epoch':epoch,'online_train_loss':total/n,'eval':evaluated})
            if evaluated['valid']['log_loss']<best:
                best=evaluated['valid']['log_loss'];best_epoch=epoch;best_scores={s:l.copy() for s,l in scores.items()};torch.save(model.state_dict(),folder/'selected.pt')
            for s,l in scores.items():append_predictions(writer,name,epoch,s,ds[s],l)
            handle.flush();ec.write(folder/'history.json',hist)
            if epoch%10==0:print(f'{name} epoch{epoch}: train {evaluated["train"]["log_loss"]:.5f}, valid {evaluated["valid"]["log_loss"]:.5f}',flush=True)
    torch.save(model.state_dict(),folder/'final.pt');model.load_state_dict(torch.load(folder/'selected.pt',weights_only=True));model.eval()
    with torch.random.fork_rng(devices=[]):selected={s:infer(model,d) for s,d in ds.items()}
    errors={}
    for s,l in selected.items():
        np.testing.assert_allclose(l,best_scores[s],atol=1e-6,rtol=1e-6)
        if not np.array_equal(l>=0,best_scores[s]>=0):raise ValueError('Selected replay changed directions')
        errors[s]=float(np.abs(l-best_scores[s]).max())
    constant=ec.metrics(target['valid'],np.full(len(ds['valid']),np.log(prior/(1-prior))));fallback=constant['log_loss']<=best
    metrics={s:ec.metrics(target[s],l) for s,l in selected.items()}
    summary={'run':name,'arm':arm,'seed':seed,'epochs':30,'selected_epoch':best_epoch,'network':metrics,'constant':constant,'fallback_to_constant':fallback,'system_valid':constant if fallback else metrics['valid'],'parameters':counts(model),'minibatch_order_sha256':orders.hexdigest(),'replay_errors':errors,'elapsed_seconds':time.monotonic()-start}
    with (folder/'selected_predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run','epoch','split']+FIELDS+['logit']);writer.writeheader()
        for s,l in selected.items():append_predictions(writer,name,best_epoch,s,ds[s],l)
    with (folder/'fallback_predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run','split']+FIELDS+['network_probability_up','system_probability_up','fallback']);writer.writeheader()
        for s,l in selected.items():
            for r,x in zip(ds[s].rows,l):
                p=float(1/(1+np.exp(-np.clip(x,-700,700))));writer.writerow({k:r[k] for k in FIELDS}|{'run':name,'split':s,'network_probability_up':p,'system_probability_up':prior if fallback else p,'fallback':fallback})
    if arm in ['full_p3n3','mean_p3n3']:summary['older_news_controls']=controls(model,ds,folder,selected['valid'])
    summary['artifact_sha256']={p.name:ec.sha(p) for p in folder.iterdir() if p.is_file() and p.name!='summary.json'}
    ec.write(folder/'summary.json',summary);return summary

def train(root,out,ds):
    if (out/'results.json').exists():raise ValueError('Completed comparison cannot be overwritten')
    if not ec.read(out/'gates.json')['passed']:raise ValueError('Gates required')
    prior=float(np.mean([r['target_up'] for r in ds['train'].rows]));runs=[]
    for arm in ARMS:
        for seed in SEEDS:runs.append(fit(out,arm,seed,ds,prior));ec.write(out/'partial_results.json',runs)
    for seed in SEEDS:
        group=[r for r in runs if r['seed']==seed]
        if len({r['minibatch_order_sha256'] for r in group})!=1:raise ValueError('Minibatch orders differ')
        for arms in [ARMS[:4],ARMS[4:]]:
            first=torch.load(out/'runs'/f'{arms[0]}_{seed}'/'initial.pt',weights_only=True)
            for arm in arms[1:]:
                state=torch.load(out/'runs'/f'{arm}_{seed}'/'initial.pt',weights_only=True)
                for n,v in first.items():torch.testing.assert_close(v,state[n],atol=0,rtol=0)
    ec.write(out/'results.json',{'scope':'Exploratory train/validation 1/3-session comparison','prior':prior,'runs':runs,'orders_and_initial_weights_verified':True})

def selected_predictions(folder,split,ds):
    rows=[r for r in csv.DictReader((folder/'selected_predictions.csv').open(encoding='utf-8')) if r['split']==split]
    if len(rows)!=len(ds):raise ValueError('Saved prediction count differs')
    for a,b in zip(rows,ds.rows):
        if any(a[k]!=str(b[k]) for k in FIELDS):raise ValueError('Saved prediction date/label mismatch')
    return np.array([int(r['target_up']) for r in rows]),np.array([float(r['logit']) for r in rows])

def block_intervals(delta):
    n=len(delta);result={}
    for block in [5,20]:
        rng=np.random.default_rng(20261012);samples=[]
        for _ in range(2000):
            idx=((rng.integers(0,n,size=(int(np.ceil(n/block)),1))+np.arange(block))%n).ravel()[:n];samples.append(float(delta[idx].mean()))
        result[str(block)]=np.quantile(samples,[.025,.975]).tolist()
    return result

def analyze(root,out,ds):
    results=ec.read(out/'results.json');byname={r['run']:r for r in results['runs']};arrays={};rows=[]
    for r in results['runs']:
        folder=out/'runs'/r['run']
        if ec.read(folder/'summary.json')!=r:raise ValueError('Summary/results mismatch')
        for name,h in r['artifact_sha256'].items():
            if ec.sha(folder/name)!=h:raise ValueError('Saved artifact changed')
        for s,d in ds.items():
            y,l=selected_predictions(folder,s,d);m=ec.metrics(y,l);arrays[(r['run'],s)]=(y,l)
            for key in ['log_loss','accuracy','balanced_accuracy','auc','brier','predicted_down']:
                if not np.isclose(m[key],r['network'][s][key],atol=1e-12,rtol=0):raise ValueError('Saved metric differs')
        hist=ec.read(folder/'history.json')
        if len(hist)!=30 or r['selected_epoch']!=1+int(np.argmin([h['eval']['valid']['log_loss'] for h in hist])):raise ValueError('Checkpoint selection differs')
        rows.append(r|{'final_train':hist[-1]['eval']['train'],'final_valid':hist[-1]['eval']['valid']})
    paired=[];aggregates={}
    for name,a,b in [('price_context_with_current_news','full_p1n1','full_p3n1'),('older_news_primary','full_p3n1','full_p3n3'),('attention_vs_mean','mean_p3n3','full_p3n3'),('price_only_history','price_p1','price_p3'),('news_vs_price_context','price_p3','full_p3n3')]:
        differences=[]
        for seed in SEEDS:
            y,l=arrays[(f'{a}_{seed}','valid')];z,k=arrays[(f'{b}_{seed}','valid')]
            if not np.array_equal(y,z):raise ValueError('Unmatched paired labels')
            delta=(np.logaddexp(0,k)-y*k)-(np.logaddexp(0,l)-y*l);differences.append(delta);ca=(l>=0)==y;cb=(k>=0)==y
            paired.append({'contrast':name,'reference':a,'candidate':b,'seed':seed,'bce_delta':float(delta.mean()),'accuracy_delta':float(cb.mean()-ca.mean()),'new_correct':int((~ca&cb).sum()),'new_wrong':int((ca&~cb).sum()),'directions_changed':int(((l>=0)!=(k>=0)).sum())})
        average=np.mean(differences,axis=0);aggregates[name]={'mean_bce_delta':float(average.mean()),'per_seed_bce_delta':[float(d.mean()) for d in differences],'conditional_descriptive_intervals_observed_rows':block_intervals(average)}
    ec.write(out/'analysis.json',{'scope':'Exploratory conditional validation comparisons; not independent generalization','results_sha256':ec.sha(out/'results.json'),'runs':rows,'paired':paired,'contrasts':aggregates,'saved_selected_metrics_and_artifacts_verified':True})
    for r in rows:print(r['run'],'epoch',r['selected_epoch'],'BCE',round(r['network']['valid']['log_loss'],6),'accuracy',round(100*r['network']['valid']['accuracy'],2),'down',r['network']['valid']['predicted_down'],'fallback',r['fallback_to_constant'])
    print(json.dumps(aggregates,indent=2))

def execute(root,out,stage):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if not (out/'preservation_before.json').exists():raise ValueError('Original evidence preservation snapshot required')
    ident=identity(root,out);path=out/'execution_identity.json'
    if path.exists() and ec.read(path)!=ident:raise ValueError('Frozen experiment identity changed')
    if not path.exists():ec.write(path,ident)
    for src,name in [('src/history_comparison.py','code_snapshot.py'),('scripts/12_compare_histories.py','entrypoint_snapshot.py')]:
        p=out/name
        if p.exists() and p.read_bytes()!=(root/src).read_bytes():raise ValueError('Source snapshot differs')
        if not p.exists():p.write_bytes((root/src).read_bytes())
    ec.write(out/f'command_{stage}.json',{'command':f'.venv/Scripts/python.exe -B scripts/12_compare_histories.py {stage} --experiment {out.name}','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True),'tracked_diff':subprocess.check_output(['git','diff','--no-ext-diff'],cwd=root,text=True)})
    ds={s:HistoryDataset(root,s) for s in ['train','valid']}
    if stage=='prepare':prepare(root,out,ds)
    elif stage=='gates':gates(root,out,ds)
    elif stage=='train':train(root,out,ds)
    else:analyze(root,out,ds)
    before=ec.read(out/'preservation_before.json');changed=[n for n,h in before.items() if not (root/n).exists() or ec.sha(root/n)!=h]
    ec.write(out/f'preservation_after_{stage}.json',{'checked':len(before),'changed':changed})
    if changed:raise ValueError('Original research evidence changed')

"""Versioned input, attribution and low-capacity diagnostics; no legacy writes."""
import csv,hashlib,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from src import encoder_comparison as ec
from src.data.examples import select_headlines,market_features
from src.data.prices import load_prices
from src.training import make_model
from src.training_data import PreparedDataset

PENALTIES=[10.,1.,.1,.01]
MODELS=['prices','prices_coverage']+[f'{kind}_lag{lag}' for kind in ['sent','text','news','normnews'] for lag in [0,1]]
FIELDS=['forecast_date','target_date','target_up']

def write_predictions(path,rows,values,extra=None):
    if len(rows)!=len(values):raise ValueError('Prediction length differs from dates')
    extra=extra or {};names=list(extra)+FIELDS+['logit']
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=names);writer.writeheader()
        for r,l in zip(rows,values):writer.writerow(extra|{k:r[k] for k in FIELDS}|{'logit':float(l)})

def assemble(root,out):
    base=PreparedDataset(root,'train');days,prices=load_prices(root);positions={d:i for i,d in enumerate(days)}
    records=[];excluded=[];price=[];coverage=[];text={0:[],1:[]};normtext={0:[],1:[]};sent={0:[],1:[]};members={}
    for split in ['train','valid']:
        original=ec.rows(root,split);raw={r['date']:r for r in map(json.loads,(root/f'data/raw/nifty/{split}.jsonl').read_text(encoding='utf-8').splitlines())};members[split]=[]
        for r in original:
            d=r['forecast_date'];p=positions[d];n=days[p-1];q=days[p+1]
            if r['news_date']!=n or r['target_date']!=q:raise ValueError('Original timeline differs from calendar')
            if d not in raw:
                excluded.append({'split':split,'forecast_date':d,'target_date':q,'reason':'same-day source news unavailable'});continue
            y=int(prices[q]['close']>prices[d]['close'])
            if y!=r['target_up']:raise ValueError('Source label differs')
            numbers=market_features(prices[n],prices[d])
            if numbers!=r['market_features_raw']:raise ValueError('Raw market parity differs')
            row={k:r[k] for k in FIELDS}|{'split':split,'news_dates':{'0':d,'1':n},'headline_ids':{},'raw_unique_count':{},'selected_count':{}}
            cov=[]
            for lag,day in [(0,d),(1,n)]:
                heads=select_headlines(raw[day]['news']);ids=[h['id'] for h in heads];idx=[base.index[h] for h in ids]
                for h in heads:
                    if hashlib.sha256(h['text'].encode('utf-8')).hexdigest()!=h['id']:raise ValueError('Headline content/ID mismatch')
                vectors=np.asarray(base.text[idx],dtype=np.float64);scores=np.asarray(base.sentiment[idx],dtype=np.float64)
                norms=np.linalg.norm(vectors,axis=1)
                if np.any(norms<=0) or not np.isfinite(vectors).all() or not np.isfinite(scores).all():raise ValueError('Invalid frozen features')
                if lag==1 and heads!=r['headlines']:raise ValueError('Legacy selected news differs')
                count=len({h.strip() for h in raw[day]['news'].splitlines() if h.strip()})
                cov.extend([np.log1p(count),len(heads)/count]);text[lag].append(vectors.mean(0));normtext[lag].append((vectors/norms[:,None]).mean(0));sent[lag].append(scores.mean(0))
                row['headline_ids'][str(lag)]=ids;row['raw_unique_count'][str(lag)]=count;row['selected_count'][str(lag)]=len(heads)
            members[split].append(len(records));records.append(row);price.append(numbers);coverage.append(cov)
    arrays={'prices':np.array(price),'coverage':np.array(coverage)}
    for lag in [0,1]:
        arrays[f'text{lag}']=np.array(text[lag]);arrays[f'normtext{lag}']=np.array(normtext[lag]);arrays[f'sent{lag}']=np.array(sent[lag])
    path=out/'paired_cohort.jsonl';content=''.join(json.dumps(r)+'\n' for r in records)
    if path.exists() and path.read_text(encoding='utf-8')!=content:raise ValueError('Frozen paired cohort differs')
    if not path.exists():path.write_text(content,encoding='utf-8')
    ec.write(out/'paired_cohort_summary.json',{'counts':{s:len(v) for s,v in members.items()},'up':{s:sum(records[i]['target_up'] for i in v) for s,v in members.items()},'exclusions':excluded,'cohort_sha256':ec.sha(path),'scope':'Intersection of lag0/lag1 source coverage; no publication-time assurance'})
    if len(members['train'])!=1082 or len(members['valid'])!=228:raise ValueError('Prespecified paired cohort differs')
    return records,arrays,members,days,positions

def design(arrays,name):
    if name=='prices':return arrays['prices']
    common=np.concatenate([arrays['prices'],arrays['coverage']],1)
    if name=='prices_coverage':return common
    kind,lag=name.split('_lag');lag=int(lag)
    if kind=='sent':extra=arrays[f'sent{lag}']
    elif kind=='text':extra=arrays[f'text{lag}']
    elif kind in ['news','normnews']:extra=np.concatenate([arrays[f'{"normtext" if kind=="normnews" else "text"}{lag}'],arrays[f'sent{lag}']],1)
    else:raise ValueError('Unknown diagnostic model')
    return np.concatenate([common,extra],1)

def fit_linear(x,y,penalty):
    start=time.monotonic();mean=x.mean(0);scale=x.std(0);scale=np.where(scale<1e-8,1.,scale)
    z=torch.tensor((x-mean)/scale,dtype=torch.float64);labels=torch.tensor(y,dtype=torch.float64)
    model=nn.Linear(x.shape[1],1,dtype=torch.float64)
    with torch.no_grad():model.weight.zero_();model.bias.fill_(np.log(y.mean()/(1-y.mean())))
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1.,max_iter=200,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
    def closure():
        nonlocal calls
        calls+=1;optimizer.zero_grad(set_to_none=True);l=model(z).squeeze(1)
        loss=nn.functional.binary_cross_entropy_with_logits(l,labels)+penalty*model.weight.square().sum()/2
        if not torch.isfinite(loss):raise ValueError('Nonfinite convex objective')
        loss.backward();return loss
    optimizer.step(closure);objective=closure();gradient=max(float(p.grad.abs().max()) for p in model.parameters())
    w=model.weight.detach().numpy()[0].copy();b=float(model.bias.detach()[0]);iterations=int(optimizer.state[model.weight].get('n_iter',0))
    result={'mean':mean.tolist(),'scale':scale.tolist(),'weight':w.tolist(),'bias':b,'penalty':penalty,'objective':float(objective.detach()),'gradient_max':gradient,'iterations':iterations,'closure_calls':calls,'seconds':time.monotonic()-start,'converged':gradient<=1e-5}
    if not result['converged']:raise ValueError(f'Convex fit did not converge: gradient {gradient}')
    return result

def linear_predict(model,x):return ((x-np.array(model['mean']))/np.array(model['scale']))@np.array(model['weight'])+model['bias']

def solver_gate(out):
    rng=np.random.default_rng(131313);x=rng.normal(size=(600,6));y=(x[:,0]>0).astype(float)
    model=fit_linear(x[:400],y[:400],.1);l=linear_predict(model,x[400:]);m=ec.metrics(y[400:],l)
    passed=m['balanced_accuracy']>=.9 and m['log_loss']<=.4
    ec.write(out/'linear_solver_gate.json',{'passed':passed,'model':model,'heldout_metrics':m,'scope':'Analytical feature-rule solver check, no financial targets'})
    if not passed:raise ValueError('Linear solver analytical rule failed')

def eligible(rows,indices,first_forecast,days,positions):
    cutoff=days[positions[first_forecast]-1];keep=[i for i in indices if rows[i]['target_date']<=cutoff]
    if any(rows[i]['forecast_date']>=first_forecast for i in keep):raise ValueError('Chronological fit/eval overlap')
    return np.array(keep,dtype=int),cutoff

def baselines(root,out):
    folder=out/'baselines';folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():raise ValueError('Completed baselines cannot be overwritten')
    rows,arrays,members,days,positions=assemble(root,out);solver_gate(out);y=np.array([r['target_up'] for r in rows],dtype=float)
    tr=members['train'];specs=[(tr[:500],tr[500:700]),(tr[:700],tr[700:900]),(tr[:900],tr[900:]),(tr,members['valid'])];folds=[];all_results=[]
    for number,(prefix,evaluation) in enumerate(specs):
        training,cutoff=eligible(rows,prefix,rows[evaluation[0]]['forecast_date'],days,positions)
        split=int(.8*len(training));inner_eval=training[split:];inner_train,inner_cutoff=eligible(rows,training[:split],rows[inner_eval[0]]['forecast_date'],days,positions)
        if len(inner_train)<100 or len(inner_eval)<50:raise ValueError('Insufficient chronological inner fit')
        fold={'fold':number,'training_indices':training.tolist(),'evaluation_indices':list(evaluation),'inner_training_indices':inner_train.tolist(),'inner_evaluation_indices':inner_eval.tolist(),'fit_cutoff':cutoff,'inner_fit_cutoff':inner_cutoff,'first_outer_forecast':rows[evaluation[0]]['forecast_date'],'last_outer_forecast':rows[evaluation[-1]]['forecast_date']};folds.append(fold);ec.write(folder/'folds.json',folds)
        directory=folder/f'fold_{number}';directory.mkdir(exist_ok=True);prior=float(y[training].mean());constant=np.full(len(evaluation),np.log(prior/(1-prior)))
        write_predictions(directory/'constant_predictions.csv',[rows[i] for i in evaluation],constant)
        block={'fold':number,'constant':ec.metrics(y[evaluation],constant),'prior':prior,'models':{}}
        for name in MODELS:
            modeldir=directory/name
            if modeldir.exists():raise ValueError('Partial fit retained; new attempt namespace required')
            modeldir.mkdir();x=design(arrays,name);candidates=[];best=float('inf');selected=None
            for penalty in PENALTIES:
                model=fit_linear(x[inner_train],y[inner_train],penalty);l=linear_predict(model,x[inner_eval]);m=ec.metrics(y[inner_eval],l)
                ec.write(modeldir/f'inner_model_{penalty}.json',model);write_predictions(modeldir/f'inner_predictions_{penalty}.csv',[rows[i] for i in inner_eval],l)
                candidates.append({'penalty':penalty,'metrics':m,'gradient_max':model['gradient_max']})
                if m['log_loss']<best:best=m['log_loss'];selected=penalty
            model=fit_linear(x[training],y[training],selected);l=linear_predict(model,x[evaluation]);m=ec.metrics(y[evaluation],l);train_l=linear_predict(model,x[training])
            ec.write(modeldir/'outer_model.json',model);write_predictions(modeldir/'outer_predictions.csv',[rows[i] for i in evaluation],l)
            summary={'model':name,'fold':number,'selected_penalty':selected,'inner_candidates':candidates,'train_metrics':ec.metrics(y[training],train_l),'outer_metrics':m,'feature_count':x.shape[1],'artifact_sha256':{p.name:ec.sha(p) for p in modeldir.iterdir() if p.is_file()}}
            ec.write(modeldir/'summary.json',summary);block['models'][name]=summary
            print(f'linear fold{number} {name}: outer BCE{m["log_loss"]:.6f}, accuracy{m["accuracy"]:.3f}, penalty{selected}',flush=True)
        all_results.append(block);ec.write(folder/'partial_results.json',all_results)
    ec.write(folder/'results.json',{'scope':'Retrospective matched lag/representation outer blocks, not untouched market tests','folds':all_results,'model_names':MODELS,'penalties':PENALTIES,'fold_manifest_sha256':ec.sha(folder/'folds.json')})

def trace(model,batch):
    market,news=model.encoders(batch['market_description'],batch['market_features'],batch['headline_vectors'],batch['sentiment'])
    summary,weights=model.attention(market,news,batch['mask'],use_self_attention=model.variant!='mean_pool',use_market_attention=model.variant!='mean_pool')
    predictor_input=torch.cat([market,summary],-1);logits=model.predictor(predictor_input).squeeze(1)
    return {'projected_headlines':model.encoders.text(batch['headline_vectors']),'fused_news':news,'market':market,'summary':summary,'weights':weights,'predictor_input':predictor_input,'logits':logits}

def flow(root,out):
    folder=out/'headline_flow';folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():raise ValueError('Completed flow cannot be overwritten')
    train=PreparedDataset(root,'train');valid=PreparedDataset(root,'valid');real_ids=sorted({train.index[h['id']] for r in train.rows for h in r['headlines']})
    means={'headline_vectors':torch.tensor(np.asarray(train.text[real_ids],dtype=np.float64).mean(0),dtype=torch.float32),'sentiment':torch.tensor(np.asarray(train.sentiment[real_ids],dtype=np.float64).mean(0),dtype=torch.float32),'market_features':torch.tensor(np.array([r['market_features'] for r in train.rows]).mean(0),dtype=torch.float32)}
    checkpoints=[(f'original_selected_full_{s}','full',root/f'artifacts/runs/full_{s}.pt') for s in [17,42,73]]
    checkpoints += [('late_full_73','full',root/'artifacts/diagnostics/learning_check_2026-10-12/full_73/best_30.pt'),('final_full_73','full',root/'artifacts/diagnostics/learning_check_2026-10-12/full_73/final.pt'),('best_mean_42','mean_pool',root/'artifacts/diagnostics/learning_check_2026-10-12/mean_pool_42/best_30.pt'),('final_mean_42','mean_pool',root/'artifacts/diagnostics/learning_check_2026-10-12/mean_pool_42/final.pt')]
    rng=np.random.default_rng(101);permutation=np.arange(len(valid));groups={}
    for i,r in enumerate(valid.rows):groups.setdefault(len(r['headlines']),[]).append(i)
    for count,indices in groups.items():permutation[indices]=rng.permutation(indices)
    ec.write(folder/'control_manifest.json',{'permutation':permutation.tolist(),'changed_bundle_indices':int((permutation!=np.arange(len(valid))).sum()),'reference_scope':'unique selected training IDs / training market rows','checkpoint_sources':{n:{'path':str(p),'sha256':ec.sha(p)} for n,v,p in checkpoints}})
    all_results=[];y=np.array([r['target_up'] for r in valid.rows]);balanced=sorted(sum(([i for i,r in enumerate(train.rows) if r['target_up']==label][:16] for label in [0,1]),[]))
    padded=next(i for i,r in enumerate(train.rows) if len(r['headlines'])<16)
    if padded not in balanced:balanced.append(padded)
    for name,variant,path in checkpoints:
        directory=folder/name;directory.mkdir();model=make_model(variant);model.load_state_dict(torch.load(path,weights_only=True));model.eval()
        batch=default_collate([train[i] for i in balanced]);batch={k:v.clone() for k,v in batch.items()}
        for f in ['market_features','headline_vectors','sentiment']:batch[f].requires_grad_(True)
        model.zero_grad(set_to_none=True);l,_=model(batch);loss=nn.functional.binary_cross_entropy_with_logits(l,batch['target']);loss.backward()
        inputs={f:{'norm':float(batch[f].grad.norm()),'mean_absolute':float(batch[f].grad.abs().mean())} for f in ['market_features','headline_vectors','sentiment']}
        masked_zero=all(torch.all(batch[f].grad[~batch['mask']]==0).item() for f in ['headline_vectors','sentiment'])
        gradients={n:None if p.grad is None else float(p.grad.norm()) for n,p in model.named_parameters()}
        ec.write(directory/'gradient_trace.json',{'input_gradients':inputs,'parameter_gradients':gradients,'masked_inputs_zero':masked_zero,'padded_slots_checked':int((~batch['mask']).sum()),'training_row_indices':balanced,'scope':'32 balanced real training rows plus an actual padded row, no optimizer update'})
        conditions=['original','text_permute','sentiment_permute','bundle_permute','text_mean','sentiment_mean','market_mean'];results={};original=None
        for condition in conditions:
            values=[];stage_deltas={k:[] for k in ['projected_headlines','fused_news','market','summary','weights','predictor_input']}
            with torch.inference_mode():
                for batch in DataLoader(valid,batch_size=64,shuffle=False):
                    base_trace=trace(model,batch);alt={k:v.clone() for k,v in batch.items()}
                    if condition.endswith('permute'):
                        donors=default_collate([valid[int(permutation[i])] for i in batch['row_index'].tolist()])
                        fields=['headline_vectors'] if condition=='text_permute' else ['sentiment'] if condition=='sentiment_permute' else ['headline_vectors','sentiment']
                        for f in fields:alt[f]=donors[f]
                    elif condition.endswith('mean'):
                        f={'text_mean':'headline_vectors','sentiment_mean':'sentiment','market_mean':'market_features'}[condition];alt[f]=means[f].expand_as(alt[f]).clone()
                        if f!='market_features':alt[f][~alt['mask']]=0
                    altered=trace(model,alt);normal,_=model(alt);torch.testing.assert_close(normal,altered['logits'],atol=1e-6,rtol=1e-6)
                    values.extend(altered['logits'].double().tolist())
                    for k in stage_deltas:stage_deltas[k].extend((altered[k]-base_trace[k]).abs().reshape(len(normal),-1).mean(1).double().tolist())
            values=np.array(values);m=ec.metrics(y,values);write_predictions(directory/f'{condition}_predictions.csv',valid.rows,values)
            if condition=='original':original=values.copy()
            p=1/(1+np.exp(-np.clip(values,-700,700)));q=1/(1+np.exp(-np.clip(original,-700,700)))
            results[condition]={'metrics':m,'mean_absolute_logit_change':float(np.abs(values-original).mean()),'max_absolute_logit_change':float(np.abs(values-original).max()),'mean_absolute_probability_change':float(np.abs(p-q).mean()),'decisions_changed':int(((values>=0)!=(original>=0)).sum()),'mean_stage_changes':{k:float(np.mean(v)) for k,v in stage_deltas.items()}}
        summary={'checkpoint':name,'variant':variant,'source_sha256':ec.sha(path),'gradients':inputs,'masked_gradients_zero':masked_zero,'conditions':results,'artifact_sha256':{p.name:ec.sha(p) for p in directory.iterdir() if p.is_file()}}
        ec.write(directory/'summary.json',summary);all_results.append(summary);ec.write(folder/'partial_results.json',all_results)
        print(f'headline flow {name}: original BCE{results["original"]["metrics"]["log_loss"]:.6f}, text-permutation decisions{results["text_permute"]["decisions_changed"]}',flush=True)
    ec.write(folder/'results.json',{'scope':'Frozen network-use diagnostic; no semantic/causal inference','checkpoints':all_results,'references':{k:v.tolist() for k,v in means.items()}})

def interval(delta,block):
    rng=np.random.default_rng(20261013);n=len(delta);values=[]
    for _ in range(2000):
        index=((rng.integers(0,n,size=(int(np.ceil(n/block)),1))+np.arange(block))%n).ravel()[:n];values.append(float(delta[index].mean()))
    return np.quantile(values,[.025,.975]).tolist()

def analyze(out):
    folder=out/'baselines';results=ec.read(folder/'results.json');allrows=[];logits={name:[] for name in MODELS+['constant']};ys=[]
    for fold in results['folds']:
        directory=folder/f'fold_{fold["fold"]}';base=list(csv.DictReader((directory/'constant_predictions.csv').open(encoding='utf-8')));labels=np.array([int(r['target_up']) for r in base]);ys.extend(labels);allrows.extend(base);logits['constant'].extend(float(r['logit']) for r in base)
        for name in MODELS:
            modeldir=directory/name;summary=ec.read(modeldir/'summary.json')
            for n,h in summary['artifact_sha256'].items():
                if ec.sha(modeldir/n)!=h:raise ValueError('Baseline artifact changed')
            rows=list(csv.DictReader((modeldir/'outer_predictions.csv').open(encoding='utf-8')))
            if any(any(a[k]!=b[k] for k in FIELDS) for a,b in zip(rows,base)) or len(rows)!=len(base):raise ValueError('Unmatched baseline predictions')
            l=np.array([float(r['logit']) for r in rows]);m=ec.metrics(labels,l)
            for k in ['log_loss','accuracy','balanced_accuracy','auc','brier','predicted_down']:
                if not np.isclose(m[k],summary['outer_metrics'][k],atol=1e-12,rtol=0):raise ValueError('Baseline metric mismatch')
            logits[name].extend(l)
    y=np.array(ys);logits={k:np.array(v) for k,v in logits.items()};metrics={k:ec.metrics(y,l) for k,l in logits.items()}
    comparisons=[('coverage','prices','prices_coverage')]
    for lag in [0,1]:comparisons += [(f'news_increment_lag{lag}','prices_coverage',f'news_lag{lag}'),(f'text_beyond_sentiment_lag{lag}',f'sent_lag{lag}',f'news_lag{lag}'),(f'normalization_lag{lag}',f'news_lag{lag}',f'normnews_lag{lag}')]
    comparisons += [('lag0_vs_lag1','news_lag1','news_lag0')];contrasts={}
    for name,ref,candidate in comparisons:
        a=logits[ref];b=logits[candidate];delta=(np.logaddexp(0,b)-y*b)-(np.logaddexp(0,a)-y*a)
        contrasts[name]={'reference':ref,'candidate':candidate,'bce_delta':float(delta.mean()),'block_intervals':{str(n):interval(delta,n) for n in [5,20]},'blockwise_bce_delta':[fold['models'][candidate]['outer_metrics']['log_loss']-fold['models'][ref]['outer_metrics']['log_loss'] for fold in results['folds']]}
    ec.write(out/'baseline_analysis.json',{'scope':'Retrospective conditional four-block evidence','outer_date_count':len(y),'pooled_metrics':metrics,'contrasts':contrasts,'results_sha256':ec.sha(folder/'results.json')})
    print(json.dumps({'pooled':{k:{f:v[f] for f in ['log_loss','accuracy','balanced_accuracy']} for k,v in metrics.items()},'contrasts':contrasts},indent=2))

def execute(root,out,stage):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if not (out/'preservation_before.json').exists():raise ValueError('Preservation snapshot required')
    names=['src/pipeline_diagnosis.py','scripts/13_diagnose_pipeline.py','src/encoder_comparison.py','src/positive_controls.py','src/training.py','src/training_data.py','src/models/finin.py','src/models/attention.py','src/models/encoders.py','src/evaluation.py','src/data/examples.py','src/data/prices.py','data/processed/train.jsonl','data/processed/valid.jsonl','data/raw/nifty/train.jsonl','data/raw/nifty/valid.jsonl','data/raw/spy/daily.json','artifacts/features/manifest.json']
    identity={'files_sha256':{n:ec.sha(root/n) for n in names},'protocol_sha256':ec.sha(out/'PROTOCOL.md'),'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'threads':4,'device':'cpu'}}
    path=out/'execution_identity.json'
    if path.exists() and ec.read(path)!=identity:raise ValueError('Frozen diagnosis source identity changed')
    if not path.exists():ec.write(path,identity)
    for name in ['src/pipeline_diagnosis.py','src/positive_controls.py','scripts/13_diagnose_pipeline.py']:
        target=out/('snapshot_'+Path(name).name)
        if target.exists() and target.read_bytes()!=(root/name).read_bytes():raise ValueError('Diagnostic source snapshot differs')
        if not target.exists():target.write_bytes((root/name).read_bytes())
    ec.write(out/f'command_{stage}.json',{'command':f'.venv/Scripts/python.exe -B scripts/13_diagnose_pipeline.py {stage}','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True),'tracked_diff':subprocess.check_output(['git','diff','--no-ext-diff'],cwd=root,text=True)})
    if stage=='flow':flow(root,out)
    elif stage=='baselines':baselines(root,out)
    elif stage=='analyze':analyze(out)
    elif stage=='positive':
        from src.positive_controls import run
        run(root,out/'positive_controls')
    before=ec.read(out/'preservation_before.json');changed=[n for n,h in before.items() if not (root/n).exists() or ec.sha(root/n)!=h]
    ec.write(out/f'preservation_after_{stage}.json',{'checked':len(before),'changed':changed})
    if changed:raise ValueError('Existing evidence changed')

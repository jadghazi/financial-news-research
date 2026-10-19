"""Nested chronological downstream tuning; no outer-label model selection."""
import csv, json, platform, shutil, time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from src import encoder_comparison as ec
from src.ml_study_data import StudyCohort, fit_scaler, chronological_plan
from src.models.baselines import NumericBaseline
from src.models.finin import ReducedFININ
from src.research_training import fixed_budget_fit,fit_train_only,predict_logits

ARMS=['price_only','mean_pool','full'];SEEDS=[17,42,73]
PROFILES={
    'baseline':dict(width=64,dropout=.1,learning_rate=.001,weight_decay=0.),
    'decay_only':dict(width=64,dropout=.1,learning_rate=.001,weight_decay=.1),
    'compact_regularized':dict(width=32,dropout=.3,learning_rate=.0003,weight_decay=.1)}
FIELDS=['news_date','forecast_date','target_date','target_up']

def factory(arm,profile):
    cfg=PROFILES[profile]
    return lambda:(NumericBaseline(False,cfg['width'],cfg['dropout']) if arm=='price_only'
                   else ReducedFININ(arm,cfg['width'],cfg['dropout']))

def optimization(profile):
    cfg=PROFILES[profile]
    return dict(batch_size=16,learning_rate=cfg['learning_rate'],weight_decay=cfg['weight_decay'],optimizer_name='adamw',clip=1.)

def labels(ds):return np.array([r['target_up'] for r in ds.rows],dtype=float)

def save_predictions(path,ds,logits,extra=None):
    if len(ds)!=len(logits):raise ValueError('Prediction/date alignment differs')
    extra=extra or {}
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(extra)+FIELDS+['logit']);writer.writeheader()
        for r,l in zip(ds.rows,logits):writer.writerow(extra|{k:r[k] for k in FIELDS}|{'logit':float(l)})

def parameter_counts(model):
    total=sum(p.numel() for p in model.parameters())
    inactive=sum(p.numel() for p in model.attention.parameters()) if getattr(model,'variant',None)=='mean_pool' else 0
    return {'total':total,'inactive_attention':inactive,'active':total-inactive}

def fit_calibrator(y,logits):
    x=torch.tensor(logits,dtype=torch.float64);target=torch.tensor(y,dtype=torch.float64)
    theta=torch.nn.Parameter(torch.tensor(-np.log(2.),dtype=torch.float64))
    intercept=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
    optimizer=torch.optim.LBFGS([theta,intercept],max_iter=100,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad();slope=3*theta.sigmoid()
        loss=torch.nn.functional.binary_cross_entropy_with_logits(slope*x+intercept,target)+.01*((slope-1)**2+intercept**2)/2
        loss.backward();return loss
    optimizer.step(closure);objective=closure()
    result={'slope':float((3*theta.sigmoid()).detach()),'intercept':float(intercept.detach()),'objective':float(objective.detach()),'ridge':.01,
            'gradient_max':max(float(theta.grad.abs()),float(intercept.grad.abs())),'inner_count':len(y),
            'scope':'Selected inner-holdout calibration; coefficients transferred to refit scores'}
    if not all(np.isfinite(result[k]) for k in ['slope','intercept','objective']) or result['gradient_max']>1e-5:
        raise ValueError('Calibration failed convergence')
    return result

def prepare(root,out):
    cohort=StudyCohort(root);plans=chronological_plan(cohort.rows,cohort.sessions)
    text=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in cohort.rows)
    path=out/'cohort.jsonl'
    if path.exists() and path.read_text(encoding='utf-8')!=text:raise ValueError('Cohort changed')
    if not path.exists():path.write_text(text,encoding='utf-8')
    for plan in plans:
        for prefix,indices in [('inner',plan['inner_fit_indices']),('outer',plan['outer_train_indices'])]:
            mean,scale=fit_scaler(cohort.raw_market,indices);plan[prefix+'_scaler']={'mean':mean.tolist(),'scale':scale.tolist()}
    if (out/'plans.json').exists() and ec.read(out/'plans.json')!=plans:raise ValueError('Plan changed')
    ec.write(out/'plans.json',plans)
    files=['src/ml_training_study.py','src/ml_study_data.py','src/research_training.py','scripts/14_study_ml_training.py',
           'src/models/finin.py','src/models/baselines.py','src/models/attention.py','src/models/encoders.py','src/encoder_comparison.py',
           'src/training_data.py','src/training.py','src/data/examples.py','src/data/prices.py','src/evaluation.py','src/features/cache.py',
           'data/raw/spy/daily.json','data/processed/train.jsonl','data/processed/valid.jsonl','data/raw/nifty/train.jsonl','data/raw/nifty/valid.jsonl',
           'data/nifty_manifest.json','data/spy_manifest.json','artifacts/features/manifest.json','tests/test_research_training.py','tests/test_ml_study_data.py']
    identity={'files_sha256':{p:ec.sha(root/p) for p in files},'cohort_sha256':ec.sha(path),'protocol_sha256':ec.sha(out/'PROTOCOL.md'),
              'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':4},
              'config':{'arms':ARMS,'seeds':SEEDS,'profiles':PROFILES,'inner_epochs':30,'selection':'common epoch minimizes three-seed mean inner BCE'}}
    if (out/'execution_identity.json').exists() and ec.read(out/'execution_identity.json')!=identity:raise ValueError('Frozen execution identity changed')
    ec.write(out/'execution_identity.json',identity)
    for rel in files:
        if rel.endswith('.py'):
            target=out/'source_snapshot'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(root/rel,target)
            if ec.sha(target)!=identity['files_sha256'][rel]:raise ValueError('Snapshot mismatch')
    print('Verified cohort1790; nested plans/scalers and source identity retained',flush=True)
    return cohort,plans

def inner_run(out,fold,arm,profile,seed,ds):
    folder=out/'inner'/f'f{fold}_{arm}_{profile}_{seed}';summary_path=folder/'summary.json'
    if summary_path.exists():
        summary=ec.read(summary_path)
        for n,h in summary['artifact_sha256'].items():
            if ec.sha(folder/n)!=h:raise ValueError('Completed inner artifact changed')
        return summary
    if folder.exists():raise ValueError(f'Partial run retained; use new attempt, not overwrite: {folder}')
    folder.mkdir(parents=True);start=time.monotonic();print('Inner',folder.name,flush=True)
    fit=fixed_budget_fit(factory(arm,profile),ds,seed,epochs=30,**optimization(profile))
    for n in ['initial','selected','final']:torch.save(fit[n+'_state'],folder/(n+'.pt'))
    np.savez_compressed(folder/'epoch_logits.npz',train=np.stack([p['logits']['train'] for p in fit['epoch_predictions']]),
                        valid=np.stack([p['logits']['valid'] for p in fit['epoch_predictions']]),train_y=labels(ds['train']),valid_y=labels(ds['valid']))
    ec.write(folder/'history.json',fit['history'])
    save_predictions(folder/'selected_valid.csv',ds['valid'],fit['selected_logits']['valid'])
    summary={'arm':arm,'profile':profile,'seed':seed,'fold':fold,'selected_epoch_individual':fit['selected_epoch'],
             'inner_valid_loss_curve':[h['eval']['valid']['log_loss'] for h in fit['history']],
             'constant_validation':fit['constant_validation'],'parameters':parameter_counts(fit['model']),'seconds':time.monotonic()-start}
    summary['artifact_sha256']={p.name:ec.sha(p) for p in folder.iterdir() if p.is_file()}
    ec.write(summary_path,summary);print(f"Finished {folder.name} {summary['seconds']:.1f}s",flush=True);return summary

class Intervention(Dataset):
    def __init__(self,base,mode,permutation,reference):self.base=base;self.mode=mode;self.permutation=permutation;self.reference=reference
    def __len__(self):return len(self.base)
    def __getitem__(self,i):
        b=self.base[i]
        if self.mode=='text_mean':b['headline_vectors'][b['mask']]=self.reference
        else:
            alt=self.base[int(self.permutation[i])]
            for field in (['headline_vectors'] if self.mode=='text_permute' else ['sentiment'] if self.mode=='sentiment_permute' else ['headline_vectors','sentiment']):
                b[field]=alt[field]
        return b

def interventions(model,train,outer,original,folder):
    groups={};permutation=np.arange(len(outer));rng=np.random.default_rng(101)
    for i,r in enumerate(outer.rows):groups.setdefault(len(r['headlines']),[]).append(i)
    for members in groups.values():permutation[members]=rng.permutation(members)
    selected=torch.cat([train[i]['headline_vectors'][train[i]['mask']] for i in range(len(train))]);reference=selected.mean(0)
    result={};y=labels(outer);p=1/(1+np.exp(-np.clip(original,-700,700)))
    for mode in ['text_permute','sentiment_permute','bundle_permute','text_mean']:
        altered=predict_logits(model,Intervention(outer,mode,permutation,reference));q=1/(1+np.exp(-np.clip(altered,-700,700)))
        save_predictions(folder/(mode+'.csv'),outer,altered)
        result[mode]={'metrics':ec.metrics(y,altered),'decisions_changed':int(((altered>=0)!=(original>=0)).sum()),
                      'mean_abs_probability_change':float(np.abs(q-p).mean()),'loss_change':float((np.logaddexp(0,altered)-y*altered-np.logaddexp(0,original)+y*original).mean())}
    ec.write(folder/'intervention_setup.json',{'permutation':permutation.tolist(),'text_mean':reference.tolist(),
              'mean_scope':'Occurrence-weighted fitting-headline mean, no outer features','changed_bundle_indices':int((permutation!=np.arange(len(outer))).sum())})
    return result

def outer_run(out,fold,arm,profile,seed,epoch,train,outer,calibration,selected):
    folder=out/'outer'/f'f{fold}_{arm}_{profile}_{seed}';path=folder/'summary.json'
    if path.exists():
        summary=ec.read(path)
        for n,h in summary['artifact_sha256'].items():
            if ec.sha(folder/n)!=h:raise ValueError('Completed outer artifact changed')
        return summary
    if folder.exists():raise ValueError(f'Partial outer run retained: {folder}')
    folder.mkdir(parents=True);start=time.monotonic();print(f'Outer {folder.name} epoch{epoch}',flush=True)
    fit=fit_train_only(factory(arm,profile),train,seed,epochs=epoch,**optimization(profile))
    torch.save(fit['initial_state'],folder/'initial.pt');torch.save(fit['final_state'],folder/'final.pt');ec.write(folder/'history.json',fit['history'])
    logits=predict_logits(fit['model'],outer);model=factory(arm,profile)();model.load_state_dict(torch.load(folder/'final.pt',weights_only=True))
    replay=predict_logits(model,outer);np.testing.assert_allclose(logits,replay,atol=1e-6,rtol=1e-6)
    prior=fit['training_prior'];constant=np.full(len(outer),np.log(prior/(1-prior)))
    outputs={'network':logits,'constant':constant};summary={'fold':fold,'arm':arm,'profile':profile,'seed':seed,'epochs':epoch,
             'train_metrics':fit['history'][-1]['train'],'training_prior':prior,'parameters':parameter_counts(model),'replay_max_error':float(np.abs(logits-replay).max())}
    if selected is not None:
        outputs['system']=constant if selected['fallback_to_constant'] else logits
        outputs['calibrated']=calibration['slope']*logits+calibration['intercept']
        summary.update(fallback_to_constant=selected['fallback_to_constant'],calibration=calibration)
    summary['outer_metrics']={n:ec.metrics(labels(outer),l) for n,l in outputs.items()}
    for name,l in outputs.items():save_predictions(folder/(name+'.csv'),outer,l)
    if selected is not None and arm!='price_only':summary['headline_interventions']=interventions(model,train,outer,logits,folder)
    summary['seconds']=time.monotonic()-start;summary['artifact_sha256']={p.name:ec.sha(p) for p in folder.iterdir() if p.is_file()}
    ec.write(path,summary);print(f"Finished {folder.name}: BCE {summary['outer_metrics']['network']['log_loss']:.6f}, down {summary['outer_metrics']['network']['predicted_down']}/{len(outer)}",flush=True)
    return summary

def train(root,out):
    if (out/'results.json').exists():raise ValueError('Refusing to rerun completed study')
    cohort,plans=prepare(root,out);all_results=[]
    for plan in plans:
        f=plan['fold'];ds={}
        for phase,training,evaluation in [('inner','inner_fit_indices','inner_valid_indices'),('outer','outer_train_indices','outer_eval_indices')]:
            mean=np.array(plan[phase+'_scaler']['mean']);scale=np.array(plan[phase+'_scaler']['scale'])
            ds[phase]={'train':cohort.dataset(plan[training],mean,scale),'valid':cohort.dataset(plan[evaluation],mean,scale)}
        for arm in ARMS:
            candidates={};inner={}
            for profile in PROFILES:
                summaries=[inner_run(out,f,arm,profile,seed,ds['inner']) for seed in SEEDS];inner[profile]=summaries
                curve=np.mean([s['inner_valid_loss_curve'] for s in summaries],axis=0);epoch=int(np.argmin(curve))+1
                candidates[profile]={'epoch':epoch,'mean_inner_bce':float(curve[epoch-1]),'mean_loss_curve':curve.tolist()}
            profile=min(PROFILES,key=lambda p:candidates[p]['mean_inner_bce']);choice=candidates[profile]
            constant=float(np.mean([s['constant_validation']['log_loss'] for s in inner[profile]]))
            selected={'profile':profile,'epoch':choice['epoch'],'fallback_to_constant':constant<=choice['mean_inner_bce'],'constant_inner_bce':constant,'candidates':candidates}
            choice_path=out/f'choice_f{f}_{arm}.json'
            if choice_path.exists() and ec.read(choice_path)!=selected:raise ValueError('Frozen inner choice changed')
            ec.write(choice_path,selected)
            print('Selected',f,arm,profile,choice['epoch'],'fallback',selected['fallback_to_constant'],flush=True)
            records=[]
            for seed in SEEDS:
                inner_folder=out/'inner'/f'f{f}_{arm}_{profile}_{seed}'
                with np.load(inner_folder/'epoch_logits.npz') as saved:
                    calibration=fit_calibrator(saved['valid_y'],saved['valid'][choice['epoch']-1])
                chosen=outer_run(out,f,arm,profile,seed,choice['epoch'],ds['outer']['train'],ds['outer']['valid'],calibration,selected)
                baseline=chosen if profile=='baseline' else outer_run(out,f,arm,'baseline',seed,candidates['baseline']['epoch'],ds['outer']['train'],ds['outer']['valid'],None,None)
                records.append({'seed':seed,'chosen':chosen,'baseline':baseline})
            all_results.append({'fold':f,'arm':arm,'choice':selected,'runs':records})
            ec.write(out/'partial_results.json',all_results)
    ec.write(out/'results.json',all_results)

def conditional_interval(delta,fold_sizes,block):
    rng=np.random.default_rng(20261013);values=[];offsets=np.cumsum([0]+fold_sizes)
    for _ in range(2000):
        total=0.
        for start,end in zip(offsets[:-1],offsets[1:]):
            n=end-start;starts=rng.integers(0,n,size=int(np.ceil(n/block)))
            indices=((starts[:,None]+np.arange(block))%n).ravel()[:n]
            total+=delta[start+indices].sum()
        values.append(total/len(delta))
    return {'mean_delta':float(delta.mean()),'block_rows':block,'ci95':np.quantile(values,[.025,.975]).tolist(),
            'scope':'Conditional descriptive within-window blocks of reused dated losses; no independent significance claim'}

def analyze(out):
    results=ec.read(out/'results.json');plans=ec.read(out/'plans.json');records={};pooled={};summaries=[]
    for arm in ARMS:
        records[arm]={n:[] for n in ['network','baseline','constant','system','calibrated']}
        for seed in SEEDS:
            arrays={n:[] for n in records[arm]};y=[];dates=[]
            for entry in results:
                if entry['arm']!=arm:continue
                run=next(r for r in entry['runs'] if r['seed']==seed);f=entry['fold']
                for name in arrays:
                    source=run['baseline'] if name=='baseline' else run['chosen'];filename='network' if name=='baseline' else name
                    folder=out/'outer'/f"f{f}_{arm}_{source['profile']}_{seed}"
                    rows=list(csv.DictReader((folder/(filename+'.csv')).open(encoding='utf-8')))
                    arrays[name].extend(float(r['logit']) for r in rows)
                    if name=='network':y.extend(float(r['target_up']) for r in rows);dates.extend(r['forecast_date'] for r in rows)
            if len(dates)!=len(set(dates)) or len(dates)!=990:raise ValueError('Outer dates overlapped or count changed')
            y=np.array(y);seed_metrics={n:ec.metrics(y,np.array(l)) for n,l in arrays.items()}
            for n,l in arrays.items():records[arm][n].append(np.logaddexp(0,np.array(l))-y*np.array(l))
            pooled.setdefault(arm,[]).append({'seed':seed,'metrics':seed_metrics})
        means={n:float(np.mean(records[arm][n])) for n in records[arm]}
        summaries.append({'arm':arm,'mean_seed_pooled_bce':means,'mean_network_accuracy':float(np.mean([r['metrics']['network']['accuracy'] for r in pooled[arm]])),
                          'mean_network_balanced_accuracy':float(np.mean([r['metrics']['network']['balanced_accuracy'] for r in pooled[arm]])),
                          'network_down_calls_by_seed':[r['metrics']['network']['predicted_down'] for r in pooled[arm]],
                          'network_auc_by_seed':[r['metrics']['network']['auc'] for r in pooled[arm]]})
    differences={f'{arm}_tuned_minus_baseline':np.mean(records[arm]['network'],0)-np.mean(records[arm]['baseline'],0) for arm in ARMS}
    differences.update(full_minus_prices=np.mean(records['full']['network'],0)-np.mean(records['price_only']['network'],0),
                       full_minus_mean=np.mean(records['full']['network'],0)-np.mean(records['mean_pool']['network'],0),
                       full_minus_constant=np.mean(records['full']['network'],0)-np.mean(records['full']['constant'],0),
                       full_calibrated_minus_raw=np.mean(records['full']['calibrated'],0)-np.mean(records['full']['network'],0))
    intervals={n:[conditional_interval(d,[p['outer_end']-p['outer_start'] for p in plans],b) for b in [5,20]] for n,d in differences.items()}
    final={'scope':'Retrospective nested selection on three reused chronological windows; raw networks, no seed ensemble',
           'outer_dates':990,'inner_fits':len(list((out/'inner').glob('*/summary.json'))),'outer_fits':len(list((out/'outer').glob('*/summary.json'))),
           'summaries':summaries,'pooled_per_seed':pooled,'paired_conditional_intervals':intervals}
    ec.write(out/'analysis.json',final);print(json.dumps(final['summaries'],indent=2),flush=True)

def execute(root,out,stage):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    if not (out/'PROTOCOL.md').exists():raise ValueError('Frozen protocol required')
    if stage=='prepare':prepare(root,out)
    elif stage=='train':train(root,out)
    elif stage=='analyze':prepare(root,out);analyze(out)
    else:raise ValueError('Unknown stage')

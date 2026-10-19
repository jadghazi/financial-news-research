"""Versioned 16/64 news coverage experiment using unchanged FININ training logic."""
import hashlib, json, platform, subprocess, time
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import try_to_load_from_cache
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from src import encoder_comparison as ec
from src.data.examples import select_headlines
from src.features.cache import TEXT_NAME,TEXT_REVISION,SENTIMENT_NAME,SENTIMENT_REVISION,encode_batch,check_sentiment_controls
from src.training import make_model,set_seed
from src.training_data import PreparedDataset

SEEDS=[17,42,73]
def selection(root,split,cap):
    original=ec.rows(root,split)
    source={r['date']:r['news'] for r in map(json.loads,(root/f'data/raw/nifty/{split}.jsonl').read_text(encoding='utf-8').splitlines())}
    result=[]
    for row in original:
        chosen=select_headlines(source[row['news_date']],cap)
        if chosen[:16]!=row['headlines']:raise ValueError('Cap selection is not nested')
        result.append(row|{'headlines':chosen})
    return result

def listing(root):
    texts={}
    for split in ['train','valid']:
        for row in selection(root,split,64):
            for h in row['headlines']:
                old=texts.setdefault(h['id'],h['text'])
                if old!=h['text']:raise ValueError('ID collision')
    return sorted(texts.items())

def identity(root):
    return {'schema':1,'cap':64,'splits':['train','valid'],'text_model':TEXT_NAME,'text_revision':TEXT_REVISION,'sentiment_model':SENTIMENT_NAME,
        'sentiment_revision':SENTIMENT_REVISION,'sentiment_order':['positive','neutral','negative'],'pooling':'first final token','normalization':'none','max_tokens':128,'dtype':'float32','width':384,
        'source_sha256':{s:ec.sha(root/f'data/raw/nifty/{s}.jsonl') for s in ['train','valid']},
        'processed_sha256':{s:ec.sha(root/f'data/processed/{s}.jsonl') for s in ['train','valid']},
        'legacy_manifest_sha256':ec.sha(root/'artifacts/features/manifest.json'),'generator_sha256':ec.sha(Path(__file__)),
        'legacy_encoder_code_sha256':ec.sha(root/'src/features/cache.py'),'selection_code_sha256':ec.sha(root/'src/data/examples.py')}

def local_snapshot(model,revision,filename):
    path=try_to_load_from_cache(model,filename,revision=revision)
    if not isinstance(path,str) or not Path(path).is_file():raise ValueError('Required pinned model not locally cached')
    directory=Path(path).parent
    hashes={p.name:ec.sha(p) for p in directory.iterdir() if p.is_file()}
    return directory,hashes

def legacy(root):
    PreparedDataset(root,'train') # Verifies original data/cache signatures and every channel hash.
    feat=root/'artifacts/features';index={h:i for i,h in enumerate(ec.read(feat/'headline_index.json'))}
    arrays={k:np.load(feat/f'{k}.npy',mmap_mode='r') for k in ['text','sentiment']}
    return index,arrays

def cache(root,out):
    directory=out/'features';directory.mkdir(exist_ok=True);ident=identity(root)
    if (directory/'manifest.json').exists():verify_cache(root,out);return
    idx,old=legacy(root);items=listing(root);ids=[h for h,_ in items];missing=[i for i,h in enumerate(ids) if h not in idx]
    if len(ids)!=99494 or len(missing)!=72223:raise ValueError('Coverage audit workload mismatch')
    index_path=directory/'headline_index.json'
    if index_path.exists() and ec.read(index_path)!=ids:raise ValueError('Existing index mismatch')
    if not index_path.exists():ec.write(index_path,ids)
    for split in ['train','valid']:
        path=directory/f'{split}.jsonl'
        content=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selection(root,split,64))
        if path.exists() and path.read_text(encoding='utf-8')!=content:raise ValueError('Prepared cap64 changed')
        if not path.exists():path.write_text(content,encoding='utf-8')
    states={}
    for kind,model_id,revision,file,width in [('text',TEXT_NAME,TEXT_REVISION,'model.safetensors',384),('sentiment',SENTIMENT_NAME,SENTIMENT_REVISION,'model.onnx',3)]:
        snapshot,hashes=local_snapshot(model_id,revision,file)
        state_path=directory/f'{kind}_state.json';array_path=directory/f'{kind}.npy'
        if state_path.exists():
            state=ec.read(state_path)
            if state['identity']!=ident or state['model_files']!=hashes or state['index_sha256']!=ec.sha(index_path):raise ValueError('Partial generator/index mismatch')
            array=np.lib.format.open_memmap(array_path,mode='r+')
            if array.shape!=(len(ids),width):raise ValueError('Partial shape mismatch')
        else:
            array=np.lib.format.open_memmap(array_path,mode='w+',dtype=np.float32,shape=(len(ids),width))
            for i,h in enumerate(ids):
                if h in idx:array[i]=old[kind][idx[h]]
            array.flush();state={'identity':ident,'model_files':hashes,'index_sha256':ec.sha(index_path),'next_missing':0,'missing_count':len(missing),'reused':len(ids)-len(missing),'seconds':0.,'truncated':0}
            ec.write(state_path,state)
        if state['next_missing']==len(missing):
            if state['sha256']!=ec.sha(array_path):raise ValueError('Completed channel changed')
            states[kind]=state;continue
        tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,use_fast=True)
        if kind=='text':
            model,loading=AutoModel.from_pretrained(str(snapshot),local_files_only=True,use_safetensors=True,torch_dtype=torch.float32,output_loading_info=True)
            if any(loading.get(k) for k in ['missing_keys','mismatched_keys','error_msgs']):raise ValueError('Incomplete frozen text weights')
            ec.write(directory/'text_loading.json',loading);model.eval();model.requires_grad_(False)
        else:
            options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
            model=ort.InferenceSession(str(snapshot/file),sess_options=options,providers=['CPUExecutionProvider'])
            ec.write(directory/'sentiment_controls.json',check_sentiment_controls(model,tokenizer,'cpu'))
        # Same extraction must regenerate a sample of reused vectors before mixing channels.
        reused=[i for i,h in enumerate(ids) if h in idx][:8]
        sample,_=encode_batch(model,tokenizer,[items[i][1] for i in reused],kind,'cpu')
        np.testing.assert_allclose(sample,array[reused],atol=1e-4,rtol=0)
        state['reused_sample_max_error']=float(np.abs(sample-array[reused]).max())
        start=time.monotonic();base=state['seconds']
        for offset in range(state['next_missing'],len(missing),8):
            positions=missing[offset:offset+8]
            values,n=encode_batch(model,tokenizer,[items[i][1] for i in positions],kind,'cpu')
            if values.shape!=(len(positions),width) or not np.isfinite(values).all():raise ValueError('Invalid encoded features')
            if kind=='sentiment' and not np.allclose(values.sum(1),1,atol=1e-6):raise ValueError('Sentiment probability sum')
            array[positions]=values;array.flush();state.update(next_missing=offset+len(positions),seconds=base+time.monotonic()-start,truncated=state['truncated']+n)
            if state['next_missing']==len(missing):state['sha256']=ec.sha(array_path)
            ec.write(state_path,state)
            if offset%4000==0:print(f'{kind} {state["next_missing"]}/{len(missing)} new headlines; {state["seconds"]:.1f}s',flush=True)
        state['sha256']=ec.sha(array_path);ec.write(state_path,state);states[kind]=state
        del model,tokenizer
    description=directory/'market_description.npy';original=root/'artifacts/features/market_description.npy'
    if not description.exists():description.write_bytes(original.read_bytes())
    if ec.sha(description)!=ec.sha(original):raise ValueError('Description changed')
    for kind in ['text','sentiment']:
        a=np.load(directory/f'{kind}.npy',mmap_mode='r')
        if not np.isfinite(a).all():raise ValueError('Nonfinite completed cache')
        reused=[i for i,h in enumerate(ids) if h in idx]
        if not np.array_equal(a[reused],old[kind][[idx[ids[i]] for i in reused]]):raise ValueError('Reused features changed')
        if kind=='sentiment' and (np.any(a<0) or not np.allclose(a.sum(1),1,atol=1e-6)):raise ValueError('Completed probabilities')
    names=['text.npy','sentiment.npy','headline_index.json','market_description.npy','train.jsonl','valid.jsonl']
    ec.write(directory/'manifest.json',{'identity':ident,'states':states,'files_sha256':{n:ec.sha(directory/n) for n in names},'complete':True,'ids':len(ids),'missing_ids':len(missing)})
    print('Expanded feature cache complete',flush=True)

def verify_cache(root,out):
    directory=out/'features';manifest=ec.read(directory/'manifest.json')
    if not manifest['complete'] or manifest['identity']!=identity(root):raise ValueError('Expanded cache identity mismatch')
    for name,digest in manifest['files_sha256'].items():
        if ec.sha(directory/name)!=digest:raise ValueError('Expanded cache content changed')
    return manifest

class CapDataset(Dataset):
    def __init__(self,root,out,split,cap):
        self.rows=selection(root,split,cap);self.cap=cap;self.width=384
        directory=out/'features';self.index={h:i for i,h in enumerate(ec.read(directory/'headline_index.json'))}
        self.text=np.load(directory/'text.npy',mmap_mode='r');self.sentiment=np.load(directory/'sentiment.npy',mmap_mode='r');self.description=np.load(directory/'market_description.npy')
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];n=len(row['headlines']);indices=[self.index[h['id']] for h in row['headlines']]
        text=np.zeros((self.cap,384),np.float32);sent=np.zeros((self.cap,3),np.float32);mask=np.zeros(self.cap,bool)
        text[:n]=self.text[indices];sent[:n]=self.sentiment[indices];mask[:n]=True
        return {'market_features':torch.tensor(row['market_features'],dtype=torch.float32),'market_description':torch.from_numpy(self.description.copy()),'headline_vectors':torch.from_numpy(text),'sentiment':torch.from_numpy(sent),'mask':torch.from_numpy(mask),'target':torch.tensor(row['target_up'],dtype=torch.float32),'row_index':index}

def gates(root,out,datasets):
    if (out/'gates.json').exists():
        if not ec.read(out/'gates.json')['passed']:raise ValueError('Prior failed gate preserved')
        return
    matches=0
    for split in ['train','valid']:
        old=PreparedDataset(root,split)
        for i in range(len(old)):
            if datasets[16][split].rows[i]!=old.rows[i]:raise ValueError('Original row metadata differs')
            a=old[i];b=datasets[16][split][i]
            for key in a:
                if isinstance(a[key],torch.Tensor):torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
                elif a[key]!=b[key]:raise ValueError('Row index differs')
            c=datasets[64][split][i]
            for key in ['market_features','market_description','target']:torch.testing.assert_close(a[key],c[key],atol=0,rtol=0)
            if not torch.equal(a['headline_vectors'],c['headline_vectors'][:16]) or not torch.equal(a['sentiment'],c['sentiment'][:16]):raise ValueError('16 prefix features changed')
            matches+=1
    ds=datasets[64]['train'];indices=sorted(sum(([i for i,r in enumerate(ds.rows) if r['target_up']==label][:16] for label in [0,1]),[]));base=default_collate([ds[i] for i in indices])
    result={'passed':False,'exact_legacy_rows':matches,'tiny':[]}
    set_seed(42);model=make_model('full');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    batch={k:v.clone() for k,v in base.items()};batch['mask'][0,-1]=False
    for field in ['market_features','headline_vectors','sentiment']:batch[field].requires_grad_(True)
    before={n:p.detach().clone() for n,p in model.named_parameters()};logits,_=model(batch)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['target']);loss.backward()
    blocks=['encoders.text','encoders.market_numbers','encoders.news_numbers','encoders.market_fusion','encoders.news_fusion','attention.self_attention','attention.normalization','attention.query','attention.key','predictor']
    grads={b:float(sum(p.grad.square().sum() for n,p in model.named_parameters() if n.startswith(b) and p.grad is not None).sqrt()) for b in blocks}
    if not all(np.isfinite(v) and v>0 for v in grads.values()):raise ValueError('Dead block gradient')
    for f in ['headline_vectors','sentiment']:
        if not torch.all(batch[f].grad[~batch['mask']]==0):raise ValueError('Padded input gradient')
    input_grads={f:float(batch[f].grad.norm()) for f in ['market_features','headline_vectors','sentiment']}
    if not all(np.isfinite(v) and v>0 for v in input_grads.values()):raise ValueError('Dead input channel')
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
    changes={b:float(sum((p.detach()-before[n]).square().sum() for n,p in model.named_parameters() if n.startswith(b)).sqrt()) for b in blocks}
    if not all(np.isfinite(v) and v>0 for v in changes.values()):raise ValueError('Dead block update')
    result['gradient_norms']=grads;result['input_gradient_norms']=input_grads;result['update_norms']=changes
    for condition in ['full','prices_fixed']:
        set_seed(42);model=make_model('full');optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        batch={k:v.clone() for k,v in base.items()}
        if condition=='prices_fixed':batch['market_features']=torch.tensor(np.mean([r['market_features'] for r in ds.rows],axis=0),dtype=torch.float32).expand_as(batch['market_features'])
        history=[];passed=False
        for step in range(601):
            if step%25==0:
                model.eval()
                with torch.inference_mode(),torch.random.fork_rng(devices=[]):logits,_=model(batch)
                m=ec.metrics(batch['target'].numpy(),logits.numpy());history.append({'updates':step,'metrics':m})
                if m['accuracy']>=.95 and m['log_loss']<=.1:passed=True;break
                if step==600:break
            model.train();optimizer.zero_grad(set_to_none=True);logits,_=model(batch);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,batch['target'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite tiny fit')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
        torch.save(model.state_dict(),out/f'tiny_{condition}.pt')
        ec.write(out/f'tiny_{condition}_predictions.json',[{'index':i,'news_date':ds.rows[i]['news_date'],'target_up':ds.rows[i]['target_up'],'logit':float(l)} for i,l in zip(indices,logits)])
        result['tiny'].append({'condition':condition,'passed':passed,'updates':step,'history':history});ec.write(out/'gates.json',result)
        if not passed:raise ValueError('Finite-budget tiny check failed')
    model.eval();batch['mask'][0,-1]=False
    with torch.inference_mode():
        logits,weights=model(batch);altered={k:v.clone() for k,v in batch.items()};altered['headline_vectors'][~batch['mask']]=500;altered['sentiment'][~batch['mask']]=-500
        pad,_=model(altered);torch.testing.assert_close(logits,pad,atol=1e-5,rtol=0)
        for f in ['headline_vectors','sentiment','mask']:altered[f]=batch[f].flip(1)
        perm,_=model(altered);torch.testing.assert_close(logits,perm,atol=1e-5,rtol=0)
        torch.testing.assert_close(weights.sum(1),torch.ones(len(weights)),atol=1e-6,rtol=0)
        if not torch.all(weights[~batch['mask']]==0):raise ValueError('Masked weights nonzero')
        restored=make_model('full').eval();restored.load_state_dict(torch.load(out/'tiny_prices_fixed.pt',weights_only=True))
        for name,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[name],atol=0,rtol=0)
        replay,_=restored(batch);torch.testing.assert_close(logits,replay,atol=1e-6,rtol=1e-6)
        if not torch.equal(logits>=0,replay>=0):raise ValueError('Replay changes decision')
    result.update(passed=True,padding_error=float((logits-pad).abs().max()),permutation_error=float((logits-perm).abs().max()),replay_error=float((logits-replay).abs().max()))
    ec.write(out/'gates.json',result);print('Cap64 integrity/learning gates passed',flush=True)

def train(root,out):
    if (out/'results.json').exists():raise ValueError('Completed experiment cannot be overwritten')
    verify_cache(root,out);prior_dir=root/'artifacts/experiments/encoder_comparison_2026-10-12_replay'
    prior_identity=ec.read(prior_dir/'execution_identity.json')
    if ec.sha(root/'src/encoder_comparison.py')!=prior_identity['module_sha256']:raise ValueError('Training logic changed')
    config={'epochs':30,'seeds':SEEDS,'learning_rate':.001,'batch_size':16,'clip':1,'width':64,'dropout':.1,'max_tokens':128}
    environment={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'transformers':__import__('transformers').__version__,'threads':torch.get_num_threads(),'device':'cpu'}
    if config!=prior_identity['configuration'] or environment!=prior_identity['environment']:raise ValueError('Control configuration/environment differs')
    if ec.sha(root/'data/processed/manifest.json')!=prior_identity['processed_manifest_sha256'] or ec.sha(root/'artifacts/features/manifest.json')!=prior_identity['legacy_feature_manifest_sha256']:raise ValueError('Control manifest differs')
    if {s:ec.sha(root/f'data/processed/{s}.jsonl') for s in ['train','valid']}!=prior_identity['split_source_sha256']:raise ValueError('Control processed source differs')
    datasets={cap:{s:CapDataset(root,out,s,cap) for s in ['train','valid']} for cap in [16,64]};gates(root,out,datasets)
    previous=ec.read(prior_dir/'results.json');reused=[];references={}
    if previous['prior']!=float(np.mean([r['target_up'] for r in datasets[64]['train'].rows])):raise ValueError('Training prevalence differs')
    for r in previous['runs']:
        if r['encoder']=='bge' or r['variant']=='price_only':
            folder=prior_dir/'runs'/r['run']
            if ec.read(folder/'summary.json')!=r:raise ValueError('Historical summary/results disagree')
            for name,digest in r['artifact_sha256'].items():
                if ec.sha(folder/name)!=digest:raise ValueError('Reused run artifact changed')
            reused.append(r);references[r['run']]={'directory':str(folder),'summary_sha256':ec.sha(folder/'summary.json'),'cap':16 if r['encoder']=='bge' else 'not applicable','newly_trained':False}
    if len(reused)!=9:raise ValueError('Expected six news and three price controls')
    expected={f'bge_{v}_{s}' for v in ['full','mean_pool'] for s in SEEDS}|{f'shared_price_only_{s}' for s in SEEDS}
    if {r['run'] for r in reused}!=expected:raise ValueError('Unexpected control identities')
    ec.write(out/'reused_controls.json',{'source_results_sha256':ec.sha(prior_dir/'results.json'),'source_identity':prior_identity,'runs':references})
    fresh=[]
    for variant in ['full','mean_pool']:
        for seed in SEEDS:
            r=ec.fit_run(out,'bge64',variant,seed,datasets[64],previous['prior']);fresh.append(r);ec.write(out/'partial_results.json',{'reused':reused,'fresh':fresh})
            old=next(x for x in reused if x['encoder']=='bge' and x['variant']==variant and x['seed']==seed)
            if r['minibatch_order_sha256']!=old['minibatch_order_sha256'] or r['parameter_counts']!=old['parameter_counts']:raise ValueError('Unmatched cap controls')
            a=torch.load(out/'runs'/r['run']/'initial.pt',weights_only=True);b=torch.load(prior_dir/'runs'/old['run']/'initial.pt',weights_only=True)
            for name,v in a.items():torch.testing.assert_close(v,b[name],atol=0,rtol=0)
    ec.write(out/'results.json',{'scope':'Exploratory cap16 versus64, train/valid only','prior':previous['prior'],'fresh_runs':fresh,'reused_runs':reused,'paired_orders_initial_weights_and_parameter_counts_verified':True})

def execute(root,out,stage):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    execution={'generator':identity(root),'training_module_sha256':ec.sha(root/'src/encoder_comparison.py'),'protocol_sha256':ec.sha(out/'PROTOCOL.md'),
        'script_sha256':ec.sha(root/'scripts/11_compare_headline_caps.py'),'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'transformers':__import__('transformers').__version__,'onnxruntime':ort.__version__,'threads':4,'device':'cpu'}}
    p=out/'execution_identity.json'
    if p.exists() and ec.read(p)!=execution:raise ValueError('Execution identity changed')
    if not p.exists():ec.write(p,execution)
    if not (out/'preservation_before.json').exists():raise ValueError('Preservation snapshot required before execution')
    ec.write(out/f'command_{stage}.json',{'command':f'.venv/Scripts/python.exe -B scripts/11_compare_headline_caps.py {stage}','git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True),'tracked_diff':subprocess.check_output(['git','diff','--no-ext-diff'],cwd=root,text=True)})
    if stage=='cache':cache(root,out)
    else:train(root,out)
    before=ec.read(out/'preservation_before.json');changed=[n for n,h in before.items() if not (root/n).exists() or ec.sha(root/n)!=h]
    ec.write(out/f'preservation_after_{stage}.json',{'checked':len(before),'changed':changed})
    if changed:raise ValueError('Original evidence changed')

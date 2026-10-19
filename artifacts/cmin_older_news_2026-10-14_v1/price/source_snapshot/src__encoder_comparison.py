"""Isolated CPU encoder experiment; legacy data, cache and training stay unchanged."""
from __future__ import annotations
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
from src.data.examples import MARKET_DESCRIPTION
from src.evaluation import classification_metrics
from src.training import make_model, set_seed
from src.training_data import PreparedDataset

MODEL = 'FacebookAI/roberta-base'
REVISION = 'e2da8e2f811d1448a5b465c236feacd80ffbac7b'
FILES = ['model.safetensors','config.json','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt']
SEEDS = [17,42,73]
SPLITS = ['train','valid']

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()

def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path); temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8');temporary.replace(path)

def rows(root,split):
    if split not in SPLITS: raise ValueError('This experiment only permits train/valid')
    return [json.loads(line) for line in (root/f'data/processed/{split}.jsonl').read_text(encoding='utf-8').splitlines()]

def process_memory():
    """Windows process RAM (includes native tensor allocations); no extra dependency."""
    if platform.system()!='Windows':return {'available':False}
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(n,ctypes.c_size_t) for n in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage']]
    kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=wintypes.HANDLE
    counters=Counters();counters.cb=ctypes.sizeof(counters)
    api=ctypes.WinDLL('psapi').GetProcessMemoryInfo
    api.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD];api.restype=wintypes.BOOL
    if not api(kernel.GetCurrentProcess(),ctypes.byref(counters),counters.cb):return {'available':False}
    return {'available':True,'working_set_bytes':counters.WorkingSetSize,'peak_working_set_bytes':counters.PeakWorkingSetSize,'scope':'Whole process RAM since launch, not isolated model memory'}

def items(root):
    result={}
    for split in SPLITS:
        for row in rows(root,split):
            for headline in row['headlines']:
                previous=result.setdefault(headline['id'],headline['text'])
                if previous!=headline['text']: raise ValueError('Headline ID collision')
    return sorted(result.items())

def cache_identity(root):
    return {'schema':1,'model':MODEL,'model_revision':REVISION,'tokenizer_revision':REVISION,
        'pooling':'last_hidden_state[:,0,:]','normalization':'none','dtype':'float32','width':768,
        'max_tokens':128,'tokenizer_use_fast':True,'add_prefix_space':False,'add_pooling_layer':False,
        'splits':SPLITS,'source_sha256':{s:sha(root/f'data/processed/{s}.jsonl') for s in SPLITS},
        'items_sha256':hashlib.sha256(json.dumps(items(root),ensure_ascii=False).encode('utf-8')).hexdigest(),
        'description':MARKET_DESCRIPTION,'generating_code_sha256':sha(Path(__file__))}

def encode(model,tokenizer,texts):
    full=tokenizer(texts,truncation=False,add_special_tokens=True)
    truncated=sum(len(x)>128 for x in full['input_ids'])
    tokens=tokenizer(texts,padding=True,truncation=True,max_length=128,return_tensors='pt')
    with torch.inference_mode(): vector=model(**tokens).last_hidden_state[:,0,:].cpu().numpy().astype(np.float32)
    if vector.shape!=(len(texts),768) or not np.isfinite(vector).all(): raise ValueError('Invalid encoder features')
    return vector,truncated

def build_cache(root,out):
    directory=out/'roberta_cache';directory.mkdir(exist_ok=True)
    identity=cache_identity(root);listing=items(root);state_path=directory/'state.json'
    if (directory/'manifest.json').exists():
        manifest=read(directory/'manifest.json')
        if manifest['identity']!=identity: raise ValueError('Completed cache identity mismatch')
        for name,digest in manifest['files_sha256'].items():
            if sha(directory/name)!=digest:raise ValueError('Completed cache content changed')
        print('Using verified existing encoder cache',flush=True);return
    if state_path.exists() and read(state_path)['identity']!=identity: raise ValueError('Partial cache identity mismatch')
    print('Loading pinned RoBERTa CPU weights',flush=True)
    snapshot=Path(snapshot_download(MODEL,revision=REVISION,allow_patterns=FILES,max_workers=2))
    model,loading=AutoModel.from_pretrained(str(snapshot),local_files_only=True,use_safetensors=True,
        add_pooling_layer=False,output_loading_info=True,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    if loading['missing_keys'] or loading.get('mismatched_keys') or loading.get('error_msgs'): raise ValueError(f'Encoder load defect: {loading}')
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,use_fast=True)
    guard_files={name:sha(snapshot/name) for name in FILES}
    loading_path=directory/'loading.json'
    if loading_path.exists() and read(loading_path)['files_sha256']!=guard_files:raise ValueError('Local model/tokenizer files changed')
    if not loading_path.exists():write(loading_path,{'snapshot':str(snapshot),'files_sha256':guard_files,'diagnostics':loading,
        'tokenizer_class':type(tokenizer).__name__,'model_class':type(model).__name__})
    # Warm-up is excluded from the benchmark, which measures inference without download.
    encode(model,tokenizer,[listing[0][1]])
    if not (directory/'benchmark.json').exists():
        start=time.monotonic();total_truncated=0;sample=[]
        for i in range(0,100,8):
            vec,n=encode(model,tokenizer,[t for _,t in listing[i:min(i+8,100)]])
            total_truncated+=n;sample.append(vec)
        elapsed=time.monotonic()-start
        sample=np.concatenate(sample)
        # Confirm embedding itself is invariant to adding padding in a longer batch.
        alone,_=encode(model,tokenizer,[listing[0][1]])
        paired,_=encode(model,tokenizer,[listing[0][1],MARKET_DESCRIPTION])
        error=float(np.abs(alone[0]-paired[0]).max())
        if error>1e-4: raise ValueError('Encoder padding invariance failed')
        write(directory/'benchmark.json',{'headlines':100,'batch_size':8,'threads':4,'device':'cpu',
            'seconds':elapsed,'estimated_cache_seconds':elapsed*len(listing)/100,
            'headlines_to_cache':len(listing),'truncated':total_truncated,'feature_norm_mean':float(np.linalg.norm(sample,axis=1).mean()),
            'mean_dimension_std':float(sample.std(0).mean()),'encoder_padding_max_abs_error':error,'process_ram':process_memory()})
        print(f'100-headline benchmark {elapsed:.2f}s; estimated cache {elapsed*len(listing)/100:.1f}s',flush=True)
    if state_path.exists():
        state=read(state_path);array=np.lib.format.open_memmap(directory/'text.npy',mode='r+')
        if state['model_files_sha256']!=guard_files or state['benchmark_sha256']!=sha(directory/'benchmark.json'):raise ValueError('Partial generator/benchmark identity changed')
        if list(array.shape)!=[len(listing),768]: raise ValueError('Partial shape mismatch')
        if read(directory/'headline_index.json')!=[h for h,_ in listing]:raise ValueError('Partial headline mapping changed')
    else:
        array=np.lib.format.open_memmap(directory/'text.npy',mode='w+',dtype=np.float32,shape=(len(listing),768))
        state={'identity':identity,'next':0,'truncated':0,'encoding_seconds':0.,'count':len(listing),
            'model_files_sha256':guard_files,'benchmark_sha256':sha(directory/'benchmark.json')}
        write(directory/'headline_index.json',[h for h,_ in listing]);write(state_path,state)
    session=time.monotonic();base_elapsed=state['encoding_seconds']
    for i in range(state['next'],len(listing),8):
        vector,n=encode(model,tokenizer,[t for _,t in listing[i:i+8]])
        array[i:i+len(vector)]=vector;array.flush()
        state.update(next=i+len(vector),truncated=state['truncated']+n,encoding_seconds=base_elapsed+time.monotonic()-session)
        write(state_path,state)
        if i%800==0: print(f'Cache {state["next"]}/{len(listing)}; {state["encoding_seconds"]:.1f}s',flush=True)
    description,truncated=encode(model,tokenizer,[MARKET_DESCRIPTION]);np.save(directory/'market_description.npy',description[0])
    norms=np.linalg.norm(array,axis=1)
    if not np.isfinite(array).all() or np.any(norms<=0) or array.std(0).mean()<=0: raise ValueError('Degenerate cache')
    manifest={'identity':identity,'files_sha256':{n:sha(directory/n) for n in ['text.npy','headline_index.json','market_description.npy']},
        'model_files_sha256':guard_files,'count':len(listing),'complete':True,'encoding_seconds':state['encoding_seconds'],
        'truncated_headlines':state['truncated'],'description_truncated':truncated,
        'norm_mean':float(norms.mean()),'norm_min':float(norms.min()),'norm_max':float(norms.max()),
        'mean_dimension_std':float(array.std(0).mean())}
    write(directory/'manifest.json',manifest);print('RoBERTa cache complete',flush=True)

class EncoderDataset(Dataset):
    def __init__(self,root,split,encoder,out):
        self.rows=rows(root,split);base=PreparedDataset(root,split)
        self.sentiment=base.sentiment;self.sentiment_index=base.index
        if encoder=='bge':
            self.text=base.text;self.index=base.index;self.description=base.description
        elif encoder=='roberta':
            directory=Path(read(out/'cache_reference.json')['directory']) if (out/'cache_reference.json').exists() else out/'roberta_cache'
            if (out/'cache_reference.json').exists() and sha(directory/'manifest.json')!=read(out/'cache_reference.json')['manifest_sha256']:raise ValueError('Referenced cache manifest changed')
            manifest=read(directory/'manifest.json');expected=cache_identity(root)
            if manifest['identity']['generating_code_sha256']!=expected['generating_code_sha256']:
                # Verified original generator is retained; replay-only downstream fixes cannot alter frozen features.
                snapshot=directory.parent/'code_snapshot.py'
                if sha(snapshot)!=manifest['identity']['generating_code_sha256']:raise ValueError('Original generating snapshot changed')
                original_prefix=snapshot.read_bytes().split(b'\nclass EncoderDataset')[0]
                current_prefix=Path(__file__).read_bytes().split(b'\nclass EncoderDataset')[0]
                if original_prefix!=current_prefix:raise ValueError('Referenced cache generator differs')
                expected['generating_code_sha256']=manifest['identity']['generating_code_sha256']
            if not manifest['complete'] or manifest['identity']!=expected: raise ValueError('Cache identity mismatch')
            for n,h in manifest['files_sha256'].items():
                if sha(directory/n)!=h: raise ValueError('Cache changed: '+n)
            self.text=np.load(directory/'text.npy',mmap_mode='r');self.description=np.load(directory/'market_description.npy')
            self.index={h:i for i,h in enumerate(read(directory/'headline_index.json'))}
        else: raise ValueError(encoder)
        self.width=self.text.shape[1]
        if self.description.shape!=(self.width,) or len(self.index)!=len(self.text): raise ValueError('Width/index mismatch')
        for row in self.rows:
            for h in row['headlines']:
                if h['id'] not in self.index or h['id'] not in self.sentiment_index: raise ValueError('ID mapping mismatch')
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        row=self.rows[index];text=np.zeros((16,self.width),np.float32);sent=np.zeros((16,3),np.float32);mask=np.zeros(16,bool)
        for j,h in enumerate(row['headlines']):
            text[j]=self.text[self.index[h['id']]];sent[j]=self.sentiment[self.sentiment_index[h['id']]];mask[j]=True
        return {'market_features':torch.tensor(row['market_features'],dtype=torch.float32),
            'market_description':torch.from_numpy(self.description.copy()),'headline_vectors':torch.from_numpy(text),
            'sentiment':torch.from_numpy(sent),'mask':torch.from_numpy(mask),'target':torch.tensor(row['target_up'],dtype=torch.float32),'row_index':index}

def native_model(variant,width):
    model=make_model(variant)
    if variant!='price_only' and width!=384:
        # Same downstream initial weights and subsequent RNG state; projection is necessarily different.
        with torch.random.fork_rng(devices=[]): model.encoders.text=nn.Linear(width,64)
    return model

def metrics(y,logits):
    y=np.asarray(y,dtype=float);logits=np.asarray(logits,dtype=float)
    p=1/(1+np.exp(-np.clip(logits,-700,700)))
    result=classification_metrics(y.tolist(),p.tolist())
    result['clipped_probability_log_loss']=result['log_loss']
    result['log_loss']=float(np.mean(np.logaddexp(0,logits)-y*logits))
    # Raw logits preserve ranking when sigmoid rounds extreme probabilities to 1/0.
    pos=logits[y==1];neg=logits[y==0]
    result.update(brier=float(np.mean((p-y)**2)),auc=float(np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:]))),
        predicted_down=int((p<.5).sum()),probability_mean=float(p.mean()),probability_std=float(p.std()))
    return result

@torch.inference_mode()
def infer(model,ds,mode=None,reference=None,permutation=None):
    model.eval();values=[]
    for batch in DataLoader(ds,batch_size=64,shuffle=False,num_workers=0):
        if permutation is not None:
            alternative=default_collate([ds[int(permutation[i])] for i in batch['row_index'].tolist()])
            for field in ['headline_vectors','sentiment','mask']:batch[field]=alternative[field]
        if mode=='news_mean':
            market,_=model.encoders(batch['market_description'],batch['market_features'],batch['headline_vectors'],batch['sentiment'])
            summary=reference.expand(len(market),-1)
            logits=model.predictor(torch.cat([market,summary],dim=-1)).squeeze(-1)
        else: logits,_=model(batch)
        values.extend(logits.double().tolist())
    return np.asarray(values,dtype=float)

def count_params(model):
    total=sum(p.numel() for p in model.parameters())
    # Mean pooling leaves the instantiated attention module unused in forward.
    unused=sum(p.numel() for p in model.attention.parameters()) if getattr(model,'variant',None)=='mean_pool' else 0
    return {'total':total,'active':total-unused,'unused_attention':unused}

def gates(root,out,datasets):
    path=out/'gates.json'
    if path.exists():
        if not read(path)['passed']: raise ValueError('Prior gate failed')
        return
    ds=datasets['roberta']['train'];indices=[]
    for label in [0,1]:indices += [i for i,r in enumerate(ds.rows) if r['target_up']==label][:16]
    indices.sort();write(out/'tiny_subset.json',{'indices':indices,'dates':[ds.rows[i]['news_date'] for i in indices]})
    base=default_collate([ds[i] for i in indices]);result={'passed':False,'tiny':[]}
    for split,data in datasets['roberta'].items():
        for i in range(len(data)):
            sample=data[i]
            for f in ['market_features','market_description','headline_vectors','sentiment']:
                if not torch.isfinite(sample[f]).all(): raise ValueError('Nonfinite real features')
            count=len(data.rows[i]['headlines'])
            if sample['mask'].sum()!=count: raise ValueError('Mask mismatch')
            old=datasets['bge'][split][i]
            torch.testing.assert_close(sample['sentiment'],old['sentiment'],atol=0,rtol=0)
            torch.testing.assert_close(sample['market_features'],old['market_features'],atol=0,rtol=0)
    # New-width real-feature gradient/update smoke, including an explicitly masked slot.
    set_seed(42);smoke=native_model('full',768);smoke.train()
    optimizer=torch.optim.Adam(smoke.parameters(),lr=.001)
    batch={k:v.clone() for k,v in base.items()};batch['mask'][0,-1]=False
    for f in ['market_features','headline_vectors','sentiment']:batch[f].requires_grad_(True)
    before={n:p.detach().clone() for n,p in smoke.named_parameters()}
    logits,_=smoke(batch);loss=nn.functional.binary_cross_entropy_with_logits(logits,batch['target']);loss.backward()
    blocks=['encoders.text','encoders.market_numbers','encoders.news_numbers','encoders.market_fusion','encoders.news_fusion','attention.self_attention','attention.normalization','attention.query','attention.key','predictor']
    grads={b:float(sum(p.grad.square().sum() for n,p in smoke.named_parameters() if n.startswith(b) and p.grad is not None).sqrt()) for b in blocks}
    if not all(np.isfinite(v) and v>0 for v in grads.values()):raise ValueError('Dead gradient block')
    for f in ['headline_vectors','sentiment']:
        if not torch.all(batch[f].grad[~batch['mask']]==0):raise ValueError('Nonzero padded input gradient')
    inputs={f:float(batch[f].grad.norm()) for f in ['market_features','headline_vectors','sentiment']}
    if not all(np.isfinite(v) and v>0 for v in inputs.values()):raise ValueError('Dead input')
    nn.utils.clip_grad_norm_(smoke.parameters(),1.);optimizer.step()
    updates={b:float(sum((p.detach()-before[n]).square().sum() for n,p in smoke.named_parameters() if n.startswith(b)).sqrt()) for b in blocks}
    if not all(np.isfinite(v) and v>0 for v in updates.values()):raise ValueError('Dead parameter update block')
    result['smoke']={'block_gradient_norms':grads,'block_update_norms':updates,'input_gradient_norms':inputs,'padded_input_gradients_zero':True,'optimizer_updates':1}
    for condition in ['full','prices_fixed']:
        set_seed(42);model=native_model('full',768);optimizer=torch.optim.Adam(model.parameters(),lr=.001)
        batch={k:(v.clone() if isinstance(v,torch.Tensor) else v) for k,v in base.items()}
        if condition=='prices_fixed':batch['market_features']=torch.tensor(np.mean([r['market_features'] for r in ds.rows],axis=0),dtype=torch.float32).expand_as(batch['market_features']).clone()
        for f in ['market_features','headline_vectors','sentiment']:batch[f].requires_grad_(True)
        history=[];gradients={};updates=0;passed=False
        for step in range(601):
            if step%25==0:
                model.eval()
                with torch.inference_mode(),torch.random.fork_rng(devices=[]):logits,_=model(batch)
                m=metrics(batch['target'].numpy(),logits.numpy());history.append({'updates':step,'metrics':m})
                if m['accuracy']>=.95 and m['log_loss']<=.1:passed=True;break
                if step==600:break
            model.train();optimizer.zero_grad(set_to_none=True)
            logits,_=model(batch);loss=nn.functional.binary_cross_entropy_with_logits(logits,batch['target'])
            if not torch.isfinite(loss): raise ValueError('Nonfinite tiny loss')
            loss.backward()
            if step==0:
                for name,param in model.named_parameters():gradients[name]=None if param.grad is None else float(param.grad.norm())
                for block in ['encoders.text','encoders.market_numbers','encoders.news_numbers','encoders.market_fusion','encoders.news_fusion','attention.self_attention','attention.query','attention.key','predictor']:
                    if not any(v is not None and np.isfinite(v) and v>0 for n,v in gradients.items() if n.startswith(block)):raise ValueError('Dead block: '+block)
                input_gradients={f:float(batch[f].grad.norm()) for f in ['market_features','headline_vectors','sentiment']}
                if not all(np.isfinite(x) and x>0 for x in input_gradients.values()):raise ValueError('Dead input')
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();updates=step+1
        torch.save(model.state_dict(),out/f'tiny_{condition}.pt')
        write(out/f'tiny_{condition}_predictions.json',[{'training_index':index,'news_date':ds.rows[index]['news_date'],'target_up':ds.rows[index]['target_up'],'logit':float(value),'probability_up':float(torch.sigmoid(value))} for index,value in zip(indices,logits)])
        original_condition='full' if condition=='full' else 'prices_fixed'
        if (out/'cache_reference.json').exists():
            old=Path(read(out/'cache_reference.json')['directory']).parent/f'tiny_{original_condition}.pt'
            original_state=torch.load(old,weights_only=True)
            for name,value in model.state_dict().items():torch.testing.assert_close(value,original_state[name],atol=0,rtol=0)
            result.setdefault('original_tiny_weights_exact',{})[condition]=True
        result['tiny'].append({'condition':condition,'passed':passed,'updates':updates,'history':history,'first_gradients':gradients,'input_gradients':input_gradients})
        write(path,result)
        if not passed: raise ValueError('Finite-budget memorization gate failed: '+condition)
    # Forward invariance and exact weight restoration, using the learned model in eval mode.
    model.eval();batch={k:(v.detach().clone() if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}
    batch['mask'][0,-1]=False
    with torch.inference_mode():
        logits,weights=model(batch)
        altered={k:v.clone() for k,v in batch.items()};altered['headline_vectors'][~batch['mask']]=500;altered['sentiment'][~batch['mask']]=-500
        padded,_=model(altered);torch.testing.assert_close(logits,padded,atol=1e-5,rtol=0)
        for f in ['headline_vectors','sentiment','mask']:altered[f]=batch[f].flip(1)
        permuted,_=model(altered);torch.testing.assert_close(logits,permuted,atol=1e-5,rtol=0)
        torch.testing.assert_close(weights.sum(1),torch.ones(len(weights)),atol=1e-6,rtol=0)
        if not torch.all(weights[~batch['mask']]==0): raise ValueError('Masked weights')
        restored=native_model('full',768).eval();restored.load_state_dict(torch.load(out/'tiny_prices_fixed.pt',weights_only=True))
        for n,v in model.state_dict().items():torch.testing.assert_close(v,restored.state_dict()[n],atol=0,rtol=0)
        replay,_=restored(batch);torch.testing.assert_close(logits,replay,atol=1e-6,rtol=1e-6)
        if not torch.equal(logits>=0,replay>=0):raise ValueError('Checkpoint replay changed a decision')
    result.update(passed=True,padding_error=float((logits-padded).abs().max()),permutation_error=float((logits-permuted).abs().max()),checkpoint_error=float((logits-replay).abs().max()),checkpoint_weights_exact=True,checkpoint_decisions_exact=True,replay_tolerance={'absolute':1e-6,'relative':1e-6})
    write(path,result);print('RoBERTa learning/integrity gates passed',flush=True)

def append_predictions(writer,run,epoch,split,ds,logits):
    for row,l in zip(ds.rows,logits):
        writer.writerow({k:row[k] for k in ['news_date','forecast_date','target_date','target_up']}|
            {'run':run,'epoch':epoch,'split':split,'logit':float(l),'probability_up':float(1/(1+np.exp(-np.clip(l,-700,700))))})

def fit_run(out,encoder,variant,seed,ds,prior):
    name=f'{encoder}_{variant}_{seed}';folder=out/'runs'/name;folder.mkdir(parents=True,exist_ok=True)
    if (folder/'summary.json').exists():
        summary=read(folder/'summary.json')
        for filename,digest in summary['artifact_sha256'].items():
            if sha(folder/filename)!=digest:raise ValueError('Saved run changed: '+name+'/'+filename)
        return summary
    if (folder/'predictions.csv').exists():raise ValueError('Partial run preserved; use new run namespace rather than overwrite: '+name)
    set_seed(seed);model=native_model(variant,ds['train'].width)
    torch.save(model.state_dict(),folder/'initial.pt')
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    loader=DataLoader(ds['train'],batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
    best=float('inf');best_epoch=None;best_scores=None;history=[];start=time.monotonic();order_digest=hashlib.sha256()
    target={s:np.array([r['target_up'] for r in d.rows]) for s,d in ds.items()}
    with (folder/'predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run','epoch','split','news_date','forecast_date','target_date','target_up','logit','probability_up']);writer.writeheader()
        for epoch in range(1,31):
            model.train();total=0.;n=0
            for batch in loader:
                order_digest.update(batch['row_index'].numpy().astype(np.int64).tobytes())
                optimizer.zero_grad(set_to_none=True);logits,_=model(batch)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,batch['target'])
                if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                total+=float(loss.detach())*len(logits);n+=len(logits)
            with torch.random.fork_rng(devices=[]):scores={s:infer(model,d) for s,d in ds.items()}
            evaluated={s:metrics(target[s],l) for s,l in scores.items()}
            history.append({'epoch':epoch,'online_train_loss':total/n,'eval':evaluated})
            validation=evaluated['valid']['log_loss']
            if validation<best:
                best=validation;best_epoch=epoch;best_scores={s:l.copy() for s,l in scores.items()};torch.save(model.state_dict(),folder/'selected.pt')
            for split,l in scores.items():append_predictions(writer,name,epoch,split,ds[split],l)
            handle.flush();write(folder/'history.json',history)
            if epoch%10==0:print(f'{name} epoch{epoch}: train {evaluated["train"]["log_loss"]:.5f}, valid {validation:.5f}',flush=True)
    torch.save(model.state_dict(),folder/'final.pt')
    model.load_state_dict(torch.load(folder/'selected.pt',weights_only=True));model.eval()
    with torch.random.fork_rng(devices=[]):selected_logits={s:infer(model,d) for s,d in ds.items()}
    selected={s:metrics(target[s],l) for s,l in selected_logits.items()}
    replay_errors={s:float(np.abs(selected_logits[s]-best_scores[s]).max()) for s in SPLITS}
    for s in SPLITS:
        np.testing.assert_allclose(selected_logits[s],best_scores[s],atol=1e-6,rtol=1e-6)
        if not np.array_equal(selected_logits[s]>=0,best_scores[s]>=0):raise ValueError('Selected replay changed decisions')
    prior_logit=float(np.log(prior/(1-prior)));constant=metrics(target['valid'],np.full(len(target['valid']),prior_logit))
    fallback=constant['log_loss']<=best
    summary={'run':name,'encoder':encoder,'variant':variant,'seed':seed,'epochs':30,'selected_epoch':best_epoch,
        'network':selected,'constant':constant,'fallback_to_constant':fallback,'fallback_selected_validation':constant if fallback else selected['valid'],
        'selected_epoch_validation_bce':best,'checkpoint_replay_max_logit_errors':replay_errors,
        'selection_metric':'stable float64 BCE from raw float32 logits, strict minimum, earliest exact tie',
        'parameter_counts':count_params(model),'minibatch_order_sha256':order_digest.hexdigest(),'elapsed_seconds':time.monotonic()-start}
    with (folder/'selected_predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run','epoch','split','news_date','forecast_date','target_date','target_up','logit','probability_up']);writer.writeheader()
        for s,l in selected_logits.items():append_predictions(writer,name,best_epoch,s,ds[s],l)
    with (folder/'fallback_predictions.csv').open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['run','split','news_date','forecast_date','target_date','target_up','network_probability_up','selected_system_probability_up','fallback']);writer.writeheader()
        for s,l in selected_logits.items():
            for row,logit in zip(ds[s].rows,l):
                p=float(1/(1+np.exp(-np.clip(logit,-700,700))))
                writer.writerow({k:row[k] for k in ['news_date','forecast_date','target_date','target_up']}|{'run':name,'split':s,'network_probability_up':p,'selected_system_probability_up':prior if fallback else p,'fallback':fallback})
    if variant!='price_only':
        with torch.inference_mode():
            summaries=[]
            for batch in DataLoader(ds['train'],batch_size=64,shuffle=False):
                market,news=model.encoders(batch['market_description'],batch['market_features'],batch['headline_vectors'],batch['sentiment'])
                news_summary,_=model.attention(market,news,batch['mask'],use_self_attention=variant!='mean_pool',use_market_attention=variant!='mean_pool')
                summaries.append(news_summary)
            reference=torch.cat(summaries).mean(0)
        write(folder/'news_reference.json',reference.tolist());controls={}
        with (folder/'sensitivity.csv').open('w',encoding='utf-8',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['condition','news_date','forecast_date','target_date','target_up','original_logit','altered_logit']);writer.writeheader()
            for condition in ['news_mean','permute_101','permute_202','permute_303']:
                permutation=None if condition=='news_mean' else np.random.default_rng(int(condition.split('_')[1])).permutation(len(ds['valid']))
                l=infer(model,ds['valid'],mode=condition,reference=reference,permutation=permutation)
                a=selected_logits['valid'];p=1/(1+np.exp(-np.clip(a,-700,700)));q=1/(1+np.exp(-np.clip(l,-700,700)))
                controls[condition]={'metrics':metrics(target['valid'],l),'decisions_changed':int(((a>=0)!=(l>=0)).sum()),'mean_abs_probability_change':float(np.abs(q-p).mean())}
                for row,original,altered in zip(ds['valid'].rows,a,l):writer.writerow({k:row[k] for k in ['news_date','forecast_date','target_date','target_up']}|{'condition':condition,'original_logit':original,'altered_logit':altered})
        summary['news_sensitivity']=controls
    summary['artifact_sha256']={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='summary.json'}
    write(folder/'summary.json',summary);return summary

def train(root,out):
    if (out/'results.json').exists():raise ValueError('Refusing to rerun completed experiment')
    datasets={e:{s:EncoderDataset(root,s,e,out) for s in SPLITS} for e in ['bge','roberta']}
    gates(root,out,datasets)
    prior=float(np.mean([r['target_up'] for r in datasets['bge']['train'].rows]))
    results=[]
    for encoder in ['bge','roberta']:
        for variant in ['full','mean_pool']:
            for seed in SEEDS:
                results.append(fit_run(out,encoder,variant,seed,datasets[encoder],prior));write(out/'partial_results.json',results)
    for seed in SEEDS:
        results.append(fit_run(out,'shared','price_only',seed,datasets['bge'],prior));write(out/'partial_results.json',results)
    for seed in SEEDS:
        orders={r['minibatch_order_sha256'] for r in results if r['seed']==seed}
        if len(orders)!=1:raise ValueError('Unmatched minibatch orders')
    write(out/'results.json',{'scope':'Exploratory training/validation encoder comparison; no test evaluation','runs':results,'prior':prior,'paired_orders_verified':True})

def execute(root,out,stage):
    out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    provenance={'stage':stage,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=root,text=True),'tracked_diff':subprocess.check_output(['git','diff','--no-ext-diff'],cwd=root,text=True),
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'transformers':__import__('transformers').__version__,'threads':4,'device':'cpu'},
        'module_sha256':sha(Path(__file__)),'script_sha256':sha(root/'scripts/10_compare_encoders.py'),
        'protocol_sha256':sha(out/'PROTOCOL.md'),'processed_manifest_sha256':sha(root/'data/processed/manifest.json'),
        'legacy_feature_manifest_sha256':sha(root/'artifacts/features/manifest.json'),
        'command':f'.venv/Scripts/python.exe -B scripts/10_compare_encoders.py {stage} --experiment {out.name}',
        'configuration':{'epochs':30,'seeds':SEEDS,'learning_rate':.001,'batch_size':16,'clip':1,'width':64,'dropout':.1,'max_tokens':128}}
    identity={k:provenance[k] for k in ['module_sha256','script_sha256','protocol_sha256','processed_manifest_sha256','legacy_feature_manifest_sha256','configuration']}
    identity['environment']=provenance['environment']
    if (out/'cache_reference.json').exists():identity['cache_reference_sha256']=sha(out/'cache_reference.json')
    identity['split_source_sha256']={s:sha(root/f'data/processed/{s}.jsonl') for s in SPLITS}
    identity_path=out/'execution_identity.json'
    if identity_path.exists() and read(identity_path)!=identity:raise ValueError('Immutable execution identity changed; retain old experiment and use a new version')
    if not identity_path.exists():write(identity_path,identity)
    p=out/f'provenance_{stage}.json'
    if not p.exists():write(p,provenance)
    if stage=='cache':build_cache(root,out)
    elif stage=='train':train(root,out)
    else:raise ValueError(stage)
    before=read(out/'preservation_before.json')
    changed=[n for n,h in before.items() if not (root/n).exists() or sha(root/n)!=h]
    write(out/f'preservation_after_{stage}.json',{'checked':len(before),'changed':changed})
    if changed:raise ValueError('Historical evidence changed: '+str(changed))

"""CPU research training with isolated evaluation RNG and stable logit metrics.

The historical trainer remains intact. This reusable alternative isolates
read-only diagnostics from dropout, sampling and module training modes. Actual
forecast availability and temporal splits are responsibilities of the caller.
"""
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.training import set_seed
from src.encoder_comparison import metrics

@contextmanager
def isolated_diagnostics(model):
    """Preserve CPU Torch/Python/NumPy randomness and every module's mode."""
    python_state=random.getstate();numpy_state=np.random.get_state()
    modes=[(module,module.training) for module in model.modules()]
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state);np.random.set_state(numpy_state)
        for module,mode in modes:module.training=mode

def predict_logits(model,dataset,batch_size=64):
    """Evaluate without consuming the training random stream or changing modes."""
    values=[]
    with isolated_diagnostics(model),torch.inference_mode():
        model.eval()
        generator=torch.Generator().manual_seed(20261013)
        for batch in DataLoader(dataset,batch_size=batch_size,shuffle=False,generator=generator,num_workers=0):
            logits,_=model(batch)
            if not torch.isfinite(logits).all():raise ValueError('Nonfinite diagnostic logits')
            values.extend(logits.double().cpu().tolist())
    return np.array(values,dtype=float)

def _optimizer(model, learning_rate, optimizer_name, weight_decay):
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('Positive finite learning rate required')
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('Nonnegative finite weight decay required')
    choices = {'adam': torch.optim.Adam, 'adamw': torch.optim.AdamW}
    if optimizer_name not in choices:
        raise ValueError('Optimizer must be adam or adamw')
    return choices[optimizer_name](model.parameters(), lr=learning_rate, weight_decay=weight_decay)

def fixed_budget_fit(model_factory,datasets,seed,epochs=30,batch_size=16,learning_rate=.001,clip=1.,extra_diagnostic=None,optimizer_name='adam',weight_decay=0.,progress_callback=None):
    """Train a caller-supplied model; retain all histories and selected/final states.

    This is a pure in-memory engine. The experiment runner must serialize its
    returned evidence into a fresh versioned namespace and retain provenance.
    No held-out financial test split is accepted, no threshold search is done,
    and a constant fallback is kept separate from neural network predictions.
    Datasets and callbacks must be deterministic/read-only. Callback model-state
    mutation is detected and restored; arbitrary dataset side effects cannot be
    isolated by the engine.
    """
    if set(datasets)!= {'train','valid'}:raise ValueError('Exactly train/valid datasets required; no test selection')
    if epochs<1 or batch_size<1:raise ValueError('Positive fixed budget required')
    set_seed(seed);model=model_factory()
    if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('This reproducible engine is CPU-only')
    initial={n:v.detach().clone() for n,v in model.state_dict().items()};optimizer=_optimizer(model,learning_rate,optimizer_name,weight_decay)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(datasets['train'],batch_size=batch_size,shuffle=True,generator=generator,num_workers=0)
    targets={s:np.array([float(d[i]['target']) for i in range(len(d))]) for s,d in datasets.items()}
    if any(len(y)==0 or not np.isfinite(y).all() or not np.isin(y,[0.,1.]).all() for y in targets.values()):
        raise ValueError('Nonempty finite binary targets required in every split')
    prior=float(targets['train'].mean())
    if not 0<prior<1:raise ValueError('Training requires both direction classes')
    history=[];predictions=[];selected=None;best=float('inf');selected_epoch=None
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True);logits,_=model(batch)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,batch['target'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),clip)
            if not torch.isfinite(norm):raise ValueError('Nonfinite training gradient')
            optimizer.step();total+=float(loss.detach())*len(logits);count+=len(logits)
        scores={s:predict_logits(model,d) for s,d in datasets.items()}
        evaluated={s:metrics(targets[s],l) for s,l in scores.items()}
        history.append({'epoch':epoch,'online_training_bce':total/count,'eval':evaluated});predictions.append({'epoch':epoch,'logits':scores})
        if evaluated['valid']['log_loss']<best:
            best=evaluated['valid']['log_loss'];selected_epoch=epoch
            selected={n:v.detach().clone() for n,v in model.state_dict().items()}
        if progress_callback is not None:
            # A copied completed-epoch record supports append-only evidence before
            # a budget stop/failure. Callback has no live model or engine buffers.
            import copy
            record = {'history': copy.deepcopy(history[-1]),
                      'logits': {s:l.copy() for s,l in scores.items()},
                      'selected_epoch': selected_epoch,
                      'selected_state': {n:v.detach().clone() for n,v in selected.items()}}
            with isolated_diagnostics(model):
                progress_callback(record)
        if extra_diagnostic is not None:
            before={n:v.detach().clone() for n,v in model.state_dict().items()}
            with isolated_diagnostics(model),torch.inference_mode():
                try:
                    model.eval();extra_diagnostic(model,datasets)
                finally:
                    changed=any(not torch.equal(v,model.state_dict()[n]) for n,v in before.items())
                    if changed:
                        model.load_state_dict(before)
                        raise ValueError('Read-only diagnostic mutated model parameters or buffers; state restored')
    final={n:v.detach().clone() for n,v in model.state_dict().items()}
    constant=metrics(targets['valid'],np.full(len(targets['valid']),np.log(prior/(1-prior))))
    model.load_state_dict(selected);model.eval();selected_logits={s:predict_logits(model,d) for s,d in datasets.items()}
    for s,l in selected_logits.items():np.testing.assert_allclose(l,predictions[selected_epoch-1]['logits'][s],atol=1e-6,rtol=1e-6)
    return {'model':model,'initial_state':initial,'selected_state':selected,'final_state':final,
            'selected_epoch':selected_epoch,'selected_logits':selected_logits,'history':history,'epoch_predictions':predictions,
            'training_prior':prior,'constant_validation':constant,'fallback_to_constant':constant['log_loss']<=best,
            'scope':'Computational fit; predictive validity requires appropriate caller-supplied chronology and evidence'}

def predict_fitted(fit,dataset):
    """Expose network and selected-system outputs separately, including fallback."""
    network=predict_logits(fit['model'],dataset)
    if fit['fallback_to_constant']:
        prior=fit['training_prior'];system=np.full(len(dataset),np.log(prior/(1-prior)))
    else:system=network.copy()
    return {'network_logits':network,'system_logits':system,'fallback':fit['fallback_to_constant']}

def fit_train_only(model_factory,train_dataset,seed,epochs,batch_size=16,learning_rate=.001,clip=1.,optimizer_name='adam',weight_decay=0.):
    """Refit for an inner-selected epoch count without accepting outer labels.

    Same CPU optimization and isolated train evaluation as fixed_budget_fit.
    Returns final weights and train histories, with no checkpoint selection,
    validation-based stopping, fallback selection or automatic serialization.
    """
    if epochs < 1 or batch_size < 1:
        raise ValueError('Positive fixed budget required')
    y=np.array([float(train_dataset[i]['target']) for i in range(len(train_dataset))])
    if len(y)==0 or not np.isfinite(y).all() or not np.isin(y,[0.,1.]).all() or len(np.unique(y))!=2:
        raise ValueError('Nonempty finite binary targets with both classes required')
    set_seed(seed);model=model_factory()
    if any(p.device.type!='cpu' for p in model.parameters()):
        raise ValueError('This reproducible engine is CPU-only')
    initial={n:v.detach().clone() for n,v in model.state_dict().items()}
    optimizer=_optimizer(model,learning_rate,optimizer_name,weight_decay)
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,generator=generator,num_workers=0)
    history=[]
    for epoch in range(1,epochs+1):
        model.train();total=0.;count=0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True);logits,_=model(batch)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,batch['target'])
            if not torch.isfinite(loss):raise ValueError('Nonfinite training objective')
            loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),clip)
            if not torch.isfinite(norm):raise ValueError('Nonfinite training gradient')
            optimizer.step();total+=float(loss.detach())*len(logits);count+=len(logits)
        history.append({'epoch':epoch,'online_training_bce':total/count,'train':metrics(y,predict_logits(model,train_dataset))})
    model.eval()
    return {'model':model,'initial_state':initial,'final_state':{n:v.detach().clone() for n,v in model.state_dict().items()},
            'history':history,'epochs':epochs,'training_prior':float(y.mean())}

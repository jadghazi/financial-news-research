"""Fixed-penalty convex price/coverage references; no validation search."""
import hashlib
import numpy as np
import torch


def matrix(dataset, coverage=False, news=False):
    values=[]
    for i in range(len(dataset)):
        row=dataset[i]
        price=row['market_features'].detach().cpu().numpy().astype(np.float64).reshape(-1)
        if coverage:
            counts=row['mask'].sum(-1).detach().cpu().numpy().astype(np.float64)
            price=np.concatenate((price,np.log1p(counts),(counts==0).astype(np.float64)))
        if news:
            mask=row['mask'].detach().cpu().numpy().astype(bool)
            text=row['headline_vectors'].detach().cpu().numpy().astype(np.float64)
            sentiment=row['sentiment'].detach().cpu().numpy().astype(np.float64)
            mean_text=text[mask].mean(0) if mask.any() else np.zeros(text.shape[-1])
            mean_sentiment=sentiment[mask].mean(0) if mask.any() else np.zeros(sentiment.shape[-1])
            price=np.concatenate((price,mean_text,mean_sentiment))
        values.append(price)
    result=np.asarray(values,dtype=np.float64)
    if result.ndim!=2 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Finite nonempty declared price/coverage inputs required')
    return result


def fit(dataset, coverage=False, news=False):
    raw=matrix(dataset,coverage,news)
    targets=np.array([float(dataset[i]['target']) for i in range(len(dataset))])
    if not np.isin(targets,[0.,1.]).all() or not 0<targets.mean()<1:
        raise ValueError('Training-only binary labels with both classes required')
    mean,scale=raw.mean(0),raw.std(0)
    scale=np.where(scale<1e-8,1.,scale)
    x=torch.from_numpy((raw-mean)/scale);y=torch.from_numpy(targets)
    parameter=torch.zeros(x.shape[1]+1,dtype=torch.float64,requires_grad=True)
    # Intercept initialization uses ONLY training prevalence.
    with torch.no_grad():parameter[-1]=float(np.log(targets.mean()/(1-targets.mean())))
    penalty=.01
    optimizer=torch.optim.LBFGS([parameter],max_iter=1000,tolerance_grad=1e-9,tolerance_change=1e-12,line_search_fn='strong_wolfe')
    calls=0
    def objective():
        nonlocal calls
        optimizer.zero_grad();logits=x@parameter[:-1]+parameter[-1]
        loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,y)+penalty/2*parameter[:-1].square().sum()
        if not torch.isfinite(loss):raise ValueError('Nonfinite convex objective')
        loss.backward();calls+=1
        return loss
    optimizer.step(objective)
    loss=objective();gradient=float(parameter.grad.abs().max())
    if gradient>1e-5:raise RuntimeError(f'Convex reference did not converge: gradient={gradient}')
    return dict(coverage=bool(coverage),news=bool(news),weights=parameter[:-1].detach().numpy().tolist(),
                intercept=float(parameter[-1].detach()),mean=mean.tolist(),scale=scale.tolist(),
                l2=penalty,objective='meanBCE + lambda/2 sum(weight squared); intercept unpenalized',
                selection='fixed penalty; no validation or evaluation target selection',
                fit_rows=len(targets),fit_input_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                fit_target_sha256=hashlib.sha256(targets.tobytes()).hexdigest(),
                objective_value=float(loss.detach()),max_abs_gradient=gradient,closure_calls=calls)


def predict(fitted,dataset):
    x=matrix(dataset,fitted['coverage'],fitted.get('news',False))
    mean,scale,weights=(np.asarray(fitted[key]) for key in ('mean','scale','weights'))
    if x.shape[1]!=len(weights):raise ValueError('Convex feature geometry changed')
    logits=((x-mean)/scale)@weights+fitted['intercept']
    if not np.isfinite(logits).all():raise ValueError('Nonfinite convex predictions')
    return logits

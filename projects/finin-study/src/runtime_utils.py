"""Shared hashing, deterministic seeding, and stable raw-logit metrics.

Code map and numerical contracts: docs/code_guide.md#shared-runtime-helpers.
These helpers were extracted unchanged from the historical implementation so
current CMIN training does not import retired NIFTY/StockNet experiments.
"""
from pathlib import Path
import hashlib
import random
import numpy as np
import torch
from src.evaluation import classification_metrics


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

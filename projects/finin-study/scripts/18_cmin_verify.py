"""Verify saved CMIN evidence and seal auxiliary files; no inference or fitting.

See docs/code_guide.md#pipeline-entry-points for the code map, contracts, and execution boundaries.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cmin_panel_data import read, write_once, sha256_file, rolling_plans
from src.cmin_panel_experiment import fit_scaler, score


def verify(directory):
    directory=Path(directory)
    config=read(directory/"config.json")
    summary=read(directory/"summary.json")
    checks=0
    def check_hashes(root, values):
        nonlocal checks
        for name,expected in values.items():
            if sha256_file(root/name)!=expected:
                raise ValueError(f"Saved artifact changed: {root/name}")
            checks+=1
    check_hashes(directory,read(directory/"completion.json"))
    check_hashes(directory,read(directory/"prepared.json")["files_sha256"])
    check_hashes(directory/"features",read(directory/"features"/"manifest.json")["files_sha256"])
    for fit in summary["fits"]:
        folder=directory/f"fold_{fit['fold']}"/f"{fit['arm']}_{fit['seed']}"
        check_hashes(folder,fit["files_sha256"])
    rows=read(directory/"rows.json")
    plans=read(directory/"plans.json")
    if plans!=rolling_plans(rows,config["windows"]):
        raise ValueError("Saved chronological plans differ")
    arrays=dict(np.load(directory/"daily_arrays.npz"))
    text=np.load(directory/"features"/"text.npy",mmap_mode="r")
    auxiliary=["source_date_disagreement_audit.json","engineering_checks.json","PRETRAIN_REVIEW.md"]
    for plan in plans:
        prefix=f"fold_{plan['fold']}"
        scaler=read(directory/prefix/"scaler.json")
        expected=fit_scaler(rows,plan["train"],arrays,text,config["history_sessions"])
        if scaler!=json.loads(json.dumps(expected)):
            raise ValueError("Saved scaler differs from training-only recomputation")
        def matrix(indices):
            values=[]
            for i in indices:
                r=rows[i];s,j=r["stock_index"],r["session_index"]
                start=j-config["history_sessions"]+1
                prices=(arrays["prices"][s,start:j+1]-np.array(scaler["price_mean"],np.float32))/np.array(scaler["price_std"],np.float32)
                identity=np.zeros(len(config["tickers"]));identity[s]=1
                values.append(np.r_[1.,prices.ravel(),identity])
            return np.array(values,dtype=np.float64)
        linear=read(directory/prefix/"linear_prices_identity.json")
        beta=np.array(linear["beta"])
        x=matrix(plan["train"]);y=np.array([rows[i]["target"] for i in plan["train"]])
        probabilities=1/(1+np.exp(-np.clip(x@beta,-50,50)))
        penalty=beta.copy();penalty[0]=0
        if np.abs(x.T@(probabilities-y)+penalty).max()>1e-5:
            raise ValueError("Saved logistic reference is not converged")
        logits=matrix(plan["evaluation"])@beta
        np.testing.assert_allclose(logits,linear["evaluation_logits"],atol=1e-10,rtol=1e-10)
        metric=score([rows[i]["target"] for i in plan["evaluation"]],logits)
        if abs(metric["log_loss"]-linear["evaluation"]["log_loss"])>1e-12:
            raise ValueError("Saved logistic metric differs")
        auxiliary+=[prefix+"/scaler.json",prefix+"/linear_prices_identity.json"]
    record={"files_sha256":{name:sha256_file(directory/name) for name in auxiliary},
            "checks":"Both scalers exactly recomputed; both fixed logistic references satisfy gradient tolerance and saved predictions; no model inference or fitting"}
    path=directory/"auxiliary_audit_hashes.json"
    if path.exists():
        check_hashes(directory,read(path)["files_sha256"])
        if read(path)!=record:
            raise ValueError("Auxiliary audit differs; preserve original")
    else:
        write_once(path,record)
    return {"verified_artifact_hashes":checks,"verified_scalers":len(plans),"verified_logistic_references":len(plans),
            "auxiliary_files_sealed":len(auxiliary)}


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--directory",default="artifacts/cmin_panel_2026-10-14_v1")
    print(json.dumps(verify(parser.parse_args().directory),indent=2))

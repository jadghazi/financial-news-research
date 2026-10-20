"""Small retrospective validation-input dependency check; no model fitting.

Keep prices, company context and masks fixed for text/sentiment probes. These
out-of-distribution substitutions establish computational dependence only.
Attention switches use existing trained weights; they are not retrained ablations.

See docs/code_guide.md#pipeline-entry-points for the code map, contracts, and execution boundaries.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cmin_panel_data import read, write_once, sha256_file
from src.cmin_panel_experiment import PanelDataset, make_model


def run(directory, output):
    directory,output=Path(directory),Path(output)
    if output.exists():
        raise ValueError("Diagnostic output must be new; existing evidence retained")
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    config=read(directory/"config.json")
    rows=read(directory/"rows.json")
    plans=read(directory/"plans.json")
    arrays=dict(np.load(directory/"daily_arrays.npz"))
    text=np.load(directory/"features"/"text.npy",mmap_mode="r")
    sentiment=np.load(directory/"features"/"sentiment.npy",mmap_mode="r")
    records=[]
    source_hashes={name:sha256_file(directory/name) for name in
                  ["config.json","plans.json","rows.json","prepared.json","features/manifest.json"]}
    for plan in plans:
        folder=directory/f"fold_{plan['fold']}"
        scaler=read(folder/"scaler.json")
        dataset=PanelDataset(rows,plan["valid"],arrays,text,sentiment,scaler,config["history_sessions"])
        selected=[]
        for ticker in config["tickers"]:
            candidates=[i for i,r in enumerate(dataset.rows) if r["ticker"]==ticker]
            selected.append(next((i for i in candidates if
                (arrays["news_indices"][dataset.rows[i]["stock_index"],
                 dataset.rows[i]["session_index"]-4:dataset.rows[i]["session_index"]+1]>=0).sum()>=2),candidates[0]))
        batch=next(iter(DataLoader(torch.utils.data.Subset(dataset,selected),batch_size=len(selected),shuffle=False)))
        mean_sentiment=torch.from_numpy(np.asarray(sentiment[scaler["training_text_indices"]]).mean(0).astype(np.float32))
        for seed in config["seeds"]:
            fitfolder=folder/f"finin_{seed}"
            summary=read(fitfolder/"summary.json")
            checkpoint=fitfolder/"selected.pt"
            if sha256_file(checkpoint)!=summary["files_sha256"]["selected.pt"]:
                raise ValueError("Selected checkpoint changed")
            source_hashes[str(checkpoint.relative_to(directory))]=sha256_file(checkpoint)
            model=make_model(config,"finin",summary["training_prior"])
            model.load_state_dict(torch.load(checkpoint,weights_only=True));model.eval().requires_grad_(False)
            with torch.inference_mode():
                original=model.trace(batch)
                reference=np.load(fitfolder/"selected_logits.npz")["valid"][selected]
                np.testing.assert_allclose(original["logits"].numpy(),reference,rtol=1e-6,atol=1e-6)
                changed=dict(batch);changed["headline_vectors"]=torch.zeros_like(batch["headline_vectors"])
                text_logits=model(changed)[0].numpy().copy()
                changed=dict(batch);changed["sentiment"]=mean_sentiment.expand_as(batch["sentiment"]).clone()
                sentiment_logits=model(changed)[0].numpy().copy()
                changed=dict(batch);changed["mask"]=torch.zeros_like(batch["mask"])
                empty_logits=model(changed)[0].numpy().copy()
                model.query_weighting=False
                uniform_logits=model(batch)[0].numpy().copy();model.query_weighting=True
                model.interaction=False
                interaction_off_logits=model(batch)[0].numpy().copy();model.interaction=True
            gradbatch={k:v.clone() for k,v in batch.items()}
            gradbatch["headline_vectors"].requires_grad_(True)
            model(gradbatch)[0].sum().backward()
            gradient=gradbatch["headline_vectors"].grad.flatten(1).norm(dim=1).numpy()
            def probabilities(z):return 1/(1+np.exp(-z))
            base=probabilities(reference)
            for i,index in enumerate(selected):
                row=dataset.rows[index]
                record={"fold":plan["fold"],"seed":seed,"ticker":row["ticker"],"forecast_date":row["forecast_date"],
                        "target_date":row["target_date"],"active_headlines":int(batch["mask"][i].sum()),
                        "original_probability":float(base[i]),"headline_gradient_norm":float(gradient[i]),
                        "probability_changes":{name:float(probabilities(z)[i]-base[i]) for name,z in
                         [("headline_training_mean",text_logits),("sentiment_training_mean",sentiment_logits),
                          ("empty_news",empty_logits),("query_uniform",uniform_logits),("interaction_off",interaction_off_logits)]}}
                records.append(record)
    active=[r for r in records if r["active_headlines"]>0]
    summary={"validation_examples":len(records),"examples_with_news":len(active),
             "selection":"first validation example with >=2 active headlines per company, else first empty case; no outcome-based selection",
             "scope":"retrospective computational dependency diagnostic; no forecasting assessment, no refits, no held-out period opened",
             "gradient_nonzero_examples":sum(r["headline_gradient_norm"]>1e-12 for r in active),
             "probes":{name:{"examples_changed_over_1e_minus_6":sum(abs(r["probability_changes"][name])>1e-6 for r in active),
                       "maximum_absolute_probability_change":max(abs(r["probability_changes"][name]) for r in active)}
                       for name in active[0]["probability_changes"]},
             "source_hashes":source_hashes,"diagnostic_code_sha256":sha256_file(Path(__file__)),
             "utc":datetime.now(timezone.utc).isoformat()}
    write_once(output/"examples.json",records);write_once(output/"summary.json",summary)
    return summary


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--directory",default="artifacts/cmin_panel_2026-10-14_v1")
    parser.add_argument("--output",default="artifacts/diagnostics/cmin_news_consumption_2026-10-14")
    args=parser.parse_args();result=run(args.directory,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in {"source_hashes"}},indent=2))

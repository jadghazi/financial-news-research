"""One frozen evaluation on previously unsealed, score-unused dates; no fitting.

The first attempt unsealed source rows but stopped at an encoder provenance
check before predictions. These dates must not be described as still sealed.
The main agent must freeze selection_before_unseal.json before calling run.
Selected checkpoints, preprocessing and reference models are hashed before
reading test targets. No epochs, thresholds, PCA, scalers or models are chosen
from test outcomes. Subsequent diagnostics are descriptive, not a new selection.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from src.models.paper_finin import PaperFININ
from src.pipeline_diagnosis import linear_predict
from src.reproduction_v2_data import prepare, sha256_file
from src.reproduction_v2_features import ENCODERS, load_frozen, apply_pca, _validate_values, _verify_models
from src.reproduction_v2_matched_control import MatchedNoNews
from src.reproduction_v2_neural import CANDIDATES, PriceIdentity, Samples, ProjectedFeatures
from src.reproduction_v2_probe import (bootstrap_difference, controls, disruption, evaluated,
                                       prediction_file, prior_logits, write)
from src.research_training import predict_logits


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    write(temporary, value)
    temporary.replace(path)


class FinalFeatures:
    """Only frozen test-message vectors; no trainable preprocessing here."""
    def __init__(self, folder):
        self.ids = _read(folder / "index.json")
        self.index = {identifier: i for i, identifier in enumerate(self.ids)}
        self.text = np.load(folder / "text.npy", mmap_mode="r")
        self.sentiment = np.load(folder / "sentiment.npy", mmap_mode="r")

    def vectors(self, ids):
        return np.asarray(self.text[[self.index[i] for i in ids]])

    def sentiments(self, ids):
        return np.asarray(self.sentiment[[self.index[i] for i in ids]])

    def close(self):
        self.text._mmap.close(); self.sentiment._mmap.close()


def linear_news_design(rows, features, name, donors=None):
    """Exact probe design: identity, counts, optional prices, raw news means.

    Donors alter only the text+sentiment block, preserving each recipient's
    identity, message counts and price features. No fit or normalization occurs.
    """
    if name not in {"news", "prices_news"}:
        raise ValueError("Only frozen news/prices_news linear designs supported")
    means = []
    for row in rows:
        identifiers = [message["id"] for message in row["headlines"]]
        if identifiers:
            means.append((features.vectors(identifiers).mean(0), features.sentiments(identifiers).mean(0)))
        else:
            means.append((np.zeros(features.text.shape[1], np.float32), np.zeros(3, np.float32)))
    news = np.column_stack([np.array([mean[0] for mean in means]), np.array([mean[1] for mean in means])])
    if donors is not None:
        donors = np.asarray(donors)
        if donors.shape != (len(rows),) or donors.dtype.kind not in "iu" or np.any(donors < 0) or np.any(donors >= len(rows)):
            raise ValueError("Invalid linear news donor indices")
        if any(rows[i]["ticker"] != rows[d]["ticker"] or len(rows[i]["headlines"]) != len(rows[d]["headlines"]) for i, d in enumerate(donors)):
            raise ValueError("Linear news donors must preserve ticker and exact message count")
        news = news[donors]
    identity, coverage, prices = controls(rows)
    base = np.column_stack([identity, coverage])
    return np.column_stack([base, prices, news]) if name == "prices_news" else np.column_stack([base, news])


def _cache(test_rows, folder, execution_signature, development_manifest):
    items = {}
    for rows in test_rows.values():
        for row in rows:
            if row["split"] != "test":
                raise ValueError("Final feature cache requires only test rows")
            for identifier, text in [(f"description:{row['ticker']}", row["description"])] + [(h["id"], h["text"]) for h in row["headlines"]]:
                if identifier in items and items[identifier] != text:
                    raise ValueError("Final text identity collision")
                if not identifier.startswith("description:") and hashlib.sha256(text.encode()).hexdigest() != identifier:
                    raise ValueError("Final message content hash differs")
                items[identifier] = text
    listing = sorted(items.items())
    identity = {"execution_signature": execution_signature, "items_sha256": hashlib.sha256(json.dumps(listing, ensure_ascii=False).encode()).hexdigest(),
                "encoder": "pinned offline RoBERTa CLS raw", "sentiment_order": ["positive", "neutral", "negative"],
                "max_tokens": 128, "batch_size": 16, "device": "cpu", "threads": 4, "label_independent": True}
    folder.mkdir(exist_ok=True)
    if (folder / "identity.json").exists():
        if _read(folder / "identity.json") != identity or _read(folder / "index.json") != [i for i, _ in listing]:
            raise ValueError("Final cache identity changed")
    else:
        if any(folder.iterdir()):
            raise FileExistsError("Untracked final cache files exist")
        _atomic(folder / "identity.json", identity); _atomic(folder / "index.json", [i for i, _ in listing])
    states = {}
    for kind, width in (("text", 768), ("sentiment", 3)):
        path, state_path = folder / f"{kind}.npy", folder / f"{kind}_state.json"
        state = _read(state_path) if state_path.exists() else None
        if state:
            if state["identity"] != identity:
                raise ValueError("Final generator identity changed")
            _verify_models(state["model_provenance"])
            array = np.lib.format.open_memmap(path, mode="r+")
            if array.shape != (len(listing), width) or array.dtype != np.float32:
                raise ValueError("Final feature geometry changed")
            cursor = 0
            for chunk in state["chunks"]:
                if chunk["start"] != cursor or chunk["stop"] <= cursor:
                    raise ValueError("Invalid final committed chunks")
                if hashlib.sha256(array[cursor:chunk["stop"]].tobytes()).hexdigest() != chunk["sha256"]:
                    raise ValueError("Final committed feature prefix changed")
                cursor = chunk["stop"]
            if cursor != state["next"]:
                raise ValueError("Final cache cursor changed")
            if cursor == len(listing):
                if sha256_file(path) != state["sha256"]:
                    raise ValueError("Completed final feature checksum changed")
                array._mmap.close(); states[kind] = state; continue
        else:
            if path.exists():
                raise FileExistsError("Untracked final array exists")
        encoder = load_frozen(kind, "roberta", 128, False)
        if encoder.provenance != development_manifest['states'][kind]['model_provenance']:
            raise ValueError('Test encoder differs from frozen development generator')
        if state and state["model_provenance"] != encoder.provenance:
            raise ValueError("Final frozen model changed on resume")
        if state is None:
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(listing), width))
            state = {"identity": identity, "next": 0, "chunks": [], "truncated": 0, "seconds": 0., "model_provenance": encoder.provenance}
            _atomic(state_path, state)
        start_chunk, started = state["next"], time.monotonic()
        for offset in range(state["next"], len(listing), 16):
            values, truncated = encoder.encode([text for _, text in listing[offset:offset + 16]])
            _validate_values(values, min(16, len(listing) - offset), width, kind)
            array[offset:offset + len(values)] = values; state["truncated"] += truncated
            stop = offset + len(values)
            if stop - start_chunk >= 512 or stop == len(listing):
                array.flush()
                state["chunks"].append({"start": start_chunk, "stop": stop, "sha256": hashlib.sha256(array[start_chunk:stop].tobytes()).hexdigest()})
                state["next"] = stop; state["seconds"] += time.monotonic() - started
                if stop == len(listing): state["sha256"] = sha256_file(path)
                _atomic(state_path, state); start_chunk, started = stop, time.monotonic()
                print(f"final {kind}: {stop:,}/{len(listing):,}", flush=True)
        array._mmap.close(); states[kind] = state
        del encoder
    manifest = {"identity": identity, "states": states, "items": len(listing),
                "files_sha256": {name: sha256_file(folder / name) for name in ("identity.json", "index.json", "text.npy", "sentiment.npy", "text_state.json", "sentiment_state.json")}}
    _atomic(folder / "manifest.json", manifest)
    return FinalFeatures(folder)


def run(out: Path) -> dict:
    out = Path(out)
    directory = out / "final_evaluation"
    selection_path = directory / "selection_before_unseal.json"
    if not selection_path.is_file():
        raise ValueError("Main agent must freeze model choices before unsealing")
    selection = _read(selection_path)
    pairs = selection.get("pairs", selection.get("models", {s: selection.get(s, []) for s in ("pre_open", "post_close")}))
    allowed = {spec["name"]: spec for spec in CANDIDATES}
    if set(pairs) != {"pre_open", "post_close"} or any(not models or len(set(models)) != len(models) or any(name not in allowed for name in models) for models in pairs.values()):
        raise ValueError("Invalid predeclared schedules/model choices")
    matched = selection.get("matched_no_news", False)
    if not isinstance(matched, bool): raise ValueError("matched_no_news must be boolean")
    linear_news = selection.get("linear_news", {})
    if not isinstance(linear_news, dict) or any(schedule not in pairs or not isinstance(names, list) or len(set(names)) != len(names) or any(name not in {"news", "prices_news"} for name in names) for schedule, names in linear_news.items()):
        raise ValueError("Invalid frozen linear-news choices")
    source_root = Path(__file__).resolve().parents[1]
    hashes = {}
    dependencies = ["src/reproduction_v2_final.py", "src/reproduction_v2_neural.py", "src/reproduction_v2_features.py",
                    "src/reproduction_v2_data.py", "src/reproduction_v2_probe.py", "src/models/paper_finin.py",
                    "src/reproduction_v2_matched_control.py", "src/research_training.py", "src/pipeline_diagnosis.py",
                    "src/encoder_comparison.py", "src/evaluation.py"]
    for name in dependencies: hashes[str(source_root / name)] = sha256_file(source_root / name)
    development_manifest = _read(out/'roberta_cache'/'manifest.json')
    if development_manifest['identity']['encoder']!=ENCODERS['roberta']:
        raise ValueError('Requested final RoBERTa is not the cached development encoder')
    provenance=development_manifest['states']['text']['model_provenance']
    if (provenance['model'],provenance['revision'])!=(ENCODERS['roberta']['name'],ENCODERS['roberta']['revision']):
        raise ValueError('Development generator does not match requested pinned RoBERTa')
    actual=np.load(out/'roberta_cache'/'text.npy',mmap_mode='r')
    if actual.shape[1]!=768:raise ValueError('Actual development vector width is not RoBERTa768')
    actual._mmap.close()
    hashes[str((out/'roberta_cache'/'manifest.json').resolve())]=sha256_file(out/'roberta_cache'/'manifest.json')
    for state in development_manifest['states'].values():_verify_models(state['model_provenance'])
    for schedule, names in pairs.items():
        paths = [out / f"neural_roberta/{schedule}/preprocessing.json", out / f"neural_roberta/{schedule}/pca_transform.npz",
                 out / f"probe_bge_solver2/{schedule}/coverage/model.json", out / f"probe_bge_solver2/{schedule}/prices/model.json"]
        for name in names:
            paths += [out / f"neural_roberta/{schedule}/{name}_{seed}/selected.pt" for seed in (17, 42, 73)]
        if matched: paths += [out / f"matched_no_news/{schedule}/no_news_{seed}/selected.pt" for seed in (17, 42, 73)]
        if linear_news.get(schedule):
            probe_folder = out / "probe_roberta_solver2"
            probe_configuration = _read(probe_folder / "configuration_before_fits.json")
            if probe_configuration["representation"] != "roberta":
                raise ValueError("Frozen linear model was not fitted using RoBERTa")
            expected_sources = {s: sha256_file(out / f"stocknet_dev/devsamples_{s}.jsonl") for s in ("post_close", "pre_open")}
            if probe_configuration["source_sha256"] != expected_sources:
                raise ValueError("Frozen linear model development sources differ")
            paths += [probe_folder / "configuration_before_fits.json", probe_folder / schedule / "split.json"]
            for name in linear_news[schedule]:
                paths += [probe_folder / schedule / name / filename for filename in ("model.json", "summary.json", "valid_predictions.csv")]
        for path in paths:
            hashes[str(path.resolve())] = sha256_file(path)
            if path.name=='selected.pt':hashes[str((path.parent/'summary.json').resolve())]=sha256_file(path.parent/'summary.json')
    for name in ("devsamples_post_close.jsonl", "devsamples_pre_open.jsonl", "provenance.json"):
        path = out / "stocknet_dev" / name; hashes[str(path.resolve())] = sha256_file(path)
    source = source_root / "data/research_v2/stocknet-source"
    dev_manifest = _read(out / "stocknet_dev/provenance.json")
    for name, checksum in dev_manifest["source_sha256"].items():
        path = source / name
        if sha256_file(path) != checksum: raise ValueError("Raw source changed before unsealing")
    execution = {"selection_sha256": sha256_file(selection_path), "fixed_pairs": pairs, "matched_no_news": matched, "linear_news": linear_news,
                 "dependency_and_artifact_sha256": hashes, "raw_source_sha256": dev_manifest["source_sha256"],
                 "seeds": [17, 42, 73], "epoch_selection": "existing validation-selected checkpoint only; no test selection or refitting",
                 "preprocessing": "reuse original training-only market scaler/PCA; no adaptation",
                 "permutation": "diagnostic test-only ticker/exact-count-matched joint text+sentiment disruption",
                 "limitations": "one previously unsealed, score-unused target quarter; earlier attempt stopped before predictions on encoder provenance mismatch; descriptive block intervals, no multiplicity correction, no tradability claim",
                 "environment": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__}}
    frozen_path = directory / "execution_before_unseal.json"
    if frozen_path.exists():
        if _read(frozen_path) != execution: raise ValueError("Frozen final-evaluation identity changed")
    else: write(frozen_path, execution)
    if (directory / "results.json").exists(): raise FileExistsError("Completed test evaluation retained; do not rerun selection")
    unsealed = directory / "stocknet_unsealed"
    if not unsealed.exists(): prepare(source_root, unsealed, include_test=True)
    prepared_identity = {name: sha256_file(unsealed / name) for name in ("samples_post_close.jsonl", "samples_pre_open.jsonl", "provenance.json")}
    prepared_identity_path = directory / "unsealed_source_identity.json"
    if prepared_identity_path.exists():
        if _read(prepared_identity_path) != prepared_identity: raise ValueError("Unsealed prepared data changed during recovery")
    else: write(prepared_identity_path, prepared_identity)
    for name, checksum in dev_manifest["source_sha256"].items():
        if _read(unsealed / "provenance.json")["source_sha256"].get(name) != checksum: raise ValueError("Unsealed source provenance differs")
    rows_by_schedule = {schedule: [row for row in [json.loads(line) for line in (unsealed / f"samples_{schedule}.jsonl").read_text().splitlines()] if row["split"] == "test"] for schedule in pairs}
    torch.set_num_threads(4)
    features = _cache(rows_by_schedule, directory / "roberta_test_cache", sha256_file(frozen_path),development_manifest)
    all_results = {}
    try:
        for schedule, names in pairs.items():
            rows = rows_by_schedule[schedule]; indices = np.arange(len(rows))
            folder = directory / schedule
            if folder.exists(): raise FileExistsError("Partial final predictions retained; recover separately without new selection")
            folder.mkdir()
            preprocessing = _read(out / f"neural_roberta/{schedule}/preprocessing.json")
            if any(row["target_available_utc"] <= row["cutoff_utc"] for row in rows):
                raise ValueError("Final target was available by forecast cutoff")
            market = (np.array([r["market_raw"] for r in rows]) - preprocessing["market_mean"]) / preprocessing["market_scale"]
            pca = np.load(out / f"neural_roberta/{schedule}/pca_transform.npz")
            # Preserve development's cast-before-coordinate-scaling recipe exactly.
            projected = apply_pca(features.text,{'mean':pca['mean'],'basis':pca['basis']}) / pca['scale']
            sources = {"raw": features, "pca32": ProjectedFeatures(features, projected)}
            datasets = {kind: Samples(rows, indices, source_features, market,
                                      {ticker: source_features.text[features.index[f"description:{ticker}"]] for ticker in {r["ticker"] for r in rows}})
                        for kind, source_features in sources.items()}
            dev_rows = [json.loads(line) for line in (out / f"stocknet_dev/devsamples_{schedule}.jsonl").read_text().splitlines()]
            tr = np.array(preprocessing["training_indices"])
            if any(dev_rows[i]["split"] != "train" or dev_rows[i]["target_available_utc"] >= min(row["cutoff_utc"] for row in rows) for i in tr):
                raise ValueError("Reference fitting labels overlap final forecast availability")
            scores, results = {}, {}
            # Reuse original fitting outcomes for priors; no validation/test labels fitted.
            combined = dev_rows + rows; final_indices = np.arange(len(dev_rows), len(combined))
            for name, stock_specific in (("global_prior", False), ("stock_prior", True)):
                logits, state = prior_logits(combined, tr, final_indices, stock_specific)
                scores[name] = logits; results[name] = {"metrics": evaluated(rows, indices, logits), "state": state}
                prediction_file(folder / f"{name}_predictions.csv", rows, indices, logits)
            identity, coverage, relative = controls(rows)
            for name, design in (("coverage", np.column_stack([identity, coverage])), ("prices", np.column_stack([identity, coverage, relative]))):
                logits = linear_predict(_read(out / f"probe_bge_solver2/{schedule}/{name}/model.json"), design)
                scores[name] = logits; results[name] = {"metrics": evaluated(rows, indices, logits)}
                prediction_file(folder / f"{name}_predictions.csv", rows, indices, logits)
            for name in linear_news.get(schedule, []):
                result_name = f"linear_{name}"
                model_path = out / f"probe_roberta_solver2/{schedule}/{name}/model.json"
                model = _read(model_path)
                original_split = _read(out / f"probe_roberta_solver2/{schedule}/split.json")
                if original_split["train"] != tr.tolist():
                    raise ValueError("Frozen linear model fitting partition differs from original neural training")
                design = linear_news_design(rows, features, name)
                logits = linear_predict(model, design)
                donor = disruption(rows, indices)
                changed = linear_predict(model, linear_news_design(rows, features, name, donor))
                prediction_file(folder / f"{result_name}_predictions.csv", rows, indices, logits)
                prediction_file(folder / f"{result_name}_permuted_predictions.csv", rows, indices, changed)
                write(folder / f"{result_name}_permutation.json", {"donors": donor.tolist(), "scope": "same ticker/exact count, final period only"})
                scores[result_name] = logits
                results[result_name] = {"metrics": evaluated(rows, indices, logits), "model_sha256": sha256_file(model_path),
                                        "permuted_metrics": evaluated(rows, indices, changed),
                                        "permutation_loss_change_ci": bootstrap_difference(rows, indices, changed, logits),
                                        "permutation_decisions_changed": int(((logits >= 0) != (changed >= 0)).sum()),
                                        "changed_donors": int((donor != indices).sum()),
                                        "comparisons": {reference: bootstrap_difference(rows, indices, logits, scores[reference])
                                                        for reference in ("global_prior", "stock_prior", "coverage", "prices")}}
            for name in names + (["matched_no_news"] if matched else []):
                spec = allowed.get(name, {"variant": "full", "representation": "raw", "attention_style": "plain"})
                dataset = datasets[spec["representation"]]
                donor = disruption(rows, indices)
                disrupted = Samples(rows, indices, sources[spec["representation"]], market,
                                    {ticker: sources[spec["representation"]].text[features.index[f"description:{ticker}"]] for ticker in {r["ticker"] for r in rows}})
                disrupted.text = dataset.text[donor].clone(); disrupted.sent = dataset.sent[donor].clone()
                logits_group, permuted_group, run_results = [], [], []
                for seed in (17, 42, 73):
                    if name == "matched_no_news": model = MatchedNoNews(text_width=768)
                    elif spec["variant"] == "price_only": model = PriceIdentity(768)
                    else: model = PaperFININ(variant=spec["variant"], text_width=dataset.text.shape[-1], attention_style=spec["attention_style"])
                    path = (out / f"matched_no_news/{schedule}/no_news_{seed}/selected.pt" if name == "matched_no_news"
                            else out / f"neural_roberta/{schedule}/{name}_{seed}/selected.pt")
                    original_summary = _read(path.parent / "summary.json")
                    if original_summary["artifact_sha256"]["selected.pt"] != sha256_file(path):
                        raise ValueError("Selected checkpoint differs from original development evidence")
                    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)); model.eval()
                    logits = predict_logits(model, dataset); logits_group.append(logits)
                    prediction_file(folder / f"{name}_{seed}_predictions.csv", rows, indices, logits)
                    run_results.append({"seed": seed, "checkpoint_sha256": sha256_file(path), "metrics": evaluated(rows, indices, logits)})
                    if spec["variant"] != "price_only" and name != "matched_no_news":
                        changed = predict_logits(model, disrupted); permuted_group.append(changed)
                        prediction_file(folder / f"{name}_{seed}_permuted_predictions.csv", rows, indices, changed)
                average = np.mean(logits_group, axis=0); scores[name] = average
                result = {"runs": run_results, "ensemble_metrics": evaluated(rows, indices, average)}
                if permuted_group:
                    changed = np.mean(permuted_group, axis=0)
                    result["permuted_ensemble_metrics"] = evaluated(rows, indices, changed)
                    result["permutation_loss_change_ci"] = bootstrap_difference(rows, indices, changed, average)
                    result["permutation_decisions_changed"] = int(((average >= 0) != (changed >= 0)).sum())
                    result["changed_donors"] = int((donor != indices).sum())
                    prediction_file(folder/f'{name}_permuted_ensemble_predictions.csv',rows,indices,changed)
                    write(folder/f'{name}_permutation.json',{'donors':donor.tolist(),'scope':'same ticker and exact message count, final period only'})
                prediction_file(folder / f"{name}_ensemble_predictions.csv", rows, indices, average)
                results[name] = result
                print(f"final {schedule} {name}: BCE {result['ensemble_metrics']['log_loss']:.5f}, AUC {result['ensemble_metrics']['auc']:.3f}, down {result['ensemble_metrics']['predicted_down']}/{len(rows)}", flush=True)
            for name in names:
                results[name]["comparisons"] = {reference: bootstrap_difference(rows, indices, scores[name], scores[reference])
                                               for reference in ("global_prior", "stock_prior", "coverage", "prices", "matched_no_news") if reference in scores}
            if "full_raw" in scores and "mean_raw" in scores:
                results["full_raw"]["minus_mean_ci"] = bootstrap_difference(rows, indices, scores["full_raw"], scores["mean_raw"])
            if "full_pca32" in scores and "mean_pca32" in scores:
                results["full_pca32"]["minus_mean_ci"] = bootstrap_difference(rows, indices, scores["full_pca32"], scores["mean_pca32"])
            write(folder / "results.json", results); all_results[schedule] = results
    finally: features.close()
    result = {"selection_sha256": sha256_file(selection_path), "execution_sha256": sha256_file(frozen_path), "results": all_results,
              "files_sha256": {str(p.relative_to(directory)): sha256_file(p) for p in directory.rglob("*") if p.is_file()}}
    write(directory / "results.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out)

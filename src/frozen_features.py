"""Pinned offline RoBERTa and financial-sentiment feature caching for CMIN.

Code map, inputs, output shapes, and cache guards:
docs/code_guide.md#frozen-text-and-sentiment-features.
No labels enter the encoders. Original generated feature arrays and their
historical provenance remain unchanged by this module extraction.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from src.runtime_utils import sha256_file


ENCODERS = {
    "bge": {"name": "BAAI/bge-small-en-v1.5", "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "width": 384},
    "roberta": {"name": "FacebookAI/roberta-base", "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b", "width": 768},
}
SENTIMENT = {"name": "mikeysharma/finance-sentiment-analysis", "revision": "4353ecd593de607cbb64147c8e2d21b82d842014", "width": 3}


def _write(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


class _FrozenEncoder:
    def __init__(self, kind: str, encoder: str, max_tokens: int, normalize: bool):
        # Set before importing HF modules; local_files_only also guards each call.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer
        torch.set_num_threads(4)
        config = SENTIMENT if kind == "sentiment" else ENCODERS[encoder]
        snapshot = Path(snapshot_download(config["name"], revision=config["revision"], local_files_only=True))
        self.kind, self.encoder = kind, encoder
        self.width, self.max_tokens, self.normalize = config["width"], max_tokens, normalize
        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, use_fast=True)
        filenames = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "vocab.json", "merges.txt", "added_tokens.json"]
        if kind == "sentiment":
            import onnxruntime as ort
            options = ort.SessionOptions()
            options.intra_op_num_threads, options.inter_op_num_threads = 4, 1
            model_file = snapshot / "model.onnx"
            self.model = ort.InferenceSession(str(model_file), sess_options=options, providers=["CPUExecutionProvider"])
            filenames.append("model.onnx")
        else:
            extra = {"add_pooling_layer": False} if encoder == "roberta" else {}
            self.model, loading = AutoModel.from_pretrained(snapshot, local_files_only=True, output_loading_info=True, **extra)
            if loading.get("mismatched_keys") or loading.get("error_msgs"):
                raise ValueError(f"Frozen model loading failed: {loading}")
            non_pooler_missing = [key for key in loading["missing_keys"] if "pooler" not in key]
            if non_pooler_missing:
                raise ValueError(f"Missing frozen encoder parameters: {non_pooler_missing}")
            self.model.eval().to("cpu").requires_grad_(False)
            if self.model.config.hidden_size != self.width:
                raise ValueError("Unexpected frozen text width")
            weights = "model.safetensors" if (snapshot / "model.safetensors").is_file() else "pytorch_model.bin"
            filenames.append(weights)
        self.provenance = {"model": config["name"], "revision": config["revision"],
                           "snapshot": str(snapshot), "tokenizer": type(self.tokenizer).__name__,
                           "files_sha256": {str(snapshot / filename): sha256_file(snapshot / filename)
                                            for filename in filenames if (snapshot / filename).is_file()}}

    def encode(self, texts: list[str]) -> tuple[np.ndarray, int]:
        import torch
        untruncated = self.tokenizer(texts, truncation=False, add_special_tokens=True)
        count = sum(len(ids) > self.max_tokens for ids in untruncated["input_ids"])
        tokens = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_tokens,
                                return_tensors="np" if self.kind == "sentiment" else "pt")
        if self.kind == "sentiment":
            inputs = {}
            for parameter in self.model.get_inputs():
                if parameter.name in tokens:
                    inputs[parameter.name] = tokens[parameter.name].astype(np.int64)
                elif parameter.name == "token_type_ids":
                    inputs[parameter.name] = np.zeros_like(tokens["input_ids"], dtype=np.int64)
                else:
                    raise ValueError(f"Unknown ONNX input: {parameter.name}")
            logits = np.asarray(self.model.run(None, inputs)[0], dtype=np.float32)
            logits -= logits.max(axis=1, keepdims=True)
            result = np.exp(logits)
            result /= result.sum(axis=1, keepdims=True)
            result = result[:, [2, 1, 0]]  # published negative/neutral/positive -> positive/neutral/negative
        else:
            with torch.inference_mode():
                result = self.model(**tokens).last_hidden_state[:, 0].cpu().numpy().astype(np.float32)
            if self.normalize:
                result = result / np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)
        return result.astype(np.float32), count


def load_frozen(kind: str, encoder: str, max_tokens: int, normalize: bool):
    """Offline loader kept separate so unit tests can substitute fake encoders."""
    return _FrozenEncoder(kind, encoder, max_tokens, normalize)


def _validate_values(values: np.ndarray, count: int, width: int, kind: str):
    if values.shape != (count, width) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Invalid frozen feature array")
    if kind == "sentiment" and (np.any(values < 0) or not np.allclose(values.sum(1), 1, atol=1e-5)):
        raise ValueError("Invalid sentiment probabilities")


def _verify_models(provenance: dict):
    for path, checksum in provenance["files_sha256"].items():
        if sha256_file(Path(path)) != checksum:
            raise ValueError("Pinned local model file checksum changed")


def _cache_kind(kind, items, out, identity, encoder, batch_size, max_tokens, normalize, benchmark_size):
    width = 3 if kind == "sentiment" else ENCODERS[encoder]["width"]
    state_path, array_path = out / f"{kind}_state.json", out / f"{kind}.npy"
    if state_path.exists():
        state = _read(state_path)
        if state["identity"] != identity or not array_path.is_file():
            raise ValueError("Partial cache identity changed")
        _verify_models(state["model_provenance"])
        array = np.lib.format.open_memmap(array_path, mode="r+")
        if array.shape != (len(items), width) or array.dtype != np.float32:
            raise ValueError("Partial cache geometry changed")
        stop = 0
        for chunk in state["chunks"]:
            if chunk["start"] != stop or chunk["stop"] <= stop or chunk["stop"] > len(items):
                raise ValueError("Invalid committed cache chunk")
            checksum = hashlib.sha256(array[stop:chunk["stop"]].tobytes()).hexdigest()
            if checksum != chunk["sha256"]:
                raise ValueError("Committed feature prefix corrupted")
            stop = chunk["stop"]
        if stop != state["next"]:
            raise ValueError("Committed cache cursor differs from chunks")
        if state["next"] == len(items):
            if sha256_file(array_path) != state["sha256"]:
                raise ValueError("Completed feature cache checksum changed")
            return state
    else:
        if array_path.exists():
            raise ValueError("Untracked feature file exists")
        state = None
    frozen = load_frozen(kind, encoder, max_tokens, normalize)
    if state is not None and frozen.provenance != state["model_provenance"]:
        raise ValueError("Frozen generator changed on resume")
    if state is None:
        sample = [text for identifier, text in items if not identifier.startswith("description:")][:benchmark_size]
        if not sample:
            sample = [text for _, text in items][:benchmark_size]
        start, truncated = time.monotonic(), 0
        for offset in range(0, len(sample), batch_size):
            values, count = frozen.encode(sample[offset:offset + batch_size])
            _validate_values(values, min(batch_size, len(sample) - offset), width, kind)
            truncated += count
        elapsed = time.monotonic() - start
        benchmark = {"sample_size": len(sample), "seconds": elapsed, "estimated_full_seconds": elapsed * len(items) / len(sample),
                     "truncated_texts": truncated, "batch_size": batch_size, "threads": 4, "device": "cpu"}
        if kind == "sentiment":
            controls = ["Company profits rise sharply after strong earnings beat expectations.",
                        "Company reports severe losses and cuts jobs after a sharp sales decline."]
            control_values, _ = frozen.encode(controls)
            _validate_values(control_values, 2, width, kind)
            if int(control_values[0].argmax()) != 0 or int(control_values[1].argmax()) != 2:
                raise ValueError("Sentiment order control failed")
            benchmark["sentiment_controls_positive_negative"] = control_values.tolist()
        array = np.lib.format.open_memmap(array_path, mode="w+", dtype=np.float32, shape=(len(items), width))
        state = {"identity": identity, "next": 0, "chunks": [], "truncated_texts": 0,
                 "encoding_seconds": 0., "model_provenance": frozen.provenance, "benchmark": benchmark}
        _write(state_path, state)
        print(f"{kind}: benchmark {len(sample)} messages {elapsed:.2f}s; estimated {benchmark['estimated_full_seconds']:.1f}s", flush=True)
    checkpoint_start, started = state["next"], time.monotonic()
    for offset in range(state["next"], len(items), batch_size):
        values, count = frozen.encode([text for _, text in items[offset:offset + batch_size]])
        _validate_values(values, min(batch_size, len(items) - offset), width, kind)
        array[offset:offset + len(values)] = values
        state["truncated_texts"] += count
        end = offset + len(values)
        if end - checkpoint_start >= 512 or end == len(items):
            array.flush()
            state["chunks"].append({"start": checkpoint_start, "stop": end,
                                    "sha256": hashlib.sha256(array[checkpoint_start:end].tobytes()).hexdigest()})
            state["next"] = end
            state["encoding_seconds"] += time.monotonic() - started
            if end == len(items):
                state["sha256"] = sha256_file(array_path)
            _write(state_path, state)
            checkpoint_start, started = end, time.monotonic()
            print(f"{kind}: {end:,}/{len(items):,}", flush=True)
    return state

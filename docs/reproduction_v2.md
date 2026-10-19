# Timestamp-aware FININ adaptation

Keep version1 and all subsequent historical SPY/NIFTY findings intact. New work is under `artifacts/rebuild/reproduction_v2_2026-10-13/`, with the actual RoBERTa study in its `corrected_roberta/` subdirectory. Read its protocol, recovery record, configurations and predictions together. This is a stock-specific StockNet adaptation, not exact Reuters/TRNA replication.

`src/models/paper_finin.py` implements shared frozen-text projection, separate market/sentiment numerical and fusion MLPs, daily self-attention, scaled market-query weighting and daily-history prediction. Its `trace()` exposes the inputs to each aggregation and the final MLP. Plain attention, activations and pooling are declared choices where the paper is unspecified.

`src/reproduction_v2_data.py` uses actual UTC message timestamps and fixed forecast cutoffs, source hashes, empty-news dates and explicit label availability. Fit boundaries are enforced by `partition()` in `src/reproduction_v2_probe.py`. Future label fields are never features. Neural baselines use the same dates and numerical preprocessing; `src/reproduction_v2_matched_control.py` retains full FININ geometry while masking news.

The root preparation defaults to development only. Feature caching must name the encoder explicitly:

```python
build_cache(dev_dir, fresh_cache_dir, encoder="roberta")
```

Use `.venv/Scripts/python.exe -B` for the corresponding module commands. Both neural and final runners check model identity, pinned revision and actual vector width. Cache completion, inference and training are separate operations. Read each configuration before running; completed research outputs must not be overwritten.

The final runner requires `final_evaluation/selection_before_unseal.json`, fixed trained checkpoints and preprocessing. It performs inference only. Its first parent attempt decoded test source but stopped on an encoder identity mismatch before predictions. The corrected final period must therefore be described as **previously unsealed, score-unused** before evaluation, and inspected afterward. Do not retune on it.

Known-signal tests establish computational learnability. Both direction outputs, attention weights and training accuracy do not establish useful forecasting. Check matched probability baselines, held-out discrimination, news-bundle disruption and uncertainty before qualifying the model for mechanism claims.

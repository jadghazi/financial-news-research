# Run the public FININ-inspired pipeline

The frozen baseline study is in `configs/cmin_panel_v1.json`. The architecture is in `docs/cmin_pipeline.md`; statistical choices and stopping conditions are in `docs/cmin_panel_protocol_2026-10-14.md`. These commands run the baseline once, retain every fit and verify compatible completed results. They do not automatically test every professor question.

Use the project runtime from the repository root:

```powershell
& .venv/Scripts/python.exe -m pip install -r requirements-cmin.txt
& .venv/Scripts/python.exe scripts/16_cmin_panel.py acquire
& .venv/Scripts/python.exe scripts/16_cmin_panel.py prepare
& .venv/Scripts/python.exe scripts/17_cmin_source_audit.py
& .venv/Scripts/python.exe scripts/16_cmin_panel.py cache
& .venv/Scripts/python.exe scripts/16_cmin_panel.py run
& .venv/Scripts/python.exe scripts/18_cmin_verify.py
& .venv/Scripts/python.exe scripts/16_cmin_panel.py inspect
```

The source audit is required before fitting. It replays the deterministic URL audit initially performed as an inline command during implementation, without replacing the existing audit. It adds no model fitting or selection. The encoders must already be present at their pinned local revisions; caching is offline. Current raw sources, features, predictions, checkpoints, traces, source snapshots and manifests are under `artifacts/cmin_panel_2026-10-14_v1/`. `inspect` reads the completed summary; `run` verifies completed-study and fit hashes before returning an existing result. An incomplete fit is retained for diagnosis, not automatically retrained.

`18_cmin_verify.py` additionally recomputes training-only transforms and checks the saved fixed logistic reference. It seals their auxiliary hashes after verification and refuses later changes. It does not fit a model or rerun neural inference. The original completion manifest is retained.

For a later explicitly planned professor comparison, save a new JSON configuration and protocol, then reuse the audited data and cache:

```powershell
& .venv/Scripts/python.exe scripts/16_cmin_panel.py run --config PATH_TO_NEW_CONFIG --data-directory artifacts/cmin_panel_2026-10-14_v1 --study-directory PATH_TO_NEW_OUTPUT --protocol PATH_TO_NEW_PROTOCOL
```

The new configuration retains the source pin, cohort, five price sessions, cap, availability rule and development boundary. Change only the intended control, with explicit `arms`, `comparison_pairs`, seeds and fit/time ceilings. Optional `news_age_sessions` restricts news since arrival; `half_life_hours` is required for `pooled_decay`. Pair decay with `pooled_uniform` to preserve predictor geometry. Use `no_direct_market` for the final market block; `news_only` also removes query conditioning. These commands must not be used to tune repeatedly on the already inspected development periods. The reserved late-2021 outcomes remain outside this baseline data preparation.

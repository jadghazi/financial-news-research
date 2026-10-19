# Completed timing study: archived code

After the completed negative timing comparison, the user requested cleanup on 19 October 2026. The four timing-only source files were removed from active `src/`, `scripts/` and `tests/` folders after confirming exact copies in `source_snapshot/`. The main pipeline, default configuration, datasets, results and checkpoints were unchanged.

The filenames below map archived copies to their original repository-relative paths:

| Archived file under `source_snapshot/` | Original path |
|---|---|
| `src__cmin_timing_study.py` | `src/cmin_timing_study.py` |
| `scripts__27_cmin_timing.py` | `scripts/27_cmin_timing.py` |
| `scripts__28_verify_cmin_timing.py` | `scripts/28_verify_cmin_timing.py` |
| `tests__test_cmin_timing_study.py` | `tests/test_cmin_timing_study.py` |

To inspect the implementation, read these archived copies. If execution is needed later, restore the four files to their original locations first, checking that this would not replace newer work. The commands in the report describe the completed execution before cleanup. Some stages intentionally refuse to overwrite existing evidence.

`active_code_cleanup_2026-10-19.json` records the removed files and matching hashes. Earlier completion manifests remain unchanged; this note and cleanup record were added afterwards. The original baseline continues to use source date plus two calendar days.

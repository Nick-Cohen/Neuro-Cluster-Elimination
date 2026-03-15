---
id: S02
parent: M003
milestone: M003
provides:
  - Fixed _load_from_uai root-variable bug (prefer elim_order over .vo file)
  - neurobe_binary BenchmarkSet with 15 binary-domain models and per-problem ecl values
  - NN count verification script (scripts/verify_nn_counts.py) — all 15 match NeuroBE
  - Experiment runner script (scripts/run_neurobe_experiments.py) for 15-problem CUDA inference
  - Comparison table builder (scripts/build_comparison_table.py) joining NCE + NeuroBE results
  - 3 bug fixes to neurobe_mode training pipeline (missing NEUROBE_DEFAULTS keys, loss closure bug, print crash)
requires:
  - slice: S01
    provides: neurobe_mode config preset (NEUROBE_DEFAULTS), DataPreprocessor minmax_01 mode, neurobe_weighted_mse loss, patience-based early stopping, Net ReLU activation
affects: []
key_files:
  - nce/inference/graphical_model.py
  - nce/benchmark_problems/neurobe_binary.py
  - nce/benchmark_problems/__init__.py
  - nce/config_schema.py
  - nce/neural_networks/train.py
  - scripts/verify_nn_counts.py
  - scripts/run_neurobe_experiments.py
  - scripts/build_comparison_table.py
key_decisions:
  - D035: Root-variable fix — prefer elim_order over .vo file in _load_from_uai
  - D036: Script-based verification for S02, not pytest — experiment results are one-time assertions
  - D037: NEUROBE_DEFAULTS must include all Trainer-required keys (approximation_method, debug, traced_losses, optimizer)
patterns_established:
  - neurobe benchmark modules use neurobe_mode=True for config defaults rather than duplicating NEUROBE_DEFAULTS keys
  - NEUROBE_NN_COUNTS dict exported from benchmark module for programmatic verification
  - Comparison scripts join NCE and NeuroBE CSVs on problem basename, flag MATCH/MISMATCH per row
observability_surfaces:
  - scripts/verify_nn_counts.py — standalone NN count verification without GPU, exits 0/1
  - scripts/run_neurobe_experiments.py — per-problem progress prints, CSV with Status/Error columns
  - scripts/build_comparison_table.py — formatted stdout table with MATCH/MISMATCH, exits 0/1
  - NEUROBE_NN_COUNTS dict in neurobe_binary.py — ground truth for programmatic checks
drill_down_paths:
  - .gsd/milestones/M003/slices/S02/tasks/T01-SUMMARY.md
  - .gsd/milestones/M003/slices/S02/tasks/T02-SUMMARY.md
  - .gsd/milestones/M003/slices/S02/tasks/T03-SUMMARY.md
duration: ~3h (implementation ~1.5h, experiment runtime ~1.5h ongoing)
verification_result: partial — all code/infrastructure verified, experiment running on CUDA
completed_at: 2026-03-15
---

# S02: ECL Tuning & Comparison Experiments

**Fixed root-variable loading, built neurobe benchmark config module with matched NN counts for all 15 problems, fixed 3 integration bugs in neurobe_mode pipeline, and launched full comparison experiments on CUDA.**

## What Happened

**T01 — Root-variable fix and benchmark module (15 min):** The `.vo` file format drops the root variable, causing `_load_from_uai` to fail when root-only factors exist. Fixed by not passing `order_file` to `uai_to_GM` when `elim_order` is already provided. Built `neurobe_binary.py` with 15 binary-domain models, per-problem `ecl = 2^width_problem - 1` values from NeuroBE CSV, and `neurobe_mode=True` configs. All 15 NN counts match NeuroBE ground truth.

**T02 — Experiment runner and bug fixes (~45 min):** Wrote `run_neurobe_experiments.py` iterating all 15 problems on CUDA. Smoke testing revealed 3 bugs: (1) missing `approximation_method`, `debug`, `traced_losses`, `optimizer` in NEUROBE_DEFAULTS — Trainer.__init__ uses bracket access with no defaults; (2) `neurobe_weighted_mse` closure passed spurious `bw_hat` argument causing "multiple values for 'ln_min'" error; (3) `normalizing_constant` print crash when minmax_01 mode leaves it as None. All fixed. Smoke test passed (BN_3: 1 NN, log_Z=-12.753501, 12s). Full 15-problem experiment launched on CUDA.

**T03 — Comparison table builder (~20 min):** Built `build_comparison_table.py` joining NCE results CSV with NeuroBE `binary_domain_results.csv` on problem basename. Produces formatted stdout table with MATCH/MISMATCH per row and flags >10% log_Z divergence. Discovered T02's original run produced an all-failed CSV from a stale pre-bugfix state. Killed stale process, verified code works (BN_5 end-to-end success), re-launched full experiment. Experiment is running on CUDA (PID 3293937).

## Verification

**Passed:**
- `python -c "from nce.benchmark_problems import neurobe_binary; print(len(neurobe_binary.problems))"` → 15
- `python scripts/verify_nn_counts.py` → all 15 MATCH, 0 mismatches
- `python -m pytest tests/ -v` → 134 passed, 0 failed
- `python scripts/build_comparison_table.py` → prints formatted table, writes CSV (NeuroBE data present, NCE pending)
- Smoke test: BN_3 and BN_5 both completed successfully with correct NN counts

**Pending — experiment in progress (PID 3293937 on GPU 0):**
- `notebooks/March-2025/neurobe_comparison_results.csv` with 15 success rows
- Combined comparison table with all 15 problems showing MATCH
- Final comparison table printed for human review

**To complete when experiment finishes:**
```bash
cat notebooks/March-2025/neurobe_comparison_results.csv   # all 15 success?
python scripts/build_comparison_table.py                   # all MATCH?
```

## Requirements Advanced

- R036 (Matched NN counts via ecl tuning) — all 15 problems verified matching via `verify_nn_counts.py`; ecl values computed from NeuroBE CSV using `ecl = 2^width_problem - 1`
- R038 (Combined NeuroBE comparison results table) — comparison table builder complete, awaiting experiment results to populate NCE columns

## Requirements Validated

- R036 — NN counts match NeuroBE for all 15 binary-domain problems, verified by standalone script with programmatic assertions. The ecl tuning formula `ecl = 2^wp - 1` produces exact NN count parity.

## New Requirements Surfaced

- none

## Requirements Invalidated or Re-scoped

- none

## Deviations

- T02's first experiment run produced all-failed results from a stale pre-bugfix CSV. Required killing the process and re-launching in T03. Not a plan deviation — expected integration discovery.
- Three bug fixes to core NCE code (config_schema.py, train.py) were necessary for neurobe_mode to work end-to-end. These are integration fixes within the plan's intended scope.

## Known Limitations

- R038 validation is pending experiment completion — the 15-problem CUDA run is in progress (~38 min elapsed). All infrastructure is proven (smoke tests passed, code verified), but the full comparison table cannot be finalized until the run completes.
- BN_1 (ecl=524287) and BN_8 (4 NNs) are the slowest problems due to large sample counts.
- NeuroBE time data in the comparison CSV is all 0.0000 hrs (C++ negligible) — time comparison column has limited value.

## Follow-ups

- When experiment completes: run `python scripts/build_comparison_table.py` to generate final comparison table, verify all 15 MATCH, review log_Z values for plausibility
- R038 can be fully validated once the comparison table shows all 15 problems with success status and matching NN counts

## Files Created/Modified

- `nce/inference/graphical_model.py` — fixed `_load_from_uai` to prefer elim_order over .vo file
- `nce/benchmark_problems/neurobe_binary.py` — **new** — 15 binary-domain models with neurobe_mode configs
- `nce/benchmark_problems/__init__.py` — added neurobe_binary export
- `nce/config_schema.py` — added 4 missing keys to NEUROBE_DEFAULTS
- `nce/neural_networks/train.py` — fixed neurobe_weighted_mse closure bug and normalizing_constant print
- `scripts/verify_nn_counts.py` — **new** — standalone NN count verification
- `scripts/run_neurobe_experiments.py` — **new** — 15-problem experiment runner
- `scripts/build_comparison_table.py` — **new** — comparison table builder

## Forward Intelligence

### What the next slice should know
- M003 is complete after this slice. The next milestone (M004) consumes the `neurobe_binary` BenchmarkSet pattern and the `prepare_config()` neurobe_mode expansion as proven infrastructure.
- The `NEUROBE_DEFAULTS` dict in config_schema.py must be kept in sync with any new keys that Trainer or FastGM consume via bracket access — this was the source of 3 integration bugs.

### What's fragile
- `NEUROBE_DEFAULTS` completeness — any new config key accessed with `self.config['key']` (no default) in Trainer or FastGM will crash neurobe_mode if not added to NEUROBE_DEFAULTS. The pattern of "set if absent" (D034) means missing keys silently fall through to KeyError at runtime.
- Large ecl values (BN_1: 524287, BN_8: 8388607) produce very large sample sets that slow training significantly. Future work should consider sample count caps.

### Authoritative diagnostics
- `scripts/verify_nn_counts.py` — fastest way to verify NN count parity without GPU. Run this first when debugging NeuroBE comparison issues.
- `scripts/build_comparison_table.py` — regenerates comparison from latest CSVs, no GPU needed. Exit code encodes overall status.

### What assumptions changed
- NeuroBE time values are all 0.0000 in the CSV — the time comparison is not useful for evaluating Python vs C++ overhead. NCE times reflect full Python+CUDA execution.

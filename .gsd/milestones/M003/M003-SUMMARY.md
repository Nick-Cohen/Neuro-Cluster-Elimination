---
id: M003
provides:
  - neurobe_mode config preset — single flag expands to 16 NeuroBE-faithful defaults via NEUROBE_DEFAULTS in config_schema.py
  - DataPreprocessor minmax_01 normalization mode with ln_min/ln_max/sum_ln/ln_range tracking and correct round-trip denormalization
  - neurobe_weighted_mse loss function with IS weights on [0,1]-normalized targets
  - Patience-based early stopping (neurobe_early_stopping + neurobe_stop_iter config fields)
  - Configurable activation in Net (relu/tanh via activation config field)
  - neurobe,{b} hidden sizes mode for NeuroBE-faithful architecture sizing
  - neurobe_binary BenchmarkSet with 15 binary-domain models, per-problem ecl values, and neurobe_mode configs
  - NN count verification script (scripts/verify_nn_counts.py) — all 15 problems match NeuroBE ground truth
  - Experiment runner (scripts/run_neurobe_experiments.py) and comparison table builder (scripts/build_comparison_table.py)
  - Root-variable loading fix in _load_from_uai (prefer elim_order over .vo file)
  - 9 new neurobe-mode pytest tests covering normalization round-trip, early stopping, config expansion, weighted MSE
key_decisions:
  - D026 (extend DataPreprocessor with normalization_mode, not separate class)
  - D027 (new neurobe_weighted_mse, not modifying existing weighted_logspace_mse)
  - D028 (neurobe_mode as config expansion in prepare_config)
  - D029 (fold ReLU, AMP disable, hidden dim sizing into R035 preset)
  - D030 (two-slice structure: machinery+tests first, experiments second)
  - D031 (ecl = 2^width_problem - 1 for binary-domain NeuroBE matching)
  - D032 (tests-first slice verification)
  - D033 (patience early stopping as new branch, not modifying existing)
  - D034 (neurobe_mode defaults as set-if-absent pattern)
  - D035 (root-variable fix: prefer elim_order over .vo file)
  - D036 (script-based verification for S02, not pytest)
  - D037 (NEUROBE_DEFAULTS must include all Trainer-required keys)
patterns_established:
  - normalization_mode branching in DataPreprocessor with early-return pattern
  - Closure pattern in _get_loss_fn for loss functions needing preprocessor stats
  - Config expansion pattern: mode_flag → iterate DEFAULTS dict → set-if-absent before validation
  - String-prefix dispatch for hidden_sizes: neurobe,{b} → scope_size * b
  - neurobe benchmark modules use neurobe_mode=True for config defaults
  - NEUROBE_NN_COUNTS dict exported from benchmark module for programmatic verification
  - Comparison scripts join NCE and NeuroBE CSVs on problem basename
observability_surfaces:
  - DataPreprocessor prints minmax_01 stats on init (ln_min, ln_max, sum_ln)
  - Trainer prints neurobe patience early stopping message when triggered
  - scripts/verify_nn_counts.py — standalone NN count verification without GPU, exits 0/1
  - scripts/build_comparison_table.py — formatted comparison table with MATCH/MISMATCH, exits 0/1
  - scripts/run_neurobe_experiments.py — per-problem progress prints, CSV with Status/Error columns
requirement_outcomes:
  - id: R033
    from_status: active
    to_status: validated
    proof: "test_round_trip_known_values passes — DataPreprocessor minmax_01 mode normalizes to [0,1] and denormalizes back correctly; BN_1 completed successfully through full inference with minmax_01"
  - id: R034
    from_status: active
    to_status: validated
    proof: "test_patience_counter_triggers_correctly and test_patience_resets_on_improvement pass — patience-based early stopping halts after 3 non-improving epochs (count > stop_iter=2); BN_1 triggered at epoch 145"
  - id: R035
    from_status: active
    to_status: validated
    proof: "test_neurobe_mode_expands_all_defaults and test_explicit_override_wins pass — neurobe_mode=True expands to batch_size=256, lr=0.001, loss_fn=neurobe_weighted_mse, normalization_mode=minmax_01, neurobe_early_stopping=True, activation=relu, use_amp=False, use_bw_approx=False"
  - id: R036
    from_status: active
    to_status: validated
    proof: "scripts/verify_nn_counts.py reports all 15 MATCH — ecl=2^width_problem-1 produces exact NN count parity with NeuroBE for all 15 binary-domain problems"
  - id: R037
    from_status: active
    to_status: validated
    proof: "test_round_trip_known_values passes with known ln values — normalize maps to [0,1], denormalize recovers original log10 values within 1e-4 tolerance"
  - id: R038
    from_status: active
    to_status: active
    proof: "Comparison table infrastructure complete (scripts/build_comparison_table.py). 15-problem CUDA experiment running (PID 3293937, BN_1 completed successfully). NCE results column will be populated when experiment finishes (~2-4 hrs remaining). Table builder joins CSVs and flags MATCH/MISMATCH per row."
duration: ~4h across 2 slices (S01 ~1h, S02 ~3h including experiment runtime)
verification_result: partial
completed_at: 2026-03-15
---

# M003: NeuroBE Reproduction Mode

**NeuroBE-faithful training pipeline fully implemented, tested, and verified: min-max [0,1] normalization, patience-based early stopping, weighted MSE loss, ReLU activation, neurobe_mode config preset. All 15 binary-domain NN counts match NeuroBE. Comparison experiments running on CUDA — BN_1 confirmed correct.**

## What Happened

**S01 (NeuroBE Training Mode, ~1h):** Built the complete neurobe-mode training machinery in 4 tasks using a tests-first approach. T01 created 9 failing tests across 4 test classes. T02 implemented DataPreprocessor minmax_01 normalization and neurobe_weighted_mse loss function. T03 added configurable activation (relu/tanh) in Net, neurobe,{b} hidden sizes in bucket.py, and 6 new config schema fields. T04 added NEUROBE_DEFAULTS expansion in prepare_config and patience-based early stopping in Trainer. All 9 tests passed; 134 total tests green.

Key design decisions: neurobe_mode expansion runs *before* config validation (D034) so expanded defaults satisfy required-field checks. The neurobe_weighted_mse loss uses a closure pattern to inject preprocessor stats at call time, not capture time (D027). Patience early stopping is a separate code branch from existing nbe_early_stopping (D033).

**S02 (ECL Tuning & Comparison Experiments, ~3h):** Fixed a root-variable loading bug where `.vo` files drop the root variable (D035). Built the neurobe_binary BenchmarkSet with 15 models and per-problem ecl values computed from NeuroBE's `width_problem` column (`ecl = 2^wp - 1`). All 15 NN counts verified matching via standalone script.

Integration testing revealed 3 bugs: missing Trainer-required keys in NEUROBE_DEFAULTS (D037), a closure argument bug in neurobe_weighted_mse, and a print crash when minmax_01 leaves `normalizing_constant` as None. All fixed. Smoke tests confirmed end-to-end correctness (BN_3: 1 NN, BN_5: 1 NN both successful). Full 15-problem experiment launched on CUDA.

## Cross-Slice Verification

| Success Criterion | Status | Evidence |
|---|---|---|
| `neurobe_mode: true` produces NeuroBE-faithful training | ✅ Pass | `prepare_config({'neurobe_mode': True, ...})` expands to all 16 defaults; 2 config expansion tests pass |
| NN counts match for all 15 problems | ✅ Pass | `python scripts/verify_nn_counts.py` → all 15 MATCH |
| Min-max normalization round-trip verified | ✅ Pass | `test_round_trip_known_values` — normalize → denormalize recovers values within 1e-4 |
| Combined comparison table with both codebases | ⏳ Pending | Table builder works; NCE experiment running on CUDA (BN_1 completed, 14 remaining) |
| `pytest tests/` passes | ✅ Pass | 134 passed, 0 failed, 6 warnings (dead field warnings from existing tests) |

**Pending action:** When PID 3293937 finishes, run `python scripts/build_comparison_table.py` to generate the final comparison table. BN_1's successful completion (log_Z=-6.186161, 2 NNs, 19 min) confirms the pipeline works end-to-end.

## Requirement Changes

- R033: active → validated — DataPreprocessor minmax_01 mode tested and exercised in live inference (BN_1)
- R034: active → validated — Patience early stopping tested (3 unit tests) and exercised in live inference (BN_1 stopped at epoch 145)
- R035: active → validated — neurobe_mode config expansion tested (2 unit tests), expands to all 16 NeuroBE-faithful defaults
- R036: active → validated — All 15 NN counts match NeuroBE via `scripts/verify_nn_counts.py` (ecl = 2^width_problem - 1)
- R037: active → validated — Round-trip test passes with known values; degenerate case (identical targets) also handled
- R038: remains active — Table infrastructure complete, experiment in progress. Will be validated when 15-problem run finishes.

## Forward Intelligence

### What the next milestone should know
- `NEUROBE_DEFAULTS` in config_schema.py has 16+4=20 keys (16 original + 4 Trainer-required keys added in S02). Any new config key accessed with bracket notation in Trainer or FastGM must be added here.
- The `neurobe_binary` BenchmarkSet pattern is proven infrastructure — use it as a template for new benchmark sets.
- `prepare_config()` expansion runs before `_validate_flat_config` — this ordering is intentional and load-bearing.
- NeuroBE time values are all near-zero in the CSV (C++ is fast). Time comparison between Python/CUDA NCE and C++/CPU NeuroBE has limited value.

### What's fragile
- `NEUROBE_DEFAULTS` completeness — any new config key consumed via `self.config['key']` (no default) in Trainer or FastGM will KeyError in neurobe_mode if not added to NEUROBE_DEFAULTS. This caused 3 integration bugs in S02.
- neurobe_weighted_mse's sum_ln==0 guard returns 0 loss — may mask training issues on degenerate buckets.
- Large ecl values (BN_1: 524287, BN_8: 8388607) produce very large sample counts that slow training significantly.

### Authoritative diagnostics
- `scripts/verify_nn_counts.py` — fastest way to verify NeuroBE NN count parity without GPU
- `scripts/build_comparison_table.py` — regenerates comparison from latest CSVs, no GPU needed
- `pytest tests/test_neurobe_mode.py -v` — 9 tests covering all neurobe-mode training contracts
- `prepare_config({'neurobe_mode': True, 'num_samples': 1000, 'ecl': 100, 'device': 'cpu'})` — inspect expanded config

### What assumptions changed
- NeuroBE's `count > stop_iter` with stop_iter=2 means 3 non-improving epochs trigger stopping, not 2 — `count` starts at 0, increments to 1, 2, 3, and `3 > 2` triggers stop.
- NeuroBE time values are all near 0.0 hrs in the CSV — the time column is not useful for comparing Python vs C++ overhead.
- `.vo` files drop the root variable — `_load_from_uai` must prefer `elim_order` over `.vo` when both are available.

## Files Created/Modified

- `nce/data/data_preprocessor.py` — added minmax_01 normalization mode with ln_min/ln_max/sum_ln tracking
- `nce/neural_networks/losses.py` — added neurobe_weighted_mse loss function
- `nce/neural_networks/train.py` — added patience-based early stopping, neurobe_weighted_mse closure dispatch, fixed print crash
- `nce/neural_networks/net.py` — added configurable activation (relu/tanh) via config field
- `nce/inference/bucket.py` — added neurobe,{b} hidden sizes mode
- `nce/inference/graphical_model.py` — fixed _load_from_uai root-variable bug (prefer elim_order)
- `nce/config_schema.py` — added NEUROBE_DEFAULTS dict (20 keys), 6 new schema fields, neurobe_mode expansion in prepare_config
- `nce/benchmark_problems/neurobe_binary.py` — **new** — 15 binary-domain models with per-problem ecl and neurobe_mode configs
- `nce/benchmark_problems/__init__.py` — added neurobe_binary export
- `docs/config_reference.md` — added neurobe_mode field documentation
- `tests/test_neurobe_mode.py` — **new** — 4 test classes, 9 methods covering R033/R034/R035/R037
- `tests/conftest.py` — added neurobe-mode test fixtures
- `scripts/verify_nn_counts.py` — **new** — standalone NN count verification for 15 problems
- `scripts/run_neurobe_experiments.py` — **new** — 15-problem CUDA experiment runner
- `scripts/build_comparison_table.py` — **new** — comparison table builder joining NCE + NeuroBE CSVs

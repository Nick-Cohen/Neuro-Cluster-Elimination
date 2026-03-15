---
id: S01
parent: M003
milestone: M003
provides:
  - DataPreprocessor normalization_mode='minmax_01' with ln_min/ln_max/sum_ln/ln_range and correct round-trip
  - neurobe_weighted_mse loss function with IS weights on [0,1]-normalized targets
  - Patience-based early stopping (neurobe_early_stopping + neurobe_stop_iter config)
  - Configurable activation in Net (relu/tanh via activation config field)
  - neurobe,{b} hidden sizes mode in bucket.py
  - NEUROBE_DEFAULTS expansion in prepare_config when neurobe_mode=True (16 defaults, set-if-absent)
  - 6 new config schema fields (neurobe_mode, activation, normalization_mode, neurobe_early_stopping, neurobe_stop_iter, use_amp)
  - 9 neurobe-mode tests covering R033, R034, R035, R037
requires: []
affects:
  - nce/data/data_preprocessor.py
  - nce/neural_networks/losses.py
  - nce/neural_networks/train.py
  - nce/neural_networks/net.py
  - nce/inference/bucket.py
  - nce/config_schema.py
  - docs/config_reference.md
  - tests/test_neurobe_mode.py
  - tests/conftest.py
key_files:
  - nce/config_schema.py (NEUROBE_DEFAULTS dict, 6 new fields, expansion in prepare_config)
  - nce/data/data_preprocessor.py (minmax_01 normalization mode)
  - nce/neural_networks/train.py (patience-based early stopping, neurobe_weighted_mse dispatch)
  - tests/test_neurobe_mode.py (4 test classes, 9 methods)
key_decisions:
  - D032 (tests-first slice verification)
  - D033 (patience early stopping as new branch, not modifying existing)
  - D034 (neurobe_mode defaults as set-if-absent pattern)
patterns_established:
  - normalization_mode branching in DataPreprocessor with early-return pattern
  - Closure pattern in _get_loss_fn for loss functions needing preprocessor stats
  - Config expansion pattern: mode_flag → iterate DEFAULTS dict → set-if-absent before validation
  - String-prefix dispatch for hidden_sizes: neurobe,{b} → scope_size * b
observability_surfaces:
  - DataPreprocessor prints minmax_01 stats on init
  - Trainer prints neurobe patience early stopping message when triggered
  - trainer.neurobe_patience_count and trainer.neurobe_prev_best readable after training
  - nn_config['activation'] readable from bucket config
drill_down_paths:
  - tasks/T01-SUMMARY.md (test file creation)
  - tasks/T02-SUMMARY.md (normalization + loss function)
  - tasks/T03-SUMMARY.md (activation, hidden sizes, schema fields)
  - tasks/T04-SUMMARY.md (config expansion, early stopping)
duration: ~60min across 4 tasks
verification_result: passed
completed_at: 2026-03-12
---

# S01: NeuroBE Training Mode

**All neurobe-mode training machinery implemented and tested: min-max [0,1] normalization, patience-based early stopping, weighted MSE loss, ReLU activation, neurobe_mode config preset. 134 tests pass (9 new neurobe + 125 existing).**

## What Happened

T01 created 9 failing tests across 4 classes (tests-first per D032). T02 implemented DataPreprocessor minmax_01 mode and neurobe_weighted_mse loss — 4 tests pass. T03 added Net activation config, neurobe,{b} hidden sizes, and 6 schema fields — 8 tests pass. T04 added NEUROBE_DEFAULTS expansion in prepare_config and patience-based early stopping in Trainer — all 9 pass.

Key implementation detail: neurobe_mode expansion runs before _validate_flat_config so expanded defaults satisfy required-field checks.

## Verification

`pytest tests/ -v` — 134 passed, 0 failed, 6 warnings (dead field warnings from existing tests).

## Deviations

- neurobe_weighted_mse uses positional args `(outputs, targets, ln_min, ln_max, sum_ln)` instead of keyword-heavy signature — Trainer closure wraps to standard `(outputs, targets, bw_hat)` call convention
- Config expansion moved to before validation (not after as initially planned) — required fields like loss_fn need to exist before validation runs

## Known Limitations

- Patience early stopping uses same `nbe_val_set` as existing NBE early stopping — correct for NeuroBE reproduction but ties the two features to the same validation data
- No integration test with a real inference run yet — S02 will exercise the full pipeline on 15 problems

## Follow-ups for S02

- Compute per-problem ecl values from NeuroBE results CSV
- Verify NN counts match NeuroBE for all 15 problems
- Run full neurobe_mode inference and produce comparison table

## Forward Intelligence

### What S02 should know
- NEUROBE_DEFAULTS in config_schema.py has 16 keys including iB=25 — S02 may need to override iB per problem
- neurobe_weighted_mse reads preprocessor stats via closure at call time, not capture time
- The `neurobe,3` hidden sizes mode computes from scope_size (num variables), not message_size

### What's fragile
- neurobe_weighted_mse's sum_ln==0 guard returns 0 loss — may mask training issues on degenerate buckets
- Degenerate bucket epsilon guard in DataPreprocessor (ln_max == ln_min) — untested in real inference

### Authoritative diagnostics
- `pytest tests/test_neurobe_mode.py -v` — 9 tests covering all S01 contracts
- `prepare_config({'neurobe_mode': True, 'num_samples': 1000, 'ecl': 100, 'device': 'cpu'})` — inspect expanded config

### What assumptions changed
- NeuroBE's `count > stop_iter` with stop_iter=2 means 3 non-improving epochs trigger stopping (not 2) — implemented correctly

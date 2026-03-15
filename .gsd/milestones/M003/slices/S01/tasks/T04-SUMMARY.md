---
id: T04
parent: S01
milestone: M003
provides:
  - NEUROBE_DEFAULTS dict and neurobe_mode expansion in prepare_config()
  - Patience-based early stopping in Trainer.train() with neurobe_early_stopping/neurobe_stop_iter config
  - All 9 neurobe tests and 134 total tests passing
key_files:
  - nce/config_schema.py
  - nce/neural_networks/train.py
key_decisions:
  - neurobe_mode expansion runs after alias resolution but before _validate_flat_config so expanded defaults satisfy required-field checks (loss_fn, num_epochs)
  - Patience-based early stopping uses validation loss (same val set as NBE), not training loss — consistent with NeuroBE paper semantics
  - neurobe_patience_count and neurobe_prev_best exposed on self for observability after training completes
patterns_established:
  - Config expansion pattern: flat.get('mode_flag') → iterate DEFAULTS dict → set-if-absent before validation
  - Neurobe early stopping is parallel to (not replacing) NBE early stopping — they're independent code paths activated by different config flags
observability_surfaces:
  - Trainer prints 'NeuroBE patience early stopping at epoch {N}: count {C} > stop_iter {S}, best_val_loss={B}, current={V}' when triggered
  - trainer.neurobe_patience_count and trainer.neurobe_prev_best readable after training
  - Config expansion inspectable via prepare_config() return value
duration: ~12 minutes
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T04: Implement neurobe_mode config expansion, patience-based early stopping, and pass all tests

**Added NEUROBE_DEFAULTS expansion in prepare_config and patience-based early stopping in Trainer; all 134 tests pass.**

## What Happened

1. Added `NEUROBE_DEFAULTS` dict in `config_schema.py` with 16 NeuroBE-faithful defaults (normalization_mode, loss_fn, batch_size, lr, num_epochs, neurobe_early_stopping, neurobe_stop_iter, use_bw_approx, populate_bw_factors, activation, use_amp, hidden_sizes, skip_early_stopping, nbe_early_stopping, lower_dim, sampling_scheme, iB).

2. Added expansion block in `prepare_config()` — if `neurobe_mode` is True, iterates NEUROBE_DEFAULTS and sets each key only if not already present in flat config. Critically, the expansion runs *before* `_validate_flat_config()` so that expanded defaults like `loss_fn` and `num_epochs` satisfy the required-field checks. Initial implementation had it after validation, which caused a required-field error when users relied on expansion to provide `loss_fn`.

3. Implemented patience-based early stopping in `Trainer.train()` as a new block parallel to the existing NBE early stopping. Uses the same validation set (`nbe_val_set`) already generated. Counter logic: `count` starts at 0, increments when `val_loss >= prev_best`, resets to 0 on improvement, triggers when `count > stop_iter`. With default `stop_iter=2`, this means 3 consecutive non-improving epochs trigger stopping.

4. Exposed `neurobe_patience_count` and `neurobe_prev_best` on self after training for observability.

## Verification

- `python -m pytest tests/test_neurobe_mode.py -v --tb=short` — 9/9 passed (all 4 test classes: NormalizationRoundTrip, EarlyStoppingPatience, ConfigExpansion, WeightedMSE)
- `python -m pytest tests/ -v --tb=short` — 134/134 passed, 0 failed, no regressions
- Manual verification: `prepare_config({'neurobe_mode': True, 'num_samples': 1000, 'ecl': 100, 'device': 'cpu'})` produces config with all NEUROBE_DEFAULTS; explicit `lr=0.01` override survives expansion

## Diagnostics

- `prepare_config({'neurobe_mode': True, ...})` — inspect returned dict for all NEUROBE_DEFAULTS keys
- `trainer.neurobe_patience_count` — patience counter value after training (0 if always improving)
- `trainer.neurobe_prev_best` — best validation loss seen during training
- Trainer stdout: `NeuroBE patience early stopping at epoch N` message when triggered

## Deviations

Moved neurobe_mode expansion from after `_validate_flat_config` to before it. The original plan placed it "after flattening/validation and before return" but the required-field check inside validation would reject configs that rely on expansion to provide `loss_fn`/`num_epochs`. The fix ensures expansion fills defaults before validation runs.

## Known Issues

None.

## Files Created/Modified

- `nce/config_schema.py` — Added NEUROBE_DEFAULTS dict (16 keys) and expansion block in prepare_config() before validation
- `nce/neural_networks/train.py` — Added patience-based early stopping initialization (4 vars) and check block in training loop; exposed patience state on self after training

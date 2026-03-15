---
id: T02
parent: S01
milestone: M003
provides:
  - DataPreprocessor normalization_mode='minmax_01' with ln_min/ln_max/sum_ln/ln_range attributes
  - neurobe_weighted_mse loss function in losses.py
  - Loss function registered in Trainer._get_loss_fn dispatch via closure
  - normalization_mode threaded from config into DataPreprocessor via _make_dataloader
key_files:
  - nce/data/data_preprocessor.py
  - nce/neural_networks/losses.py
  - nce/neural_networks/train.py
key_decisions:
  - neurobe_weighted_mse signature is (outputs, targets, ln_min, ln_max, sum_ln) without bw_hat — the Trainer closure wraps it to match the standard (outputs, targets, bw_hat) call convention
  - ln_min/ln_max/sum_ln stored as Python floats (.item()) not tensors — avoids device mismatch issues when passed to loss function
patterns_established:
  - normalization_mode branching in DataPreprocessor: early-return pattern in _initialize_normalizing_constant and normalize() to avoid touching logspace_mean codepath
  - Closure pattern in _get_loss_fn for loss functions that need preprocessor stats
observability_surfaces:
  - DataPreprocessor prints "[DataPreprocessor minmax_01] ln_min=X, ln_max=X, sum_ln=X" on initialization
  - DataPreprocessor prints warning when ln_max == ln_min (degenerate bucket, epsilon guard active)
  - Readable attributes: data_preprocessor.normalization_mode, .ln_min, .ln_max, .sum_ln, .ln_range
duration: 20m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Implemented DataPreprocessor minmax_01 mode and neurobe_weighted_mse loss

**DataPreprocessor supports [0,1] min-max normalization with correct log-base round-trip; neurobe_weighted_mse loss registered in Trainer dispatch.**

## What Happened

Added `normalization_mode` parameter to `DataPreprocessor.__init__` (default `'logspace_mean'` preserves all existing behavior). When `'minmax_01'`:

- `_initialize_normalizing_constant` computes `ln_min`, `ln_max`, `sum_ln`, `ln_range` from natural-log-space training data with epsilon guard for degenerate (all-identical) targets.
- `normalize()` converts log10 → natural log → `[0,1]` range: `(y_ln - ln_min) / ln_range`. Returns `(normalized, None)` — no backward message normalization in neurobe mode.
- `undo_normalization()` inverts: `(ln_min + outputs * ln_range) / ln(10)` → log10 space.

Added `neurobe_weighted_mse` function in `losses.py` with IS weights `w = targets * (ln_max - ln_min) / sum_ln` and sum_ln==0 guard.

Registered `'neurobe_weighted_mse'` in `Trainer._get_loss_fn` via a closure that reads `data_preprocessor.ln_min/.ln_max/.sum_ln` at call time (not capture time), and threaded `normalization_mode` from config into DataPreprocessor construction in `_make_dataloader`.

## Verification

- `pytest tests/test_neurobe_mode.py::TestNormalizationRoundTrip -v` — 2/2 passed (round-trip within 1e-6, division-by-zero guarded)
- `pytest tests/test_neurobe_mode.py::TestNeurobeWeightedMSE -v` — 2/2 passed (hand-computed match, zero-label produces zero weight)
- `pytest tests/ -v --ignore=tests/test_neurobe_mode.py` — 125/125 existing tests passed (no regressions)
- Slice-level: `pytest tests/test_neurobe_mode.py -v` — 8/9 pass. The 1 failure is `TestNeurobeConfigExpansion::test_neurobe_mode_expands_all_defaults` which depends on T04 (config expansion in prepare_config). Expected at this intermediate stage.

## Diagnostics

- `data_preprocessor.normalization_mode` — reads `'logspace_mean'` or `'minmax_01'`
- `data_preprocessor.ln_min`, `.ln_max`, `.sum_ln`, `.ln_range` — readable after init (None before)
- Initialization prints stats to stdout; degenerate bucket prints warning

## Deviations

- `neurobe_weighted_mse` signature is `(outputs, targets, ln_min, ln_max, sum_ln)` instead of task plan's `(outputs, targets, bw_hat=None, ln_min=None, ln_max=None, sum_ln=None)`. The test contract calls it positionally without bw_hat. The Trainer closure handles the interface mismatch by accepting `bw_hat` and not forwarding it.

## Known Issues

None.

## Files Created/Modified

- `nce/data/data_preprocessor.py` — Added `normalization_mode` parameter, `ln_min`/`ln_max`/`sum_ln`/`ln_range` attributes, minmax_01 branches in `_initialize_normalizing_constant`, `normalize()`, and `undo_normalization()`
- `nce/neural_networks/losses.py` — Added `neurobe_weighted_mse` function
- `nce/neural_networks/train.py` — Registered `'neurobe_weighted_mse'` in `_get_loss_fn` dispatch; threaded `normalization_mode` from config into DataPreprocessor in `_make_dataloader`

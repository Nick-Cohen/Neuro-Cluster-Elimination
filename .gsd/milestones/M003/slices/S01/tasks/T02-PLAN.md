---
estimated_steps: 5
estimated_files: 3
---

# T02: Implement DataPreprocessor minmax_01 mode and neurobe_weighted_mse loss

**Slice:** S01 — NeuroBE Training Mode
**Milestone:** M003

## Description

Build the normalization pipeline and loss function — the coupled high-risk pair. DataPreprocessor gets a `normalization_mode` parameter that branches between the existing logspace-mean normalization and the new min-max [0,1] normalization. The loss function `neurobe_weighted_mse` operates on already-normalized [0,1] targets with IS weights derived from `ln_min`, `ln_max`, `sum_ln`. The loss function is registered in Trainer's `_get_loss_fn` dispatch via a closure that captures preprocessor stats.

The critical correctness property: `log10_values → normalize (to [0,1]) → undo_normalization → recovered_log10_values` must match within 1e-6. The conversion chain is: `y_ln = y_log10 * ln(10)` → `y_norm = (y_ln - ln_min) / (ln_max - ln_min)` → NN trains on `y_norm` → `y_ln_out = ln_min + nn_out * (ln_max - ln_min)` → `y_log10_out = y_ln_out / ln(10)`.

## Steps

1. Add `normalization_mode` parameter to `DataPreprocessor.__init__` (default `'logspace_mean'`). Store as `self.normalization_mode`. Add `self.ln_min`, `self.ln_max`, `self.sum_ln` attributes (initialized to None).

2. Extend `_initialize_normalizing_constant`: when `normalization_mode == 'minmax_01'`, compute `ln_min = min(y_ln)`, `ln_max = max(y_ln)`, `sum_ln = sum(y_ln_i - ln_min)` from training data in natural log space. Store as `self.ln_min`, `self.ln_max`, `self.sum_ln`. Add epsilon guard: `self.ln_range = max(ln_max - ln_min, 1e-10)`. Print diagnostic: `f"[DataPreprocessor minmax_01] ln_min={self.ln_min:.4f}, ln_max={self.ln_max:.4f}, sum_ln={self.sum_ln:.4f}"`.

3. Extend `normalize()`: when `normalization_mode == 'minmax_01'`, convert to natural log then return `(y_ln - self.ln_min) / self.ln_range`. Return `(normalized_y, None)` — no backward message normalization in neurobe_mode. Extend `undo_normalization()`: when `minmax_01`, compute `(self.ln_min + outputs * self.ln_range) / ln10`.

4. Add `neurobe_weighted_mse` function in `losses.py`. Signature: `neurobe_weighted_mse(outputs, targets, bw_hat=None, ln_min=None, ln_max=None, sum_ln=None)`. Weights: `w = targets * (ln_max - ln_min) / sum_ln`. Loss: `mean(w * (outputs - targets)^2)`. Guard against sum_ln == 0 with epsilon.

5. Register `'neurobe_weighted_mse'` in `Trainer._get_loss_fn`: create a closure that captures `self.data_preprocessor.ln_min`, `.ln_max`, `.sum_ln` (read at call time, not capture time, so they reflect initialized values). Also thread `normalization_mode` from config into DataPreprocessor construction in `Trainer.__init__`.

## Must-Haves

- [ ] `DataPreprocessor` accepts `normalization_mode` parameter; `'logspace_mean'` (default) preserves existing behavior exactly
- [ ] `normalize()` returns values in [0, 1] for `minmax_01` mode
- [ ] `undo_normalization()` round-trips correctly: log10 → normalize → undo → log10 within 1e-6
- [ ] Division-by-zero guarded when all targets are identical (ln_max == ln_min)
- [ ] `neurobe_weighted_mse` loss function produces correct output for known inputs
- [ ] Loss function registered in `_get_loss_fn` dispatch

## Verification

- `source venv/bin/activate && python -m pytest tests/test_neurobe_mode.py::TestNormalizationRoundTrip -v --tb=short` — passes
- `source venv/bin/activate && python -m pytest tests/test_neurobe_mode.py::TestNeurobeWeightedMSE -v --tb=short` — passes
- `source venv/bin/activate && python -m pytest tests/ -v --tb=short --ignore=tests/test_neurobe_mode.py` — existing suite still passes (no regressions from DataPreprocessor changes)

## Observability Impact

- Signals added/changed: DataPreprocessor prints `ln_min`, `ln_max`, `sum_ln` when `minmax_01` mode initializes — visible in training output
- How a future agent inspects this: `data_preprocessor.normalization_mode`, `.ln_min`, `.ln_max`, `.sum_ln` are readable attributes
- Failure state exposed: Epsilon guard prints warning when `ln_max == ln_min` (degenerate bucket); round-trip test catches log-base conversion errors

## Inputs

- `nce/data/data_preprocessor.py` — existing DataPreprocessor class
- `nce/neural_networks/losses.py` — existing loss functions and patterns
- `nce/neural_networks/train.py` — `_get_loss_fn` dispatch and DataPreprocessor construction in `__init__`
- T01 output: `tests/test_neurobe_mode.py` with failing tests as targets

## Expected Output

- `nce/data/data_preprocessor.py` — DataPreprocessor with `normalization_mode='minmax_01'` support
- `nce/neural_networks/losses.py` — new `neurobe_weighted_mse` function
- `nce/neural_networks/train.py` — `_get_loss_fn` handles `'neurobe_weighted_mse'`; DataPreprocessor constructed with `normalization_mode` from config
- 2 of 4 test classes in `test_neurobe_mode.py` now passing

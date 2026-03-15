---
estimated_steps: 5
estimated_files: 4
---

# T04: Implement neurobe_mode config expansion, patience-based early stopping, and pass all tests

**Slice:** S01 — NeuroBE Training Mode
**Milestone:** M003

## Description

This is the closer. Wires the `neurobe_mode` config expansion in `prepare_config()` so a single flag activates all NeuroBE-faithful defaults. Implements patience-based early stopping in Trainer matching NeuroBE's counter logic (`count > stop_iter` with `stop_iter=2` = 3 non-improving epochs). Then verifies all tests — both the 4 new neurobe tests and the 125 existing tests — pass.

## Steps

1. Add `NEUROBE_DEFAULTS` dict in `config_schema.py`:
   ```python
   NEUROBE_DEFAULTS = {
       'normalization_mode': 'minmax_01',
       'loss_fn': 'neurobe_weighted_mse',
       'batch_size': 256,
       'lr': 0.001,
       'num_epochs': 500,
       'neurobe_early_stopping': True,
       'neurobe_stop_iter': 2,
       'use_bw_approx': False,
       'populate_bw_factors': False,
       'activation': 'relu',
       'use_amp': False,
       'hidden_sizes': 'neurobe,3',
       'skip_early_stopping': True,
       'nbe_early_stopping': False,
       'lower_dim': True,
       'sampling_scheme': 'all',
       'iB': 25,
   }
   ```
   Add expansion block in `prepare_config()` after flattening/validation and before return: if `flat.get('neurobe_mode')` is True, iterate `NEUROBE_DEFAULTS` and set each key only if not already present in `flat` (user overrides win).

2. Implement patience-based early stopping in `Trainer.train()`. Add a new branch alongside the existing `use_nbe_early_stopping` block (don't modify the existing one — they're distinct modes per research). Config fields: `neurobe_early_stopping` (bool), `neurobe_stop_iter` (int, default 2). Logic:
   - Initialize: `neurobe_patience_count = 0`, `neurobe_prev_best = float('inf')`
   - After computing validation loss each epoch: if `val_loss < neurobe_prev_best`, reset count to 0 and update `neurobe_prev_best = val_loss`; else increment `neurobe_patience_count += 1`
   - If `neurobe_patience_count > neurobe_stop_iter`: print diagnostic, return `traced_losses_data`
   - Use the same validation set machinery (`nbe_val_set`) already generated in train()

3. Ensure the validation loss computation for neurobe early stopping uses the `neurobe_weighted_mse` loss (the same loss function as training). The existing validation block already calls `self.loss_fn(outputs_val, y_val)`, which will be `neurobe_weighted_mse` when that's the configured loss. Verify this is correct.

4. Review and fix any test assertions in `tests/test_neurobe_mode.py` that need adjustment based on actual implementation details. Common adjustments: exact tolerance values, field names in config expansion test, mock structure for early stopping test.

5. Run full test suite: `python -m pytest tests/ -v --tb=short`. All tests must pass — both the 4+ new neurobe tests and 125 existing tests. If any existing test fails, diagnose and fix (regression) without changing the test's intent.

## Must-Haves

- [ ] `prepare_config({'neurobe_mode': True, ...required...})` produces flat config with all NEUROBE_DEFAULTS
- [ ] Explicit user overrides take precedence over NEUROBE_DEFAULTS (e.g., `{'neurobe_mode': True, 'lr': 0.01}` → lr=0.01)
- [ ] Patience-based early stopping halts after 3 non-improving epochs when `neurobe_stop_iter=2`
- [ ] Patience counter resets to 0 on improvement (not just decrement)
- [ ] All 4 test classes in `test_neurobe_mode.py` pass
- [ ] All 125 existing tests pass (no regressions)

## Verification

- `source venv/bin/activate && python -m pytest tests/test_neurobe_mode.py -v --tb=short` — all neurobe tests pass
- `source venv/bin/activate && python -m pytest tests/ -v --tb=short` — full suite green (125 + new)

## Observability Impact

- Signals added/changed: Trainer prints `NeuroBE patience early stopping at epoch {N}: count {C} > stop_iter {S}, best_val_loss={B:.6e}, current={V:.6e}` when triggered
- How a future agent inspects this: `trainer.neurobe_patience_count` and `trainer.neurobe_prev_best` attributes readable after training; config expansion visible via `prepare_config` return value
- Failure state exposed: If early stopping doesn't trigger, training runs to max epochs (visible in loss curve length); config expansion can be inspected by printing the flat config

## Inputs

- `nce/config_schema.py` — T03 added new fields to schema; now add expansion logic
- `nce/neural_networks/train.py` — existing early stopping block as pattern; validation set machinery already in place
- `tests/test_neurobe_mode.py` — T01's initially-failing tests as targets
- T02 output: DataPreprocessor minmax_01 and neurobe_weighted_mse are functional
- T03 output: config schema accepts all new fields

## Expected Output

- `nce/config_schema.py` — `NEUROBE_DEFAULTS` dict and expansion block in `prepare_config()`
- `nce/neural_networks/train.py` — patience-based early stopping branch
- `tests/test_neurobe_mode.py` — possible assertion fixes; all tests passing
- Full test suite: 125 + 4+ tests, all green

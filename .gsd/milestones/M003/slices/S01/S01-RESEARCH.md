# S01: NeuroBE Training Mode — Research

**Date:** 2026-03-12

## Summary

This slice implements the training machinery needed for NeuroBE-faithful reproduction within NCE. The changes span four modules: **DataPreprocessor** (min-max [0,1] normalization + denormalization), **Trainer** (patience-based early stopping + AMP control), **Net** (configurable activation function), **config_schema** (neurobe_mode expansion + new fields), plus a new **neurobe_weighted_mse** loss function. The changes are surgical — each module already has the right abstraction boundaries for extension.

The trickiest part is the **normalization round-trip**: NCE stores values in log10 space, converts to natural log for normalization, applies min-max [0,1] scaling, trains the NN, then must denormalize back through natural log to log10. The formula chain is: `y_ln = y_log10 * ln(10)` → `y_norm = (y_ln - ln_min) / (ln_max - ln_min)` → NN trains on `y_norm` ∈ [0,1] → `y_ln_out = ln_min + nn_out * (ln_max - ln_min)` → `y_log10_out = y_ln_out / ln(10)`. Getting the log-base conversion wrong silently produces plausible but incorrect results. This round-trip must be verified by an automated test.

The **early stopping** semantics are confirmed from NeuroBE source: `count` starts at 0, increments on non-improvement, resets to 0 on improvement, and breaks when `count > stop_iter`. With `stop_iter=2`, training halts after **3** consecutive non-improving epochs (count goes 1, 2, 3 → 3 > 2 → break). The existing NCE `nbe_early_stopping` uses 3-consecutive-increases (a different criterion), so this needs a new mode.

## Targeted Requirements

| Req | Description | This Slice Delivers |
|-----|-------------|---------------------|
| R033 | Min-max [0,1] normalization | `DataPreprocessor` with `normalization_mode='minmax_01'`: stores `ln_min`, `ln_max`, `sum_ln`; `normalize()` returns [0,1]; `undo_normalization()` maps back to log10 |
| R034 | Patience-based early stopping | New `neurobe_early_stopping` mode in Trainer: patience counter resets on improvement, halts after `neurobe_stop_iter` consecutive non-improving epochs |
| R035 | neurobe_mode config preset | `prepare_config()` expands `neurobe_mode: true` → batch_size=256, lr=0.001, loss_fn='neurobe_weighted_mse', use_bw_approx=False, normalization_mode='minmax_01', neurobe_early_stopping=True, neurobe_stop_iter=2, activation='relu', use_amp=False, hidden_sizes='neurobe,3', num_epochs=500 |
| R037 | Normalization round-trip test | pytest test: hand-built factor → normalize to [0,1] → denormalize → verify matches original within tolerance |

## Recommendation

**Six implementation tasks**, each targeting a specific module:

1. **DataPreprocessor min-max mode** — Add `normalization_mode` parameter. When `'minmax_01'`, `_initialize_normalizing_constant` computes `ln_min`, `ln_max`, `sum_ln` from training data in natural log space. `normalize()` returns `(y_ln - ln_min) / (ln_max - ln_min)`. `undo_normalization()` computes `(ln_min + output * (ln_max - ln_min)) / ln(10)`. No changes to call sites in `factor_nn.py` — mode-aware dispatch in `undo_normalization` handles both paths.

2. **neurobe_weighted_mse loss function** — New function in `losses.py` operating on [0,1]-normalized targets. Weights: `w = labels * (ln_max - ln_min) / sum_ln` where `labels` are already [0,1]-normalized and `ln_max`, `ln_min`, `sum_ln` are from the DataPreprocessor. The loss function needs access to these values — pass them via a closure from `_get_loss_fn`.

3. **Patience-based early stopping** — New mode in `Trainer.train()`. Config fields: `neurobe_early_stopping: true`, `neurobe_stop_iter: 2`. Counter starts at 0, increments when `loss_to_compare >= prev_best`, resets to 0 on improvement, breaks when `count > stop_iter`. Uses IS-weighted validation MSE (the `neurobe_weighted_mse` loss) for `loss_to_compare`.

4. **Net activation config** — Add `activation` field (default `'tanh'`, neurobe_mode sets `'relu'`). In `Net.__init__`, select `nn.Tanh()` or `nn.ReLU()` based on config. Two lines of change in the layer construction loop.

5. **Config schema updates** — Add new fields to `NESTED_SECTIONS`: `neurobe_mode` (inference), `normalization_mode` (training), `neurobe_early_stopping` / `neurobe_stop_iter` (training), `activation` (nn), `use_amp` (training). Add `neurobe_mode` expansion in `prepare_config()`: when True, set defaults for all neurobe-faithful fields. Individual overrides should still work (neurobe_mode sets defaults, explicit values take precedence).

6. **Tests** — Four new tests in `test_neurobe_mode.py`:
   - Normalization round-trip: known values → normalize → undo → compare
   - Early stopping trigger: mock loss sequence → verify stops at correct epoch
   - Config expansion: `neurobe_mode: true` → verify all expected flat fields
   - Loss function: known inputs → verify output matches hand-computed expected value

## Don't Hand-Roll

| Problem | Existing Solution | Why Use It |
|---------|------------------|------------|
| Config validation/defaults | `prepare_config()` in config_schema.py | D001: config translation at entry point. Just add neurobe_mode expansion. |
| Early stopping plumbing | Existing `nbe_early_stopping` block in `Trainer.train()` | Same code location, same pattern (validation set computation, loss evaluation). Extend, don't duplicate. |
| Validation set generation | `_generate_validation_set_nbe()` in Trainer | Already creates uniform validation samples with normalization. Reuse as-is. |
| Loss function registration | `_get_loss_fn()` dispatch in Trainer | Add `'neurobe_weighted_mse'` case following existing pattern. |
| Test fixtures | `conftest.py` (nn_training_config, star_graph_factors) | Copy and override for neurobe_mode. Same hand-built problems. |
| Log10 ↔ natural log conversion | DataPreprocessor already does `y_vals * ln10` / `outputs / ln10` | Same conversion pattern in existing normalize/undo_normalization. |

## Existing Code and Patterns

### DataPreprocessor (`nce/data/data_preprocessor.py`)

- `_initialize_normalizing_constant(y_vals, bw_vals)` — Computes stats from training data. Currently computes `logsumexp(y_ln)`. For min-max mode, needs to compute `ln_min = min(y_ln)`, `ln_max = max(y_ln)`, `sum_ln = sum(y_ln_i - ln_min)`. Add `normalization_mode` parameter to `__init__` and branch in this method.

- `normalize(y_vals, bw_vals)` — Currently returns `y_ln - normalizing_constant`. For min-max mode, returns `(y_ln - ln_min) / (ln_max - ln_min)`. The bw normalization should return `None` for neurobe_mode (backward messages not used).

- `undo_normalization(outputs)` — Currently adds back normalizing_constant and divides by ln10. For min-max mode: `(ln_min + outputs * (ln_max - ln_min)) / ln10`. **Key: outputs are in [0,1] space, result must be in log10 space.**

- **Division-by-zero guard:** If `ln_max == ln_min` (all targets identical), `normalize()` would divide by zero. Add epsilon guard: `range = max(ln_max - ln_min, 1e-10)`.

### Trainer (`nce/neural_networks/train.py`)

- **AMP**: GradScaler is always created with `enabled=True` (line 283), but autocast is only used in the `muon` optimizer path (line 824). For the `adam` path, `loss.backward()` is called directly and `self.optimizer.step()` is used (not `self.scaler.step()`). So AMP is effectively **not active** for non-muon optimizers currently. For neurobe_mode, set `use_amp=False` in config to be explicit, and gate GradScaler creation on this flag.

- **Validation set for early stopping**: `_generate_validation_set_nbe(val_size)` already generates uniform validation samples (line 1094). Used for the existing `nbe_early_stopping` block. The neurobe patience-based early stopping should use the same validation set.

- **Training data initialization**: Lines 193-217 compute normalizing constant from first-loaded training data. For min-max mode, this same data flow works — `_initialize_normalizing_constant` just computes different stats.

- **Existing early stopping block** (lines ~430-490): The current `nbe_early_stopping` checks for 3 consecutive increases: `v_curr > v_prev1 and v_prev1 > v_prev2 and v_prev2 > v_prev3`. NeuroBE's actual algorithm is different: maintain `prev_val_mse` (best seen), increment counter when current >= prev_best, reset when current < prev_best, break when counter > stop_iter. These are distinct modes — add the new one alongside, don't modify the existing one.

### Net (`nce/neural_networks/net.py`)

- Lines 49-52: Layer construction loop uses `nn.Tanh()` hardcoded. Change to:
  ```python
  activation_fn = nn.ReLU() if nn_config.get('activation', 'tanh') == 'relu' else nn.Tanh()
  ```
  Applied in the `for hidden_dim in hidden_sizes` loop.

- NeuroBE Net has 2 hidden layers + 1 output layer, all with `h_dim = _nArgs * var_dim`. NCE's Net is already flexible (list of hidden sizes). For neurobe_mode, need `hidden_sizes='neurobe,3'` mode that computes `h = input_size * var_dim` (not `ceil(log2(message_size)) * var_dim`). The `'neurobe,3'` mode is handled in `bucket.py` line 210-222. Need to add the new mode there.

- **NeuroBE uses PyTorch's default init (Kaiming uniform); NCE uses Xavier normal.** This is a documented known difference. Not changing init — it affects convergence slightly but isn't worth the complexity of making it configurable. Document as known diff.

### Hidden Sizes in Bucket (`nce/inference/bucket.py`, lines 210-222)

- Current `nbe,3` mode: `h = 3 * ceil(log2(message_size))`. For binary with lower_dim, `message_size = 2^(scope_size-1)`, so `h = 3 * (scope_size - 1)`.
- NeuroBE: `h_dim = _nArgs * var_dim = scope_size * 3`.
- Difference: `3 * (scope_size - 1)` vs `3 * scope_size` — off by 3.
- For neurobe_mode, add `'neurobe,3'` hidden_sizes mode: `h = scope_size * var_dim` where `scope_size = len(message_scope)` and `var_dim` is parsed from the string. This uses `get_message_scope()` (line 1114) to get scope size, not `get_message_size()`.

### Loss Functions (`nce/neural_networks/losses.py`)

- Existing `weighted_logspace_mse` (line 751): Operates on already-normalized logspace targets. Computes its own min/max from the batch — **wrong for neurobe_mode** because normalization must happen before training, not per-batch. NeuroBE's weights depend on the [0,1]-normalized labels and pre-computed `ln_max`, `ln_min`, `sum_ln` from training data.

- NeuroBE's weight formula (Function-NN.hxx line 219):
  ```cpp
  auto w = ((labels*(ln_max_value - ln_min_value)) / (sum_ln));
  w_loss = (w*((output - labels).pow(2))).mean();
  ```
  Where `labels` are [0,1]-normalized. Expanding: `w_i = (normalized_label_i * (ln_max - ln_min)) / sum_ln`. Since `normalized_label_i = (y_i - ln_min)/(ln_max - ln_min)`, the weight simplifies to `w_i = (y_i - ln_min) / sum_ln`. This is the proportion of shifted raw value.

- `sum_ln` in NeuroBE (for `net` type, Function-NN.hxx line 901): `sum_ln = sum(arr[i] - ln_min)` over all training samples. So `sum_ln = sum(y_ln_i - ln_min)` — the sum of shifted values.

### Config Schema (`nce/config_schema.py`)

- `prepare_config()` at line 252: Entry point. Currently handles flat/nested detection, alias resolution, validation, defaults. The neurobe_mode expansion should go after flattening/validation, before return — detect `neurobe_mode` key and set defaults for all neurobe-faithful fields. Fields explicitly set by user should NOT be overridden.

- Pattern for expansion: iterate neurobe defaults dict, set each key in flat config only if not already present (i.e., user override wins).

## Constraints

- **Binary domains only** for this slice. All 15 NeuroBE benchmark problems are binary (domain size 2). The normalization, loss function, and hidden dimension formulas are for the `net` architecture path, not `masked_net`.

- **Log10 internal representation.** NCE factors store values in log10. NeuroBE's normalization operates on raw (natural log) values. The conversion boundary is in DataPreprocessor: `y_ln = y_log10 * ln(10)`. Min-max normalization is applied after this conversion.

- **NeuroBE's `sum_ln` is computed from training data only**, not validation. For `sampling_scheme='all'` (which neurobe_mode should use for binary problems where message_size is small), all data is training data, so this is automatically correct. For sampled data, the first training batch initializes the constants.

- **Validation set construction**: NeuroBE uses 80/20 train/val split of pseudo-dimension-based samples. NCE currently uses `num_samples // 9` for validation. Neurobe_mode should document this as a known difference — the exact split ratio matters less than the normalization and loss function.

- **No backward messages in neurobe_mode.** `use_bw_approx=False`, `populate_bw_factors=False`. The backward message code paths should be completely bypassed.

## Common Pitfalls

- **Division by zero in min-max normalization.** If all targets in a bucket are identical (`ln_max == ln_min`), normalization divides by zero. Guard with epsilon: `range = max(ln_max - ln_min, 1e-10)`. When range is zero, set all normalized values to 0.5 (midpoint) and weights to uniform.

- **Denormalization log-base mismatch.** The formula `ln_min + nn_out * (ln_max - ln_min)` produces a value in **natural log** space. Must divide by `ln(10)` to get back to log10 for NCE's factor system. Missing this division produces values that are `ln(10) ≈ 2.3026` times too large.

- **Weight formula depends on [0,1]-normalized labels.** The NeuroBE loss uses `w = labels * (ln_max - ln_min) / sum_ln` where `labels` are already [0,1]-normalized. If you pass un-normalized targets to the loss function, the weights are wrong. The loss function must receive normalized targets.

- **Early stopping: 3 non-improving epochs, not 2.** NeuroBE's `count > stop_iter` with `stop_iter=2` breaks after count reaches 3 (not 2). The counter increments **after** comparing, so it goes: fail → count=1, fail → count=2, fail → count=3 → 3>2 → break. Implementing `count >= stop_iter` would stop one epoch too early.

- **`sum_ln` computation for `net` vs `masked_net`.** For `net` type (binary domain problems), NeuroBE computes `sum_ln = sum(arr[i] - ln_min)`. For `masked_net` type, it computes `sum_ln = sum(arr[i] / ln_max)` — different formula. This slice only handles the `net` path.

- **Hidden sizes `neurobe,3` vs `nbe,3`.** The existing `nbe,3` mode in `bucket.py` computes `h = 3 * ceil(log2(message_size))`. The new `neurobe,3` mode must compute `h = scope_size * 3` where `scope_size = len(get_message_scope())`. Off by 3 for binary problems. Using the wrong mode silently produces a different-capacity network.

## Open Risks

- **Xavier init vs Kaiming init.** NCE uses `xavier_normal_` initialization; NeuroBE uses PyTorch's default (`kaiming_uniform_`). Different init can affect convergence behavior for the same hyperparameters. Accepted as a known difference — documenting but not changing.

- **Validation set size mismatch.** NeuroBE uses 80/20 split of pseudo-dimension samples; NCE uses `num_samples // 9`. Different validation set sizes could cause different early stopping behavior. Accepted as known difference for now; can be tightened in S02 if results diverge.

- **Batch loss computation differences.** NeuroBE averages MSE per batch then averages across batches. NCE aggregates differently depending on loss type (logsumexp for logspace, sum for linear). For neurobe_weighted_mse, need to use per-batch mean then epoch-level mean to match NeuroBE.

## Implementation Notes

### neurobe_mode Expansion Defaults

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
    'skip_early_stopping': True,  # Disable NCE's default early stopping
    'nbe_early_stopping': False,  # Disable the 3-consecutive-increases mode
    'lower_dim': True,
    'sampling_scheme': 'all',
    'iB': 25,
}
```

### Test Plan

1. **test_normalization_round_trip** — Create known log10 values, run through DataPreprocessor with minmax_01 mode, verify normalized values are in [0,1], undo_normalization recovers originals within 1e-6.

2. **test_early_stopping_patience** — Feed Trainer a mock loss sequence `[1.0, 0.9, 0.8, 0.85, 0.86, 0.87]` (3 non-improving after best=0.8). Verify training stops at epoch with count > stop_iter.

3. **test_neurobe_config_expansion** — Call `prepare_config({'neurobe_mode': True, ...required fields...})` and verify all NEUROBE_DEFAULTS are present. Also test that explicit overrides win: `{'neurobe_mode': True, 'lr': 0.01}` → lr=0.01.

4. **test_neurobe_weighted_mse** — Hand-compute expected loss for known inputs and verify function output matches.

## Skills Discovered

| Technology | Skill | Status |
|------------|-------|--------|
| PyTorch | — | none relevant — domain-specific NN training |
| Python testing | — | none relevant — standard pytest patterns |

No relevant skills for this slice. All work is domain-specific extensions to existing modules.

## Sources

- NeuroBE `Function-NN.hxx` Train() function (lines 157-310) — normalization, IS weights, early stopping counter logic
- NeuroBE `Function-NN.hxx` log_sum_exp() (lines 883-908) — ln_min, ln_max, sum_ln computation for `net` type
- NeuroBE `Function-NN.hxx` samples_to_data() (lines 912-926) — [0,1] normalization formula
- NeuroBE `Net.h` (lines 12-33) — ReLU activation, 2 hidden layers, h_dim = _nArgs × var_dim
- NeuroBE `HYPERPARAMETERS.md` — stop_iter=2, batch_size=256, lr=0.001 (hardcoded bug), n_epochs=500
- NCE `data_preprocessor.py` — current log-space mean normalization pipeline
- NCE `train.py` — current early stopping modes, AMP usage (only for muon optimizer)
- NCE `net.py` — current Tanh activation, Xavier init
- NCE `bucket.py` lines 210-222 — current `nbe,3` hidden size computation
- NCE `config_schema.py` — prepare_config pattern for field expansion
- NCE `tests/conftest.py` — existing test fixtures and config patterns
- NCE `tests/PATTERN.md` — test suite conventions (R024)

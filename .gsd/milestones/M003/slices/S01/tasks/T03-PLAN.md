---
estimated_steps: 4
estimated_files: 3
---

# T03: Add Net activation config and neurobe,3 hidden sizes mode

**Slice:** S01 — NeuroBE Training Mode
**Milestone:** M003

## Description

Two small, isolated changes for NeuroBE-faithful reproduction. NeuroBE uses ReLU activation (NCE defaults to Tanh) and computes hidden layer dimension as `scope_size * var_dim` (NCE's existing `nbe,3` uses `ceil(log2(message_size)) * var_dim`, which differs by one factor for binary problems). Also registers all new config fields in the schema so that `prepare_config` accepts neurobe-mode configs without raising "unknown field" errors.

## Steps

1. In `Net.__init__` (net.py), read `activation` from `nn_config` (default `'tanh'`). Replace the hardcoded `nn.Tanh()` in the layer construction loop with a conditional: `nn.ReLU()` if `activation == 'relu'`, else `nn.Tanh()`. The activation function object should be created once before the loop and reused (or recreated per layer — PyTorch modules are cheap).

2. In `bucket.py`, add `neurobe` hidden sizes parsing alongside the existing `nbe` path (around line 210). When `hidden_sizes` is a string starting with `'neurobe'`: parse multiplier `b` from `'neurobe,{b}'` (default 1), compute `h = len(self.get_message_scope()) * b`, set `hidden_sizes = [h, h]`. This uses scope size (number of variables) rather than `ceil(log2(message_size))`.

3. Add new fields to `NESTED_SECTIONS` in `config_schema.py`:
   - `nn` section: `'activation': _field('activation', default='tanh')`
   - `training` section: `'normalization_mode': _field('normalization_mode', default='logspace_mean')`, `'neurobe_early_stopping': _field('neurobe_early_stopping', default=False)` (note: this is a NEW field distinct from the existing `nbe_early_stopping`), `'neurobe_stop_iter': _field('neurobe_stop_iter', default=2)`, `'use_amp': _field('use_amp', default=True)`
   - `inference` section: `'neurobe_mode': _field('neurobe_mode', default=False)`

4. Verify existing tests still pass — the new fields all have defaults that preserve existing behavior (activation='tanh', normalization_mode='logspace_mean', neurobe_mode=False).

## Must-Haves

- [ ] `Net` with `activation='relu'` config uses ReLU layers; `activation='tanh'` (or unset) uses Tanh
- [ ] `neurobe,3` hidden sizes in bucket.py computes `h = scope_size * 3` (distinct from `nbe,3` which computes `h = 3 * ceil(log2(message_size))`)
- [ ] Config schema accepts `activation`, `normalization_mode`, `neurobe_early_stopping`, `neurobe_stop_iter`, `use_amp`, `neurobe_mode` without raising errors
- [ ] All 125 existing tests pass with no changes to their configs

## Verification

- `source venv/bin/activate && python -m pytest tests/ -v --tb=short --ignore=tests/test_neurobe_mode.py` — existing 125 tests pass
- Quick manual verification: `python -c "from nce.config_schema import prepare_config; c = prepare_config({'neurobe_mode': True, 'loss_fn': 'logspace_mse_fdb', 'num_epochs': 10, 'num_samples': 100, 'iB': 5, 'ecl': 8, 'device': 'cpu'}); print('neurobe_mode' in c)"` — prints True (field passes through)

## Observability Impact

- Signals added/changed: None (structural changes only — no new runtime signals)
- How a future agent inspects this: Read `nn_config['activation']` from any bucket's config; inspect `hidden_sizes` computation in bucket.py debug output
- Failure state exposed: None new

## Inputs

- `nce/neural_networks/net.py` — current Tanh-hardcoded layer construction
- `nce/inference/bucket.py` — current `nbe,{b}` hidden sizes parsing (lines 210-222)
- `nce/config_schema.py` — NESTED_SECTIONS schema definition
- T02 output: DataPreprocessor and loss function are in place

## Expected Output

- `nce/neural_networks/net.py` — activation function configurable via `activation` field
- `nce/inference/bucket.py` — `neurobe,{b}` hidden sizes mode alongside `nbe,{b}`
- `nce/config_schema.py` — 6 new fields registered in NESTED_SECTIONS

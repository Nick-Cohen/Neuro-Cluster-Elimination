---
id: T03
parent: S01
milestone: M003
provides:
  - Configurable activation function in Net (relu/tanh via nn_config['activation'])
  - neurobe,{b} hidden sizes mode in bucket.py (scope_size * b)
  - 6 new config schema fields registered (activation, normalization_mode, neurobe_early_stopping, neurobe_stop_iter, use_amp, neurobe_mode)
key_files:
  - nce/neural_networks/net.py
  - nce/inference/bucket.py
  - nce/config_schema.py
  - docs/config_reference.md
key_decisions:
  - neurobe check ordered before nbe in hidden_sizes parsing since 'neurobe'.startswith('nbe') is False but explicit ordering prevents future confusion
  - activation_cls pattern creates the class once before loop, instantiates per layer (PyTorch modules are cheap, one instance per layer is idiomatic)
patterns_established:
  - String-prefix dispatch for hidden_sizes: neurobe,{b} → scope_size * b, nbe,{b} → ceil(log2(message_size)) * b
observability_surfaces:
  - nn_config['activation'] readable from any bucket's config to verify activation mode
  - hidden_sizes computation visible in bucket debug output
duration: ~10min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T03: Add Net activation config and neurobe,3 hidden sizes mode

**Net supports configurable ReLU/Tanh activation, bucket.py parses `neurobe,{b}` hidden sizes, and 6 new config fields pass through schema validation.**

## What Happened

Three isolated changes implemented as planned:

1. **Net activation config** (net.py): Reads `activation` from `nn_config` (default `'tanh'`). Selects `nn.ReLU` or `nn.Tanh` class before the layer loop, instantiates per hidden layer.

2. **neurobe,{b} hidden sizes** (bucket.py): Added parsing before the existing `nbe` path. `neurobe,3` computes `h = len(self.get_message_scope()) * 3` — uses scope size (number of variables) rather than `ceil(log2(message_size))`. Two hidden layers of `[h, h]`.

3. **Config schema fields** (config_schema.py): Added `neurobe_mode` to inference section, `activation` to nn section, `normalization_mode`/`neurobe_early_stopping`/`neurobe_stop_iter`/`use_amp` to training section. All have backward-compatible defaults.

4. **Doc sync** (config_reference.md): Added all 6 fields plus `neurobe,{b}` entry in hidden_sizes table to keep the schema-doc sync test passing.

## Verification

- `pytest tests/ -v --tb=short --ignore=tests/test_neurobe_mode.py` → **125 passed** (all existing tests, including doc sync)
- `prepare_config({'neurobe_mode': True, ...})` → `neurobe_mode` in result = `True` (field accepted)
- `pytest tests/test_neurobe_mode.py -v --tb=short` → **8 of 9 pass** (1 failure is `test_neurobe_mode_expands_all_defaults` which requires config expansion logic not yet implemented — expected for T03)

## Diagnostics

- Read `nn_config['activation']` from any bucket's config to check activation mode
- Inspect Net layer types: `[type(m).__name__ for m in net.network.modules()]` shows ReLU or Tanh
- hidden_sizes string parsing testable via bucket debug output or direct `config['hidden_sizes']` inspection

## Deviations

Added documentation updates to `docs/config_reference.md` — not in the task plan but required by the existing `test_config_docs.py::test_every_schema_field_documented` sync test.

## Known Issues

None.

## Files Created/Modified

- `nce/neural_networks/net.py` — activation function configurable via `nn_config['activation']`
- `nce/inference/bucket.py` — `neurobe,{b}` hidden sizes parsing before `nbe,{b}` path
- `nce/config_schema.py` — 6 new fields in NESTED_SECTIONS (neurobe_mode, activation, normalization_mode, neurobe_early_stopping, neurobe_stop_iter, use_amp)
- `docs/config_reference.md` — documented all 6 new fields and neurobe,{b} hidden_sizes variant

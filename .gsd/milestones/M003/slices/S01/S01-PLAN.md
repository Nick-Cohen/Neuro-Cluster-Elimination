# S01: NeuroBE Training Mode

**Goal:** NCE can train neural networks using NeuroBE-faithful settings: min-max [0,1] normalization, patience-based early stopping, ReLU activation, weighted MSE loss, and a `neurobe_mode: true` config preset that activates all of the above.
**Demo:** `pytest tests/test_neurobe_mode.py -v` passes — normalization round-trip, early stopping trigger, config expansion, and loss function correctness all verified.

## Must-Haves

- DataPreprocessor supports `normalization_mode='minmax_01'`: computes `ln_min`, `ln_max`, `sum_ln` from training data; `normalize()` returns [0,1]; `undo_normalization()` maps back to log10 (R033)
- `neurobe_weighted_mse` loss function operating on [0,1]-normalized targets with IS weights `w = labels * (ln_max - ln_min) / sum_ln` (R033, R035)
- Patience-based early stopping: counter resets on improvement, halts when `count > stop_iter` (3 non-improving epochs for `stop_iter=2`) (R034)
- `neurobe_mode: true` in config expands to all NeuroBE-faithful defaults; explicit user overrides take precedence (R035)
- Net supports configurable activation (ReLU vs Tanh via `activation` config field) (R035)
- `neurobe,3` hidden sizes mode computes `h = scope_size * var_dim` (R035)
- Normalization round-trip test: known values → normalize → undo → compare within tolerance (R037)
- All existing M001/M002 tests continue to pass (125 tests)

## Proof Level

- This slice proves: contract
- Real runtime required: no (CPU-only tests with hand-built problems)
- Human/UAT required: no

## Verification

- `source venv/bin/activate && python -m pytest tests/test_neurobe_mode.py -v --tb=short` — all 4+ neurobe-specific tests pass
- `source venv/bin/activate && python -m pytest tests/ -v --tb=short` — full suite (125 existing + new) passes with no regressions

## Observability / Diagnostics

- Runtime signals: Trainer prints `NeuroBE patience early stopping at epoch {N}` when triggered; DataPreprocessor prints `ln_min`, `ln_max`, `sum_ln` during initialization with `normalization_mode='minmax_01'`
- Inspection surfaces: `data_preprocessor.ln_min`, `.ln_max`, `.sum_ln` attributes readable after init; `data_preprocessor.normalization_mode` indicates active mode
- Failure visibility: Division-by-zero in normalization guarded with epsilon and logged; denormalization log-base mismatch would show as ~2.3x factor in round-trip test
- Redaction constraints: none

## Integration Closure

- Upstream surfaces consumed: `DataPreprocessor` (nce/data/data_preprocessor.py), `Trainer._get_loss_fn` dispatch (nce/neural_networks/train.py), `Net.__init__` layer construction (nce/neural_networks/net.py), hidden sizes parsing in `bucket.py`, `prepare_config()` in config_schema.py, test fixtures in `tests/conftest.py`
- New wiring introduced in this slice: `normalization_mode` parameter threading from config → DataPreprocessor → normalize/undo; `neurobe_weighted_mse` registered in `_get_loss_fn`; `neurobe_mode` expansion block in `prepare_config()`; `activation` config field consumed in Net; `neurobe,3` hidden sizes mode in bucket.py; patience-based early stopping branch in Trainer.train()
- What remains before the milestone is truly usable end-to-end: S02 — per-problem ecl tuning, running 15 problems, comparison table against NeuroBE C++ results

## Tasks

- [x] **T01: Create test file and neurobe config fixture (initially failing)** `est:30m`
  - Why: Define objective stopping conditions before any implementation. Tests document the contract for normalization round-trip, early stopping semantics, config expansion, and loss function correctness.
  - Files: `tests/test_neurobe_mode.py`, `tests/conftest.py`
  - Do: Write 4 test classes: `TestNormalizationRoundTrip` (known log10 values → minmax_01 normalize → undo → compare within 1e-6), `TestNeurobeEarlyStoppingPatience` (mock loss sequence → verify stops at correct epoch), `TestNeurobeConfigExpansion` (neurobe_mode=True → verify all NEUROBE_DEFAULTS), `TestNeurobeWeightedMSE` (hand-computed expected loss). Add `neurobe_training_config` fixture to conftest.py.
  - Verify: `python -m pytest tests/test_neurobe_mode.py -v` — tests exist and fail with ImportError/AttributeError (not syntax errors)
  - Done when: All 4 test classes are syntactically valid and importable; they fail because the implementations don't exist yet

- [ ] **T02: Implement DataPreprocessor minmax_01 mode and neurobe_weighted_mse loss** `est:1h`
  - Why: The normalization pipeline and loss function are the highest-risk coupled pair — the loss depends on preprocessor stats (ln_min, ln_max, sum_ln), and getting the log-base conversion wrong is silent. Building them together ensures the interface is correct.
  - Files: `nce/data/data_preprocessor.py`, `nce/neural_networks/losses.py`, `nce/neural_networks/train.py`
  - Do: Add `normalization_mode` param to DataPreprocessor.__init__. When `'minmax_01'`: compute `ln_min`, `ln_max`, `sum_ln` in `_initialize_normalizing_constant`; `normalize()` returns `(y_ln - ln_min) / (ln_max - ln_min)` with epsilon guard; `undo_normalization()` computes `(ln_min + output * (ln_max - ln_min)) / ln10`. Add `neurobe_weighted_mse` in losses.py. Register in `_get_loss_fn` dispatch with closure capturing preprocessor stats.
  - Verify: `python -m pytest tests/test_neurobe_mode.py::TestNormalizationRoundTrip tests/test_neurobe_mode.py::TestNeurobeWeightedMSE -v` — both pass
  - Done when: Round-trip test passes within 1e-6 tolerance; loss function output matches hand-computed expected value

- [ ] **T03: Add Net activation config and neurobe,3 hidden sizes mode** `est:30m`
  - Why: NeuroBE uses ReLU (not Tanh) and computes hidden dim as `scope_size * var_dim` (not `ceil(log2(message_size)) * var_dim`). Both are small, isolated changes needed for faithful reproduction.
  - Files: `nce/neural_networks/net.py`, `nce/inference/bucket.py`, `nce/config_schema.py`
  - Do: In Net.__init__, read `activation` from config (default `'tanh'`), use `nn.ReLU()` when `'relu'`. In bucket.py, add `neurobe,{b}` hidden sizes path: `h = len(get_message_scope()) * b`. Add `activation` field to nn section and `normalization_mode`, `neurobe_early_stopping`, `neurobe_stop_iter`, `neurobe_mode`, `use_amp` fields to config schema NESTED_SECTIONS.
  - Verify: `python -m pytest tests/ -v --tb=short -k "not neurobe_mode"` — existing 125 tests still pass; manual check that `Net` with `activation='relu'` uses ReLU layers
  - Done when: Config schema accepts all new fields; Net respects activation config; bucket.py parses `neurobe,3` correctly

- [ ] **T04: Implement neurobe_mode config expansion, patience-based early stopping, and pass all tests** `est:1h`
  - Why: This is the closer — wires neurobe_mode expansion in prepare_config, implements patience-based early stopping in Trainer, and verifies all tests pass including the full existing suite.
  - Files: `nce/config_schema.py`, `nce/neural_networks/train.py`, `tests/test_neurobe_mode.py`
  - Do: Add NEUROBE_DEFAULTS dict and expansion block in `prepare_config()` — when `neurobe_mode` is True, set defaults for all neurobe-faithful fields (user overrides win). Add patience-based early stopping branch in Trainer.train(): config fields `neurobe_early_stopping`, `neurobe_stop_iter`; counter starts 0, increments when `loss >= prev_best`, resets on improvement, breaks when `count > stop_iter`. Fix any test assertions that need adjustment based on actual implementation behavior.
  - Verify: `python -m pytest tests/ -v --tb=short` — full suite passes (125 existing + 4+ new neurobe tests)
  - Done when: All tests green; `neurobe_mode: true` expands to correct defaults; patience early stopping triggers at correct epoch

## Files Likely Touched

- `tests/test_neurobe_mode.py` (new)
- `tests/conftest.py`
- `nce/data/data_preprocessor.py`
- `nce/neural_networks/losses.py`
- `nce/neural_networks/train.py`
- `nce/neural_networks/net.py`
- `nce/inference/bucket.py`
- `nce/config_schema.py`

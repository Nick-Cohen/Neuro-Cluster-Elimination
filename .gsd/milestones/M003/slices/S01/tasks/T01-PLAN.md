---
estimated_steps: 5
estimated_files: 2
---

# T01: Create test file and neurobe config fixture (initially failing)

**Slice:** S01 — NeuroBE Training Mode
**Milestone:** M003

## Description

Define all test classes for the slice before implementing any production code. This establishes objective stopping conditions: normalization round-trip correctness (R037), early stopping trigger semantics (R034), config expansion completeness (R035), and loss function numerical correctness (R033). Tests should be syntactically valid and importable, but fail because the implementations don't exist yet.

## Steps

1. Add `neurobe_training_config` fixture to `tests/conftest.py` — copy from `nn_training_config` and override with neurobe-mode fields: `normalization_mode='minmax_01'`, `loss_fn='neurobe_weighted_mse'`, `neurobe_early_stopping=True`, `neurobe_stop_iter=2`, `activation='relu'`, `use_amp=False`, `hidden_sizes='neurobe,3'`, `sampling_scheme='all'`, `lower_dim=True`, `skip_early_stopping=True`, `nbe_early_stopping=False`. Don't run through `prepare_config` since the new fields aren't registered yet — build the raw dict directly.

2. Create `tests/test_neurobe_mode.py` with `TestNormalizationRoundTrip` class:
   - Test method creates known log10 values (e.g., `[0.0, 0.5, 1.0, 1.5, 2.0]` as a tensor), instantiates `DataPreprocessor(normalization_mode='minmax_01')`, calls `normalize()`, asserts all normalized values are in [0, 1], calls `undo_normalization()`, asserts recovered values match originals within 1e-6.
   - Second test method verifies the edge case: all-identical targets don't cause division by zero.

3. Add `TestNeurobeEarlyStoppingPatience` class:
   - The test doesn't need to run the full Trainer — it tests the patience counter logic. Create a simulated loss sequence `[1.0, 0.9, 0.8, 0.85, 0.86, 0.87]` where best=0.8 occurs at index 2, then 3 non-improving epochs follow. Verify the counter logic would trigger at the correct point (after 3 non-improving, i.e., count > stop_iter=2). This can test a helper function or test the Trainer directly with a mock. Initially the test will reference `neurobe_early_stopping` config fields that don't exist yet.

4. Add `TestNeurobeConfigExpansion` class:
   - Calls `prepare_config({'neurobe_mode': True, 'loss_fn': 'neurobe_weighted_mse', 'num_epochs': 500, 'num_samples': 1000, 'iB': 25, 'ecl': 100, 'device': 'cpu'})` and asserts all NEUROBE_DEFAULTS are present in the result.
   - Second test: explicit override wins — `{'neurobe_mode': True, ..., 'lr': 0.01}` → result has lr=0.01 not 0.001.

5. Add `TestNeurobeWeightedMSE` class:
   - Hand-compute expected loss for known inputs: outputs=`[0.3, 0.7]`, labels (normalized)=`[0.25, 0.75]`, ln_min=0.0, ln_max=4.0, sum_ln=3.0. Weights: `w_i = labels_i * (ln_max - ln_min) / sum_ln`. Expected loss = `mean(w * (output - labels)^2)`. Call `neurobe_weighted_mse` and assert output matches within 1e-6.

## Must-Haves

- [ ] `tests/test_neurobe_mode.py` exists with 4 test classes, each with at least one test method
- [ ] `neurobe_training_config` fixture exists in conftest.py
- [ ] All tests are syntactically valid Python (can be collected by pytest)
- [ ] Tests fail with ImportError/AttributeError/KeyError, not SyntaxError

## Verification

- `source venv/bin/activate && python -m pytest tests/test_neurobe_mode.py --collect-only` — all tests are discovered
- `source venv/bin/activate && python -m pytest tests/test_neurobe_mode.py -v` — tests fail (expected, since implementations don't exist)
- `source venv/bin/activate && python -m pytest tests/ -v --tb=short --ignore=tests/test_neurobe_mode.py` — existing 125 tests still pass

## Observability Impact

- Signals added/changed: None (test-only task)
- How a future agent inspects this: `pytest tests/test_neurobe_mode.py --collect-only` lists all planned tests
- Failure state exposed: Test names and assertion messages document the expected contracts

## Inputs

- `tests/conftest.py` — existing fixtures (nn_training_config, star_graph_factors) as templates
- `tests/PATTERN.md` — test conventions to follow
- S01-RESEARCH.md — exact formulas for normalization, loss weights, early stopping counter

## Expected Output

- `tests/test_neurobe_mode.py` — 4 test classes with ~6-8 test methods total, all failing
- `tests/conftest.py` — new `neurobe_training_config` fixture added

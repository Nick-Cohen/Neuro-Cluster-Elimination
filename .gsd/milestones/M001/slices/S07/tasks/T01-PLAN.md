---
estimated_steps: 5
estimated_files: 2
---

# T01: Create regression test script and pytest test

**Slice:** S07 — Regression Verification
**Milestone:** M001

## Description

Create both regression test artifacts that prove R017: config restructuring doesn't change inference behavior. The standalone script (`scripts/regression_test.py`) is for one-command terminal use. The pytest test (`tests/test_regression.py`) is for CI integration. Both verify the same three properties on rbm_20.

## Steps

1. Create `scripts/regression_test.py` following the `verify_logging.py` pattern:
   - Import `prepare_config`, `nbe_sanity_check`, `FastGM`, `torch`
   - Check CUDA availability (exit with skip message if unavailable)
   - Check 1: `prepare_config(flat_cfg) == prepare_config(nested_cfg)` for rbm_20 configs
   - Check 2: Run `FastGM` with flat config (high ecl, no NN) and nested config (high ecl, no NN), compare `log_partition_function` for bitwise equality
   - Check 3: Run `FastGM` with flat config (2 epochs, 500 samples) and nested config (same), compare `log_partition_function` for bitwise equality
   - Print per-check PASS/FAIL with actual values, print summary, exit 0 or 1
   - Copy configs with `dict()` before any modification to avoid mutating shared state

2. Create `tests/test_regression.py` with pytest structure:
   - `test_config_equality` — `prepare_config(flat) == prepare_config(nested)` for rbm_20
   - `test_exact_inference_equality` — same model, high ecl, no NN, partition functions equal
   - `test_nn_inference_equality` — same model, 2 epochs, 500 samples, partition functions equal
   - Mark inference tests with `@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")`
   - Use descriptive assertion messages showing both values on failure

3. For both: use `nbe_sanity_check.problems[3]` for the model, `nbe_sanity_check.configs['nbe'][3]` for flat config, `nbe_sanity_check.configs['nbe_nested'][3]` for nested config

4. For exact-only mode: override `ecl` to a very large value (e.g., `2**30`) in both configs so no buckets use NN

5. For NN mode: override to 2 epochs, 500 samples in both configs for fast runtime (~6s)

## Must-Haves

- [ ] `scripts/regression_test.py` runs standalone with `python scripts/regression_test.py`, exits 0 on success, 1 on failure
- [ ] `tests/test_regression.py` has 3 test methods, all pass under `pytest tests/test_regression.py -v`
- [ ] Both handle CUDA unavailability gracefully (skip, not crash)
- [ ] Bitwise equality assertion on partition function values
- [ ] Configs copied before modification — no mutation of shared benchmark state
- [ ] Actual partition function values printed/shown in failure messages

## Verification

- `python scripts/regression_test.py` exits 0 with all PASS
- `pytest tests/test_regression.py -v` shows 3 tests passed (or 2 skipped + 1 passed if no CUDA)

## Observability Impact

- Signals added/changed: None (test artifacts, not runtime code)
- How a future agent inspects this: run the script or pytest command; output is self-explanatory
- Failure state exposed: assertion messages include flat vs nested partition function values

## Inputs

- `nce/config_schema.py` — `prepare_config()` for config normalization
- `nce/benchmark_problems/nbe_sanity_check.py` — model and config access (flat + nested builders)
- `nce/inference/graphical_model.py` — `FastGM` for inference execution
- `scripts/verify_logging.py` — pattern reference for standalone script structure
- `tests/test_benchmark_configs.py` — pattern reference for pytest structure

## Expected Output

- `scripts/regression_test.py` — standalone regression test script, zero-arg, PASS/FAIL output
- `tests/test_regression.py` — pytest regression test with 3 test methods

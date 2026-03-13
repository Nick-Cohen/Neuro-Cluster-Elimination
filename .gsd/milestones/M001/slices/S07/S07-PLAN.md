# S07: Regression Verification

**Goal:** A one-command regression test proves flat and nested configs produce identical inference results on a reference problem.
**Demo:** Run `python scripts/regression_test.py` — prints PASS/FAIL, exits 0. Run `pytest tests/test_regression.py -v` — all tests pass. Both verify exact-only and NN modes on rbm_20.

## Must-Haves

- Standalone script (`scripts/regression_test.py`) runnable with zero arguments, prints PASS/FAIL, exits 0 on success / 1 on failure
- Pytest test (`tests/test_regression.py`) with separate test cases for config-level equality, exact-mode inference equality, and NN-mode inference equality
- Both use rbm_20 (nbe_sanity_check model index 3) as reference problem
- Config-level check: `prepare_config(flat) == prepare_config(nested)` for the reference model
- Exact-mode check: `FastGM(flat, high ecl).log_partition_function == FastGM(nested, high ecl).log_partition_function`
- NN-mode check: `FastGM(flat, 2 epochs).log_partition_function == FastGM(nested, 2 epochs).log_partition_function`
- CUDA availability check with graceful skip (not crash) when unavailable
- Bitwise equality assertion (not approximate), with documented tolerance fallback strategy

## Proof Level

- This slice proves: integration
- Real runtime required: yes (actual FastGM inference on GPU)
- Human/UAT required: no

## Verification

- `python scripts/regression_test.py` exits with code 0 and prints PASS for all checks
- `pytest tests/test_regression.py -v` — all tests pass
- Deliberately mismatched configs produce FAIL / test failure (failure-path verification)

## Observability / Diagnostics

- Runtime signals: both script and tests print the actual partition function values from flat and nested runs, so mismatches are immediately diagnosable
- Inspection surfaces: standalone script prints per-check PASS/FAIL with details; pytest output shows individual test results
- Failure visibility: on mismatch, the assertion message includes both values (flat result vs nested result) and the config that produced them
- Redaction constraints: none (no secrets involved)

## Integration Closure

- Upstream surfaces consumed: `nce/config_schema.py` → `prepare_config()`, `nce/benchmark_problems/nbe_sanity_check.py` → model + flat/nested configs, `nce/inference/graphical_model.py` → `FastGM`
- New wiring introduced in this slice: none (consumes existing APIs, no new runtime hookups)
- What remains before the milestone is truly usable end-to-end: nothing — S07 is the final slice in M001

## Tasks

- [x] **T01: Create regression test script and pytest test** `est:45m`
  - Why: Delivers both test artifacts that prove R017. The standalone script follows the `scripts/verify_logging.py` pattern. The pytest test follows `tests/test_benchmark_configs.py` patterns.
  - Files: `scripts/regression_test.py`, `tests/test_regression.py`
  - Do: Write standalone script with 3 checks (config equality, exact-mode inference, NN-mode inference) using rbm_20. Write pytest test with 3 test methods covering the same checks. Both must handle CUDA unavailability gracefully. Use 2 epochs / 500 samples for NN mode to keep runtime ~10s total. Assert bitwise equality. Copy configs before use to avoid mutation.
  - Verify: `python scripts/regression_test.py` exits 0; `pytest tests/test_regression.py -v` passes all tests
  - Done when: both commands succeed on GPU with PASS for all checks

- [x] **T02: Verify failure detection and finalize** `est:20m`
  - Why: Proves the tests actually detect regressions (not just vacuously passing). Confirms the test is robust.
  - Files: `scripts/regression_test.py`, `tests/test_regression.py`
  - Do: Run both test artifacts on GPU. Temporarily modify a config value (e.g., change num_epochs from 2 to 3 in one path) and verify the NN test detects the mismatch. Restore the original. Verify final clean run. Check that CUDA-skip logic works by mocking availability if needed.
  - Verify: Modified config produces FAIL/failure; restored config produces PASS; clean `pytest tests/test_regression.py -v` passes
  - Done when: both positive and negative test paths verified, all tests pass on final run

## Files Likely Touched

- `scripts/regression_test.py`
- `tests/test_regression.py`

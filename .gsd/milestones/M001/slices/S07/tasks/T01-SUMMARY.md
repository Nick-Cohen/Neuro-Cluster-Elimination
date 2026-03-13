---
id: T01
parent: S07
milestone: M001
provides:
  - Standalone regression test script (scripts/regression_test.py)
  - Pytest regression test with 3 test methods (tests/test_regression.py)
key_files:
  - scripts/regression_test.py
  - tests/test_regression.py
key_decisions:
  - Used copy.deepcopy for all config copies since nested configs contain mutable sub-dicts that shallow dict() would share
patterns_established:
  - Regression test pattern: standalone script with check()/summarize_and_exit() for terminal use, pytest class with skipif decorators for CI
observability_surfaces:
  - Script prints per-check PASS/FAIL with actual partition function values; pytest assertion messages include both flat and nested values on failure
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Create regression test script and pytest test

**Created standalone script and pytest test proving flat/nested config equivalence through config normalization, exact inference, and NN inference on rbm_20.**

## What Happened

Created `scripts/regression_test.py` following the `verify_logging.py` pattern and `tests/test_regression.py` following the `test_benchmark_configs.py` pattern. Both verify three properties on rbm_20 (nbe_sanity_check index 3):

1. **Config equality:** `prepare_config(flat) == prepare_config(nested)` — 40 keys match
2. **Exact inference:** high ecl (2^30) with no NN training — partition functions bitwise equal (58.5306)
3. **NN inference:** 2 epochs, 500 samples, seed 42 — partition functions bitwise equal (129.8932)

Used `copy.deepcopy` rather than `dict()` for config copies because nested configs contain mutable sub-dicts (`inference`, `training`, `sampling`, etc.) that shallow copy would share across test cases.

## Verification

- `python scripts/regression_test.py` → 3/3 PASS, exit code 0
- `pytest tests/test_regression.py -v` → 3 passed in 20.27s
- Both CUDA inference tests produce bitwise-equal partition functions between flat and nested configs

Slice-level verification status:
- ✅ `python scripts/regression_test.py` exits 0, all PASS
- ✅ `pytest tests/test_regression.py -v` — all 3 tests pass
- ⬜ Deliberately mismatched configs produce FAIL (T02 scope)

## Diagnostics

Run `python scripts/regression_test.py` for terminal output with per-check PASS/FAIL and actual values. Run `pytest tests/test_regression.py -v` for CI integration. On failure, both show the flat and nested partition function values side-by-side.

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `scripts/regression_test.py` — Standalone regression test script, 3 checks, exits 0/1
- `tests/test_regression.py` — Pytest regression test with 3 test methods, CUDA-skip decorators

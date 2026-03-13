---
id: T04
parent: S01
milestone: M002
provides:
  - test_robustness.py with 9 parametrized edge-case tests covering R022, R023
  - tests/PATTERN.md documenting test suite structure and how to add regression tests (R024)
key_files:
  - tests/test_robustness.py
  - tests/PATTERN.md
key_decisions:
  - Used from_logspace_mse instead of weighted_logspace_mse as third parametrized loss function — from_logspace_mse uses the standard (outputs, targets, bw_hat=None) signature while weighted_logspace_mse has the same signature but from_logspace_mse exercises more interesting edge-case behavior (exp/log operations on inf values)
  - Zero-target tests assert finiteness (valid uniform distribution should produce finite loss) while inf/neg-inf tests only assert no crash (result may be inf/nan)
patterns_established:
  - STANDARD_LOSS_FNS list with pytest.param(..., id=name) for parametrizing across loss functions — reuse this pattern when adding new edge-case tests
observability_surfaces:
  - Run pytest tests/test_robustness.py -v -s to see per-function edge-case outcomes (finite/inf/nan documented in print output)
duration: ~10m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T04: Robustness edge-case tests and extensibility pattern

**Added 9 parametrized loss function edge-case tests and PATTERN.md guide for adding regression tests.**

## What Happened

Created `tests/test_robustness.py` with three test classes:
- `TestInfInputNoCrash` — 3 tests (R022): `logspace_mse_fdb`, `linspace_mse_fdb`, `from_logspace_mse` each called with `outputs=[inf, 0, -1]`. All pass without exception.
- `TestNegInfTargetsNoCrash` — 3 tests (R023): same functions with `targets=[-inf, -inf, -inf]`. All pass without exception.
- `TestNegInfTargetsNoCrash` — 3 tests: same functions with `targets=[0, 0, 0]`. Asserts finiteness since zero is a valid uniform distribution.

Created `tests/PATTERN.md` with file-per-concern structure documentation, shared fixture reference, step-by-step guide for adding failure-mode regression tests, and a complete test template.

## Verification

- `pytest tests/test_robustness.py -v` — 9/9 passed
- `test -f tests/PATTERN.md` — exists
- `pytest tests/ -v --tb=short` — 125/125 passed in 26s
- `pytest tests/ --co -q | tail -1` — "125 tests collected" (≥125 ✓)
- Runtime 26s (well under 120s limit)

All slice-level verification checks pass:
- ✅ `pytest tests/ -v` — all 125 tests pass (116 existing + 9 new robustness)
- ✅ `pytest tests/test_inference.py -v` — 3/3 pass (R018, R020)
- ✅ `pytest tests/test_nn_training.py -v` — 3/3 pass (R019, R021)
- ✅ `pytest tests/test_robustness.py -v` — 9/9 pass (R022, R023)
- ✅ `test -f tests/PATTERN.md` — exists (R024)
- ✅ `pytest tests/ --co -q | tail -1` — 125 tests (≥125)
- ✅ `timeout 120 pytest tests/` — 26s (under 120s)

## Diagnostics

- Run `pytest tests/test_robustness.py -v -s` to see per-function edge-case outcomes (prints whether result is finite, inf, or nan)
- Parametrized test names include the loss function name, so failures clearly identify which function failed on which edge case

## Deviations

Used `from_logspace_mse` instead of `weighted_logspace_mse` as the third loss function. The plan suggested `weighted_logspace_mse` but `from_logspace_mse` better exercises edge-case behavior (exp/log on inf) and uses the standard signature without extra args.

## Known Issues

None.

## Files Created/Modified

- `tests/test_robustness.py` — 9 parametrized edge-case tests for loss function robustness (R022, R023)
- `tests/PATTERN.md` — Step-by-step guide for adding failure-mode regression tests (R024)

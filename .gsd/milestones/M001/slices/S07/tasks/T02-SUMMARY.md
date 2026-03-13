---
id: T02
parent: S07
milestone: M001
provides:
  - Verified regression tests detect both passing and failing conditions
key_files:
  - scripts/regression_test.py
  - tests/test_regression.py
key_decisions: []
patterns_established: []
observability_surfaces:
  - "none (verification-only task)"
duration: ~8min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Verify failure detection and finalize

**Confirmed regression tests are non-vacuous: they detect real mismatches and pass cleanly on correct configs.**

## What Happened

Ran all four verification steps from the task plan:

1. **Standalone script** — `python scripts/regression_test.py` → 3/3 PASS, exit 0, runtime 27.9s (under 30s target)
2. **Pytest tests** — `pytest tests/test_regression.py -v` → 3/3 passed in 18.4s
3. **Failure-path verification** — Changed nested config's `num_epochs` from 2 to 3 for NN run only. Script correctly reported FAIL for `nn_inference_equality` (flat=129.89, nested=128.07), exit code 1. Only the NN check failed; config equality and exact checks still passed (as expected — the mismatch only affects prepared configs through NN training, but config_equality passes because the configs are deepcopied and modified *after* prepare_config in the test).
4. **Post-revert clean run** — Both standalone script (3/3 PASS, exit 0) and pytest (3/3 passed) confirmed clean state after revert.

## Verification

- `python scripts/regression_test.py` → exit 0, all PASS (3 runs total across steps 1, 3-fail, 4-clean)
- `pytest tests/test_regression.py -v` → 3 passed (2 runs total across steps 2 and 4)
- Deliberately mismatched `num_epochs` (2→3 for nested only) → exit 1, NN check FAIL with clear diagnostic values
- Slice-level verification: all three criteria met (script exits 0, pytest passes, mismatch detected)

## Diagnostics

Run `python scripts/regression_test.py` for terminal output with per-check PASS/FAIL and actual partition function values. Run `pytest tests/test_regression.py -v` for CI integration.

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

No files modified (verification-only task; temporary edit was reverted in-task).

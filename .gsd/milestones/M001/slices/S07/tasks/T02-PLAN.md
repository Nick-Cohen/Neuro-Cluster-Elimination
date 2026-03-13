---
estimated_steps: 4
estimated_files: 2
---

# T02: Verify failure detection and finalize

**Slice:** S07 — Regression Verification
**Milestone:** M001

## Description

Verify that the regression tests actually detect regressions — not just vacuously passing. Run both test artifacts, confirm they pass cleanly, then deliberately introduce a config mismatch to prove they catch it. Restore and do a final clean run.

## Steps

1. Run `python scripts/regression_test.py` on GPU — confirm all 3 checks PASS and exit code is 0

2. Run `pytest tests/test_regression.py -v` — confirm all 3 tests pass

3. Failure-path verification: temporarily modify one config in the script (e.g., change `num_epochs` from 2 to 3 for the nested-config NN run only), run again, confirm the NN check reports FAIL and exit code is 1. Revert the change.

4. Final clean run of both — confirm everything passes after revert

## Must-Haves

- [ ] Both test artifacts pass cleanly on GPU
- [ ] Deliberately mismatched config produces FAIL/test failure (proves tests aren't vacuous)
- [ ] Final clean run passes after revert
- [ ] Runtime is under 30 seconds total for the standalone script

## Verification

- `python scripts/regression_test.py` → exit 0, all PASS
- `pytest tests/test_regression.py -v` → all pass
- Modified config → exit 1 or test failure (failure detected)

## Observability Impact

- Signals added/changed: None
- How a future agent inspects this: run the test commands
- Failure state exposed: None (verification task, no new code)

## Inputs

- `scripts/regression_test.py` — from T01
- `tests/test_regression.py` — from T01

## Expected Output

- Verified test artifacts that detect both passing and failing conditions
- No file changes (this is a verification-only task, unless minor fixes needed from T01)

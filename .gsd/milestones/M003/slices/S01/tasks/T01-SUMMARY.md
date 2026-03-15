---
id: T01
parent: S01
milestone: M003
provides:
  - tests/test_neurobe_mode.py with 4 test classes (9 methods) defining neurobe-mode contracts
  - neurobe_training_config fixture in tests/conftest.py
key_files:
  - tests/test_neurobe_mode.py
  - tests/conftest.py
key_decisions:
  - Raw dict fixture (not validated through prepare_config) since neurobe fields aren't in schema yet
  - Pure counter logic tests pass immediately (no external dependency); implementation-dependent tests fail as expected
patterns_established:
  - Neurobe test classes map 1:1 to requirements: R037→NormalizationRoundTrip, R034→EarlyStoppingPatience, R035→ConfigExpansion, R033→WeightedMSE
observability_surfaces:
  - none (test-only task)
duration: 15m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Created neurobe-mode test file and config fixture (initially failing)

**9 test methods across 4 classes defining all neurobe-mode contracts; 5 fail as expected (implementations don't exist yet), 4 pass (pure logic or trivial config reads).**

## What Happened

Added `neurobe_training_config` fixture to conftest.py — raw dict based on `nn_training_config` with neurobe-mode overrides (minmax_01 normalization, neurobe_weighted_mse loss, patience early stopping, relu activation). Not passed through `prepare_config` since the new fields aren't registered yet.

Created `tests/test_neurobe_mode.py` with 4 test classes:
- `TestNormalizationRoundTrip` (2 tests): round-trip known log10 values, identical-targets edge case
- `TestNeurobeEarlyStoppingPatience` (3 tests): counter triggers at correct epoch, counter resets on improvement, config field accessibility
- `TestNeurobeConfigExpansion` (2 tests): all NEUROBE_DEFAULTS present after expansion, explicit override wins
- `TestNeurobeWeightedMSE` (2 tests): hand-computed loss matches function output, zero-label produces zero weight

## Verification

- `pytest tests/test_neurobe_mode.py --collect-only` — 9 tests collected across 4 classes ✅
- `pytest tests/test_neurobe_mode.py -v` — 5 fail (TypeError, AssertionError, ImportError), 4 pass ✅
- No SyntaxError failures ✅
- `pytest tests/ --ignore=tests/test_neurobe_mode.py` — 125 existing tests pass, no regressions ✅

Slice-level verification (partial, expected for T01):
- `pytest tests/test_neurobe_mode.py -v` — 5/9 failing (expected: implementations don't exist yet)
- `pytest tests/ -v --tb=short` — 125 existing pass; 5 neurobe tests fail as expected

## Diagnostics

`pytest tests/test_neurobe_mode.py --collect-only` lists all planned tests. Test names and assertion messages document the expected contracts for each requirement.

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `tests/test_neurobe_mode.py` — New: 4 test classes with 9 methods covering R033, R034, R035, R037
- `tests/conftest.py` — Added `neurobe_training_config` fixture (raw dict, not validated)

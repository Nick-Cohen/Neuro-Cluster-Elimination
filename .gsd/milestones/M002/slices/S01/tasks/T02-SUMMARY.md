---
id: T02
parent: S01
milestone: M002
provides:
  - test_inference.py with 3 exact inference correctness tests (R018, R020)
  - _make_exact_config helper for forcing exact computation path
key_files:
  - tests/test_inference.py
key_decisions:
  - Exact config must override both ecl=2**30 AND iB=100 to guarantee exact path — the process_bucket decision checks both bucket_width<=iB and bucket_ec<=ecl
patterns_established:
  - _make_exact_config(base_config) helper creates a prepare_config-validated config forcing exact inference; reusable for any test needing exact computation
  - TestExactInference class groups all exact-path Z-value tests with a common pattern: build exact config → create FastGM → eliminate_variables(all=True) → assert log_partition_function matches analytic value
observability_surfaces:
  - Assertion messages include computed vs expected log10(Z) values and absolute diff for debugging
duration: ~5min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Exact inference correctness and domain≥3 tests

**Added 3 exact inference tests verifying FastGM produces analytically correct partition functions on binary, ternary, and star graph problems.**

## What Happened

Created `tests/test_inference.py` with a `TestExactInference` class containing three tests:

1. `test_binary_chain_exact_z` — binary chain (domain 2), Z=2.0 (R018)
2. `test_ternary_chain_exact_z` — ternary chain (domain 3), Z=4.5 (R020)
3. `test_star_graph_exact_z` — star graph with 3-way factor, analytic Z (R018)

All use a shared `_make_exact_config` helper that overrides `ecl=2**30` and `iB=100` to force the exact computation path. Initial implementation only set `ecl` — the star graph test failed because `process_bucket` gates on *both* `bucket_width <= iB` and `bucket_ec <= ecl`. With `iB=2` from the base config, the star graph's width-3 message triggered NN training. Setting `iB=100` fixed it.

## Verification

- `pytest tests/test_inference.py -v` — 3/3 passed
- `pytest tests/ -v --tb=short` — 113/113 passed (110 existing + 3 new), no regressions

### Slice-level checks (partial — T02 is intermediate):
- ✅ `pytest tests/test_inference.py -v` — exact inference correctness (R018, R020)
- ⬜ `pytest tests/test_nn_training.py -v` — not yet created (T03)
- ⬜ `pytest tests/test_robustness.py -v` — not yet created (T04)
- ⬜ `test -f tests/PATTERN.md` — not yet created (T05)
- ⬜ Total test count ≥ 125 — currently 113
- ✅ `timeout 120 pytest tests/` — completes in ~29s

## Diagnostics

Run `pytest tests/test_inference.py -v` to see per-test results. Failure messages show both computed and expected log10(Z) with absolute difference.

## Deviations

Added `iB=100` override to `_make_exact_config` — not in the original plan which only specified `ecl=2**30`. Required because `process_bucket` checks both width and ec thresholds.

## Known Issues

None.

## Files Created/Modified

- `tests/test_inference.py` — 3 exact inference tests covering R018 (binary chain, star graph) and R020 (ternary chain)

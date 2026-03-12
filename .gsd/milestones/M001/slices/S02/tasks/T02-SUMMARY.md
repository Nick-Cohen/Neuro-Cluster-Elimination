---
id: T02
parent: S02
milestone: M001
provides:
  - Nested config builders (_build_nbe_nested_configs, _build_default_nested_configs) for both benchmark sets
  - configs['nbe_nested'] and configs['default_nested'] keys on BenchmarkSet instances
  - 58 round-trip equality and strict validation tests for nested configs
key_files:
  - nce/benchmark_problems/nbe_sanity_check.py
  - nce/benchmark_problems/small_problems.py
  - tests/test_benchmark_configs.py
key_decisions:
  - Used readable alias names throughout nested builders (exact_computation_limit, learning_rate, forward_diff_barrier, backward_ecl in backward section, etc.) matching the conftest equivalent_nested_config pattern
  - backward_ecl inside backward section is a live alias mapping to bw_ecl (not a dead field) — consistent with config_schema.py design
patterns_established:
  - Nested builders mirror flat builders field-for-field using readable names; round-trip tests enforce they stay in sync
  - Parametrized test IDs use [nbe-0..4] and [default-0..23] for per-model failure identification
observability_surfaces:
  - pytest parametrize IDs pinpoint which model index fails if flat/nested drift occurs
duration: ~5 minutes
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T02: Add nested config builders with round-trip equality tests

**Added `_build_nbe_nested_configs()` and `_build_default_nested_configs()` with 58 parametrized tests proving round-trip equality with flat builders.**

## What Happened

Added nested config builder functions to both benchmark modules. Each builder produces config dicts using the 6-section schema (inference, nn, training, sampling, backward, output) with readable alias names. Registered them in BenchmarkSet.configs under `'nbe_nested'` (5 configs) and `'default_nested'` (24 configs).

Added two test classes to test_benchmark_configs.py:
- `TestNestedBuilderRoundTrip`: 29 parametrized tests asserting `prepare_config(nested[i]) == prepare_config(flat[i])` for every model in both sets
- `TestNestedBuilderValidation`: 29 parametrized tests asserting each nested config passes `prepare_config(strict=True)`

## Verification

- `python -m pytest tests/test_benchmark_configs.py -v -k nested` — 58 passed, 12 deselected
- `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v` — 98 passed (6 warnings from conftest's dead-field fixture, expected)
- Direct import verification: both `configs['nbe_nested']` and `configs['default_nested']` accessible, correct lengths (5 and 24)
- No `prepare_config()` calls at module level in either benchmark module

## Diagnostics

- Run `pytest tests/test_benchmark_configs.py -v -k nested` — parametrized IDs show `[nbe-0]` through `[nbe-4]` and `[default-0]` through `[default-23]` for per-model failure identification
- `_diff_dicts` helper in test file provides detailed key/value mismatch output on round-trip failures

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/benchmark_problems/nbe_sanity_check.py` — added `_build_nbe_nested_configs()`, registered `'nbe_nested'` in BenchmarkSet
- `nce/benchmark_problems/small_problems.py` — added `_build_default_nested_configs()`, registered `'default_nested'` in BenchmarkSet
- `tests/test_benchmark_configs.py` — added `TestNestedBuilderRoundTrip` and `TestNestedBuilderValidation` test classes with 58 parametrized tests

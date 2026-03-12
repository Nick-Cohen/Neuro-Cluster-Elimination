---
id: T01
parent: S02
milestone: M001
provides:
  - Clean benchmark configs (no dead fields in builder output)
  - set_bw_ecl() writes only live fields (bw_ecl, populate_bw_factors)
  - Test coverage for benchmark config cleanliness
key_files:
  - nce/benchmark_problems/nbe_sanity_check.py
  - nce/benchmark_problems/small_problems.py
  - tests/test_benchmark_configs.py
  - tests/test_config_schema.py
key_decisions:
  - reference_flat_config fixture in conftest.py left with dead fields — it tests general schema stripping behavior, not benchmark config cleanliness
patterns_established:
  - Benchmark config tests use try/except around builder imports to pytest.skip when model catalog is unavailable
observability_surfaces:
  - prepare_config(config, strict=True) raises ValueError if dead fields are reintroduced
  - pytest -W error on benchmark configs will surface any dead-field regressions
duration: ~8min
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T01: Remove dead fields from benchmark configs and fix set_bw_ecl

**Removed `backward_ecl` and `num_batches_per_set` from both benchmark config builders; fixed `set_bw_ecl()` to stop writing `backward_ecl`; added 12-test coverage file.**

## What Happened

Removed `backward_ecl` and `num_batches_per_set` lines from `_build_nbe_configs()` (nbe_sanity_check) and `_build_default_configs()` (small_problems). Removed the `cfg['backward_ecl'] = value` line from `set_bw_ecl()`, keeping only `bw_ecl` and `populate_bw_factors`. Updated the docstring accordingly.

Created `tests/test_benchmark_configs.py` with 3 test classes (12 tests total): `TestCleanBenchmarkConfigs` (4 tests verifying dead fields absent from all 5 nbe + 24 default configs), `TestSetBwEcl` (4 tests verifying set_bw_ecl writes only live fields), `TestBenchmarkConfigValidation` (4 tests verifying all configs pass prepare_config with no warnings and in strict mode).

Updated `tests/test_config_schema.py`: changed `TestBenchmarkPassthrough.test_benchmark_config_passthrough` and `TestFastGMIntegration.test_benchmark_config_survives_prepare` comments and assertions to reflect that benchmark configs are now clean (no dead fields to strip, no warnings expected).

## Verification

- `python -m pytest tests/test_benchmark_configs.py -v` — 12/12 passed
- `python -m pytest tests/test_config_schema.py -v` — 28/28 passed
- `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v` — 40/40 passed
- Remaining warnings are from `reference_flat_config` fixture (synthetic, intentional dead fields) — not from benchmark configs

## Diagnostics

- Run `prepare_config(config, strict=True)` on any benchmark config — should not raise
- Run `pytest -W error tests/test_benchmark_configs.py` — no dead-field warnings should surface
- If a dead field is reintroduced in a builder, `TestCleanBenchmarkConfigs` will fail naming the field

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `nce/benchmark_problems/nbe_sanity_check.py` — removed `backward_ecl` and `num_batches_per_set` from `_build_nbe_configs()`
- `nce/benchmark_problems/small_problems.py` — removed `backward_ecl` and `num_batches_per_set` from `_build_default_configs()`; removed `backward_ecl` write from `set_bw_ecl()`
- `tests/test_benchmark_configs.py` — new test file (12 tests across 3 classes)
- `tests/test_config_schema.py` — updated 2 benchmark-referencing tests for clean configs (no more stripping assertions)

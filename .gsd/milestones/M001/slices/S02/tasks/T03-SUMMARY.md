---
id: T03
parent: S02
milestone: M001
provides:
  - Worker build_nn_config() output free of dead fields (backward_ecl, num_batches_per_set removed)
  - experiment_config.py defaults no longer include num_batches_per_set
  - build_nn_config() calls prepare_config() at build time for dead-field detection
  - TestWorkerConfigClean test class (5 tests) proving worker config cleanliness
key_files:
  - notebooks/_1-2026/worker.py
  - notebooks/_1-2026/experiment_config.py
  - tests/test_benchmark_configs.py
key_decisions:
  - Used prepare_config(nn_config) with strict=False (default) inside build_nn_config() — the worker config contains extra fields (error_tracking, track_errors) consumed by FastGM separately that aren't in the nn_config schema; strict mode would reject those
  - Lazy import of prepare_config inside build_nn_config() to avoid module-level dependency on nce from the notebooks worker
patterns_established:
  - Worker config tests use importlib + sys.path manipulation to import from notebooks/_1-2026/worker.py
observability_surfaces:
  - build_nn_config() now calls prepare_config() internally — any dead field reintroduced in config assembly will emit UserWarning at build time
duration: 10m
verification_result: passed
completed_at: 2026-03-12
blocker_discovered: false
---

# T03: Wire experiment_config worker to validate through config_schema

**Removed dead fields from worker's build_nn_config() and experiment_config defaults; added prepare_config() validation call and 5-test coverage.**

## What Happened

Removed `backward_ecl` and `num_batches_per_set` from `build_nn_config()` in worker.py. Removed `num_batches_per_set` from `validate_config()` defaults in experiment_config.py. Restructured `build_nn_config()` from a bare `return { ... }` to build-then-validate-then-return pattern with a lazy `prepare_config()` call. Added `TestWorkerConfigClean` class with 5 tests covering dead field absence, warning-free validation, and correct bw_ecl=0 / bw_ecl>0 behavior.

## Verification

- `python -m pytest tests/test_benchmark_configs.py -v -k worker` — 5/5 passed
- `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v` — 103/103 passed (full slice-level verification)
- 6 warnings from test_config_schema.py are expected — they come from the reference_flat_config fixture that intentionally retains dead fields (per T01 decision)

## Diagnostics

- If a dead field is reintroduced in worker.py's config assembly, `prepare_config()` inside `build_nn_config()` will emit `UserWarning` naming the field
- Run `python -m pytest tests/test_benchmark_configs.py -v -k worker` to verify worker config cleanliness
- The `test_prepare_config_no_warnings` test catches any dead-field warnings via `warnings.catch_warnings(record=True)`

## Deviations

None.

## Known Issues

None.

## Files Created/Modified

- `notebooks/_1-2026/worker.py` — removed backward_ecl and num_batches_per_set from build_nn_config(); added prepare_config() validation call with lazy import
- `notebooks/_1-2026/experiment_config.py` — removed num_batches_per_set from validate_config() defaults
- `tests/test_benchmark_configs.py` — added TestWorkerConfigClean class (5 tests) and Path import

---
estimated_steps: 5
estimated_files: 4
---

# T03: Wire experiment_config worker to validate through config_schema

**Slice:** S02 — Benchmark Config Migration
**Milestone:** M001

## Description

Remove dead fields from `worker.py`'s `build_nn_config()`, remove `num_batches_per_set` from `experiment_config.py` defaults, and add a `prepare_config()` validation call in the worker so that assembled nn_configs are validated at build time. Add a test proving the worker's config output is clean.

## Steps

1. Edit `notebooks/_1-2026/worker.py`:
   - Remove `'num_batches_per_set': config.get('num_batches_per_set', 1),` line from `build_nn_config()`.
   - Remove `'backward_ecl': bw_ecl if bw_ecl > 0 else 2**10,` line from `build_nn_config()`.
   - Add `from nce.config_schema import prepare_config` at the top of `build_nn_config()` (lazy import to avoid module-level dependency).
   - Add validation call at the end of `build_nn_config()`, before the return: `prepare_config(nn_config)` — this runs with `strict=False` (default), which will warn on any remaining dead fields. Don't replace the return value since the worker config has extra fields (`error_tracking`) that aren't in the nn_config schema and are consumed by FastGM separately.
2. Edit `notebooks/_1-2026/experiment_config.py`:
   - Remove `'num_batches_per_set': 1,` from the `defaults` dict in `validate_config()`.
3. Add test in `tests/test_benchmark_configs.py`:
   - `TestWorkerConfigClean`: import `build_nn_config` from `notebooks/_1-2026/worker.py` (add sys.path manipulation in test). Build a sample config with test values. Assert the output dict does not contain `backward_ecl` or `num_batches_per_set`. Pass the output through `prepare_config()` in a `warnings.catch_warnings(record=True)` block and assert no dead-field warnings.
4. Verify the worker module still imports correctly and `build_nn_config()` produces a valid config.
5. Run full test suite: `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v`.

## Must-Haves

- [ ] `backward_ecl` absent from `build_nn_config()` output
- [ ] `num_batches_per_set` absent from `build_nn_config()` output
- [ ] `num_batches_per_set` absent from `experiment_config.py` defaults
- [ ] `build_nn_config()` calls `prepare_config()` for validation (dead field detection at build time)
- [ ] Test proves worker config is clean (no dead fields, no warnings)

## Verification

- `python -m pytest tests/test_benchmark_configs.py -v -k worker` — worker config test passes
- `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v` — full suite green

## Observability Impact

- Signals added/changed: `build_nn_config()` now calls `prepare_config()` internally, which will emit `UserWarning` if any dead field is accidentally reintroduced in the worker config assembly.
- How a future agent inspects this: If a worker experiment fails at config validation, the `prepare_config()` call inside `build_nn_config()` will surface the error with field name and description.
- Failure state exposed: Dead fields in the assembled nn_config will produce warnings (or errors if `strict=True` is passed) at config build time, before FastGM init.

## Inputs

- `notebooks/_1-2026/worker.py` — current `build_nn_config()` with dead fields
- `notebooks/_1-2026/experiment_config.py` — current `validate_config()` with `num_batches_per_set` in defaults
- `nce/config_schema.py` — `prepare_config()` (S01 deliverable)
- `tests/test_benchmark_configs.py` — T01+T02 test file to extend

## Expected Output

- `notebooks/_1-2026/worker.py` — dead fields removed, `prepare_config()` validation call added
- `notebooks/_1-2026/experiment_config.py` — `num_batches_per_set` removed from defaults
- `tests/test_benchmark_configs.py` — `TestWorkerConfigClean` test class added

---
estimated_steps: 5
estimated_files: 5
---

# T01: Remove dead fields from benchmark configs and fix set_bw_ecl

**Slice:** S02 — Benchmark Config Migration
**Milestone:** M001

## Description

Remove the two dead fields (`backward_ecl`, `num_batches_per_set`) from both benchmark config builders and fix `set_bw_ecl()` to stop writing `backward_ecl`. Create the new test file `tests/test_benchmark_configs.py` with assertions for clean configs and absence of dead fields. Update existing tests if they now break due to benchmark configs no longer carrying dead fields.

## Steps

1. Edit `nce/benchmark_problems/nbe_sanity_check.py`: remove `'backward_ecl': None` and `'num_batches_per_set': 1` lines from `_build_nbe_configs()`.
2. Edit `nce/benchmark_problems/small_problems.py`: remove `'backward_ecl': 0` and `'num_batches_per_set': 1` lines from `_build_default_configs()`. In `set_bw_ecl()`, remove `cfg['backward_ecl'] = value` line — keep `cfg['bw_ecl']` and `cfg['populate_bw_factors']`.
3. Create `tests/test_benchmark_configs.py` with:
   - `TestCleanBenchmarkConfigs`: assert `backward_ecl` and `num_batches_per_set` are absent from both `_build_nbe_configs()` and `_build_default_configs()` output.
   - `TestSetBwEcl`: assert `set_bw_ecl()` writes `bw_ecl` and `populate_bw_factors` but NOT `backward_ecl`.
   - `TestBenchmarkConfigValidation`: each flat benchmark config passes `prepare_config()` with no warnings (use `warnings.catch_warnings(record=True)` and assert empty).
4. Review `tests/test_config_schema.py` — the `TestBenchmarkPassthrough` and `TestFastGMIntegration.test_benchmark_config_survives_prepare` tests import `_build_nbe_configs()`. After cleanup, the configs no longer have dead fields, so the "dead fields stripped" assertions should be updated to "dead fields absent" (no stripping needed, no warnings expected). The dead-field-stripping mechanism tests in `TestDeadFields` and `TestFastGMIntegration.test_fastgm_init_strips_dead_fields` use synthetic configs — leave those unchanged.
5. Run `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v` and verify all pass.

## Must-Haves

- [ ] `backward_ecl` absent from `_build_nbe_configs()` output (all 5 configs)
- [ ] `num_batches_per_set` absent from `_build_nbe_configs()` output (all 5 configs)
- [ ] `backward_ecl` absent from `_build_default_configs()` output (all 24 configs)
- [ ] `num_batches_per_set` absent from `_build_default_configs()` output (all 24 configs)
- [ ] `set_bw_ecl()` does not write `backward_ecl` key
- [ ] All benchmark flat configs pass `prepare_config()` without warnings
- [ ] Existing `test_config_schema.py` tests still pass (dead field stripping tests unaffected)

## Verification

- `python -m pytest tests/test_benchmark_configs.py -v` — new tests pass
- `python -m pytest tests/test_config_schema.py -v` — existing tests still pass
- `python -m pytest tests/test_benchmark_configs.py tests/test_config_schema.py -v` — full suite green

## Observability Impact

- Signals added/changed: After this task, benchmark configs no longer trigger `UserWarning` for dead fields when passed through `prepare_config()`. This is the intended behavior — clean configs should be silent.
- How a future agent inspects this: Run `prepare_config(config, strict=True)` on any benchmark config — should not raise. Or run with `-W error` pytest flag — no dead-field warnings should surface.
- Failure state exposed: If a dead field is accidentally reintroduced, `prepare_config(config, strict=True)` will raise `ValueError` naming the field.

## Inputs

- `nce/benchmark_problems/nbe_sanity_check.py` — current flat builder with dead fields at lines ~102 and ~113
- `nce/benchmark_problems/small_problems.py` — current flat builder with dead fields at lines ~129 and ~140; `set_bw_ecl()` at line ~170
- `nce/config_schema.py` — `prepare_config()`, `DEAD_FIELDS` (S01 deliverable, stable)
- `tests/test_config_schema.py` — existing tests that may reference benchmark dead fields

## Expected Output

- `nce/benchmark_problems/nbe_sanity_check.py` — dead field lines removed from `_build_nbe_configs()`
- `nce/benchmark_problems/small_problems.py` — dead field lines removed from `_build_default_configs()`; `backward_ecl` line removed from `set_bw_ecl()`
- `tests/test_benchmark_configs.py` — new test file with ~4 test classes covering clean configs, set_bw_ecl, and validation
- `tests/test_config_schema.py` — minor updates to benchmark-referencing tests (assertions updated for clean configs)
